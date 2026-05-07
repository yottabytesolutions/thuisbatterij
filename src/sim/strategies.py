"""Dispatch-strategieën voor de batterij.

Elke strategie krijgt verbruik, ZP en commodity-prijs per kwartier en geeft
een dispatch-serie terug. Positief = laden (kWh AC), negatief = ontladen.

Alles is heuristisch, behalve `perfect_foresight` en `optimal_lp`. De
heuristiek is geschreven voor leesbaarheid, niet voor optimaal traden.

Heuristiekvolgorde per kwartier:
  1. ZP-overschot opvangen.
  2. Negatieve day-ahead absorberen (gratis energie).
  3. Onbalans-percentiel: laden bij extreem laag, ontladen bij extreem hoog.
  4. Vlakke dag: round-trip-verlies dekt het spread niet, dus stilstaan.
  5. Dag-rank-arbitrage: laden in goedkoopste N kwartieren, ontladen in
     duurste. ZP-bewuste skip voorkomt ochtendladen als ZP de batterij toch
     volgooit.
  6. Adaptieve ontlaaddiepte (saldering): in dure set ontladen naar rato van
     prijsrang, zodat capaciteit voor de echte piek bewaard blijft.
"""


import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from numba import njit

from .battery import BatterySpec, charge_step, discharge_step


# JIT-loop. De meeste rekentijd zit in de per-kwartier-beslissing over 35.040
# buckets per scenario. Numba compileert naar native; eerste call ~5 s warmup,
# daarna ~50× sneller dan pure Python.


@njit(cache=True)
def _dispatch_loop_jit(
    cons_arr: np.ndarray,
    pv_arr: np.ndarray,
    da_arr: np.ndarray,
    override_arr: np.ndarray,
    pct_low_arr: np.ndarray,
    pct_high_arr: np.ndarray,
    is_cheap_arr: np.ndarray,
    is_expensive_arr: np.ndarray,
    spread_arr: np.ndarray,
    top_max_arr: np.ndarray,
    exp_threshold_arr: np.ndarray,
    pv_forecast_arr: np.ndarray,
    hour_arr: np.ndarray,
    usable_kwh: float,
    max_charge_q: float,
    max_discharge_q: float,
    one_way_eff: float,
    min_spread_eur_kwh: float,
    pv_skip_hour_local: int,
    pv_skip_room_factor: float,
    negative_max_soc_frac: float,
    pct_charge_max_soc_frac: float,
    pct_discharge_min_soc_frac: float,
    adaptive_discharge_floor: float,
    allow_grid_export: bool,
    pct_override_grid_export: bool,
) -> np.ndarray:
    """Per-kwartier dispatch met alle beslissingen inline. Werkt op
    numpy-arrays en scalar floats zodat Numba compileert naar native code.
    Heuristiek-volgorde staat in de modulestring."""
    n = cons_arr.shape[0]
    out = np.zeros(n)
    soc = 0.0

    for interval_index in range(n):
        consumption = cons_arr[interval_index]
        pv = pv_arr[interval_index]
        da = da_arr[interval_index]
        override = override_arr[interval_index]
        pct_low = pct_low_arr[interval_index]
        pct_high = pct_high_arr[interval_index]
        is_cheap = is_cheap_arr[interval_index]
        is_expensive = is_expensive_arr[interval_index]
        spread = spread_arr[interval_index]
        top_max = top_max_arr[interval_index]
        exp_threshold = exp_threshold_arr[interval_index]
        pv_forecast = pv_forecast_arr[interval_index]
        hour = hour_arr[interval_index]

        net_load = consumption - pv

        # 1. ZP-overschot altijd opvangen.
        if net_load < 0.0:
            soc, ac = charge_step(soc, -net_load, max_charge_q, usable_kwh, one_way_eff)
            out[interval_index] = ac
            continue

        # 2. Negatieve day-ahead: gratis energie.
        if da < 0.0 and soc < negative_max_soc_frac * usable_kwh:
            soc, ac = charge_step(soc, max_charge_q, max_charge_q, usable_kwh, one_way_eff)
            out[interval_index] = ac
            continue

        # 3. Onbalans-percentiel override.
        if not np.isnan(override):
            if override <= pct_low and soc < pct_charge_max_soc_frac * usable_kwh:
                soc, ac = charge_step(
                    soc, max_charge_q, max_charge_q, usable_kwh, one_way_eff
                )
                out[interval_index] = ac
                continue
            if override >= pct_high and soc > pct_discharge_min_soc_frac * usable_kwh:
                # Bij imbalance-trading: maximaal vermogen, want de bonus op kWh
                # boven net_load betaalt round-trip-verlies plus terugleverpremie.
                # Anders volgt het de allow_grid_export-vlag.
                if pct_override_grid_export:
                    target = max_discharge_q
                else:
                    target = max_discharge_q if allow_grid_export else net_load
                if target > 0.0:
                    soc, ac = discharge_step(soc, target, max_discharge_q, one_way_eff)
                    out[interval_index] = -ac
                    continue

        # 4. Spread te klein om RT-verlies te dekken: stilstaan.
        if spread < min_spread_eur_kwh:
            continue

        # 5. Goedkoop venster: laden vanaf net, tenzij ZP de batterij toch volgooit.
        if is_cheap:
            room = usable_kwh - soc
            if room < 0.0:
                room = 0.0
            is_morning = hour < pv_skip_hour_local
            skip_for_pv = (
                is_morning
                and pv_forecast > pv_skip_room_factor * room
                and room > 0.0
            )
            if not skip_for_pv:
                soc, ac = charge_step(
                    soc, max_charge_q, max_charge_q, usable_kwh, one_way_eff
                )
                out[interval_index] = ac
                continue

        # 6. Duur venster: ontladen met adaptieve diepte.
        if is_expensive:
            if allow_grid_export:
                if top_max > exp_threshold:
                    ramp = (da - exp_threshold) / (top_max - exp_threshold)
                    if ramp < 0.0:
                        ramp = 0.0
                    if ramp > 1.0:
                        ramp = 1.0
                else:
                    ramp = 1.0
                depth = adaptive_discharge_floor + (1.0 - adaptive_discharge_floor) * ramp
                target = depth * max_discharge_q
            else:
                target = net_load
            if target > 0.0:
                soc, ac = discharge_step(soc, target, max_discharge_q, one_way_eff)
                out[interval_index] = -ac
                continue

        # 7. Standaard: ontlaad om netto-belasting te dekken.
        if net_load > 0.0:
            soc, ac = discharge_step(soc, net_load, max_discharge_q, one_way_eff)
            out[interval_index] = -ac

    return out


def _spec_scalars(spec: BatterySpec) -> tuple[float, float, float, float]:
    """Pak een BatterySpec uit naar de scalar-argumenten van de JIT-loop."""
    return (
        spec.usable_kwh,
        spec.max_charge_kwh_per_quarter(),
        spec.max_discharge_kwh_per_quarter(),
        spec.one_way_efficiency,
    )


def no_battery_dispatch(consumption: pd.Series, pv: pd.Series) -> pd.Series:
    return pd.Series(0.0, index=consumption.index, name="battery_ac_kwh")


@njit(cache=True)
def _pv_self_consume_loop(
    cons_arr: np.ndarray,
    pv_arr: np.ndarray,
    usable_kwh: float,
    max_charge_q: float,
    max_discharge_q: float,
    one_way_eff: float,
) -> np.ndarray:
    """JIT-loop voor `pv_self_consume`. ZP-overschot opvangen, verder
    ontladen naar netto-belasting; geen prijsbewustzijn."""
    n = cons_arr.shape[0]
    out = np.zeros(n)
    soc = 0.0
    for i in range(n):
        net_load = cons_arr[i] - pv_arr[i]
        if net_load < 0.0:
            soc, ac = charge_step(
                soc, -net_load, max_charge_q, usable_kwh, one_way_eff
            )
            out[i] = ac
        elif net_load > 0.0:
            soc, ac = discharge_step(soc, net_load, max_discharge_q, one_way_eff)
            out[i] = -ac
    return out


def pv_self_consume(
    consumption: pd.Series,
    pv: pd.Series,
    spec: BatterySpec,
) -> pd.Series:
    """Basisstrategie: batterij vangt alleen ZP-overschot op en ontlaadt naar belasting.

    Geen prijsbewustzijn. Eenvoudigste zelfverbruikmodus.
    """
    usable_kwh, max_charge_q, max_discharge_q, one_way = _spec_scalars(spec)
    out = _pv_self_consume_loop(
        consumption.to_numpy(),
        pv.to_numpy(),
        usable_kwh,
        max_charge_q,
        max_discharge_q,
        one_way,
    )
    return pd.Series(out, index=consumption.index, name="battery_ac_kwh")


@dataclass(frozen=True)
class DispatchTuning:
    """Parameters voor `day_ahead_arbitrage` en `imbalance_aware`.

    Defaults getuned op ENTSO-E NL day-ahead + onbalans (2025-05 tot 2026-05)
    met een 28.7 kWh / 5 kW batterij.
    """

    n_charge: int = 16  # goedkoopste kwartieren per dag voor laden
    n_discharge: int = 16  # duurste kwartieren per dag voor ontladen
    # Onder deze commodity-spread (€/kWh) eet round-trip-verlies de arbitrage op,
    # dus die dag stilstaan. Bij NL-retail met vlakke fee ~€0.18 ligt het
    # break-even rond €0.04. €0.05 op echte ENTSO-E data trimt verlieslatend
    # cyclen op vlakke dagen.
    min_spread_eur_kwh: float = 0.05
    # Onbalans-percentiel override: laden bij extreem laag, ontladen bij extreem
    # hoog. Bij imbalance-trading gaat ontladen naar het net ook post-saldering;
    # de onbalansbonus dekt round-trip-verlies plus terugleverpremie ruim.
    #
    # Getuned op (3.0, 99.5) over een 30-daags venster. NL-onbalans is
    # asymmetrisch: de bovenkant (system Short) heeft zeldzame maar grote
    # positieve delta's, dus 99.5 boekt nog steeds de lucratiefste ontladingen;
    # de onderkant is vlakker, dus 3.0 is nodig voor voldoende laadvolume.
    im_low_percentile: float = 3.0
    im_high_percentile: float = 99.5
    # Venster voor het rollende percentiel, in dagen. 30 dagen volgt
    # seizoensverschuivingen zonder te schokken op weersafhankelijke pieken.
    im_percentile_window_days: int = 30
    # SoC-grenzen voor de percentiel-override.
    pct_charge_max_soc_frac: float = 0.95
    pct_discharge_min_soc_frac: float = 0.05
    # ZP-forecast = lopend 7-daags gemiddeld dagtotaal. Als het boven de lege
    # ruimte uitkomt, slaan we ochtendladen vanaf het net over en laten we ZP
    # gratis vullen.
    pv_forecast_window_days: int = 7
    # Voor dit lokale uur (Europe/Amsterdam) geldt de "skip morning grid-charge
    # als ZP gaat opvullen"-regel. Default = 9 ≈ "vóór 9 uur lokale tijd";
    # cheap-set ligt typisch 02-06 lokaal, dus 9 dekt het volledige venster.
    pv_skip_hour_local: int = 9
    pv_skip_room_factor: float = 1.0
    # Negatieve day-ahead: laden tot SoC-plafond.
    negative_price_charge_max_soc_frac: float = 0.95
    # Adaptieve ontlaaddiepte in het dure venster (saldering): onderkant krijgt
    # deze fractie, top krijgt vol vermogen. Bewaart capaciteit voor de echte piek.
    adaptive_discharge_floor: float = 0.6


@dataclass(frozen=True, slots=True)
class DispatchContext:
    """Voorberekende dispatch-annotaties per kwartier.

    Hergebruikbaar over batterijcapaciteiten heen. Arrays blijven numpy zodat
    de JIT-loop ze zonder kopie kan lezen.
    """

    is_cheap: np.ndarray
    is_expensive: np.ndarray
    spread: np.ndarray
    top_max: np.ndarray
    exp_threshold: np.ndarray
    pv_forecast: np.ndarray
    hour: np.ndarray
    override: np.ndarray
    pct_low: np.ndarray
    pct_high: np.ndarray


_DAY_GROUPING_TZ = "Europe/Amsterdam"


def _local_day(idx: pd.DatetimeIndex) -> np.ndarray:
    """Lokale-tijd dagsleutel (Europe/Amsterdam) voor een UTC-index.

    Day-ahead settled per lokale dag. UTC-grouperen splitst de avondpiek
    in zomertijd over twee buckets.

    Implementatie: tz_localize(None) → datetime64[D] → int64 (dagen sinds
    epoch). Sneller dan year*10000 + month*100 + day omdat we de hele
    int64-buffer in één numpy-cast trunceren.
    """
    local = idx.tz_convert(_DAY_GROUPING_TZ).tz_localize(None)
    return local.values.astype("datetime64[D]").view("int64")


def _local_hour(idx: pd.DatetimeIndex) -> np.ndarray:
    """Lokale-tijd uur (Europe/Amsterdam) voor de morning-skip check.

    Symmetrisch met `_local_day`: de heuristiek redeneert over één
    consistente lokale tijdas, niet half UTC half lokaal.
    """
    return idx.tz_convert(_DAY_GROUPING_TZ).hour.to_numpy(dtype=np.int64)


def _rolling_percentile_thresholds(
    series: pd.Series, window_days: int, low_pct: float, high_pct: float
) -> tuple[np.ndarray, np.ndarray]:
    """Lopende percentielen per kwartier van `series`.

    De eerste `window_days * 96` kwartieren is het percentiel onbepaald.
    Geef ruime sentinels (-inf / +inf) terug zodat de override stilvalt.
    """
    window = window_days * 96
    rolling = series.rolling(window=window, min_periods=window // 2)
    low = rolling.quantile(low_pct / 100.0).to_numpy()
    high = rolling.quantile(high_pct / 100.0).to_numpy()
    # Vervang NaN-warmup met sentinels die nooit triggeren.
    low = np.where(np.isnan(low), -np.inf, low)
    high = np.where(np.isnan(high), np.inf, high)
    return low, high


def _build_dispatch_context(
    da_price: pd.Series,
    pv: pd.Series,
    tune: DispatchTuning,
    *,
    override_price: pd.Series | None,
) -> DispatchContext:
    """Pre-compute van alle per-kwartier en per-dag-annotaties.

    `override_price` is de serie voor de percentiel-override (onbalans bij
    `imbalance_aware`, day-ahead bij `day_ahead_arbitrage`, `None` om uit te zetten).

    Dag-grenzen volgen Europe/Amsterdam (de afrekenmarkt), niet UTC. De
    avondpiek hoort bij dezelfde lokale dag als de ochtenddip; UTC zou dat
    in zomertijd splitsen.
    """
    df = pd.DataFrame({"da": da_price.to_numpy()}, index=da_price.index)
    df["day"] = _local_day(df.index)

    rank = df.groupby("day")["da"].rank(method="first")
    count = df.groupby("day")["da"].transform("count")
    is_cheap = (rank <= tune.n_charge).to_numpy()
    is_expensive = (rank > (count - tune.n_discharge)).to_numpy()

    cheap_means = (
        df[is_cheap].groupby("day")["da"].mean()
        if is_cheap.any()
        else pd.Series(dtype="float64")
    )
    expensive_means = (
        df[is_expensive].groupby("day")["da"].mean()
        if is_expensive.any()
        else pd.Series(dtype="float64")
    )
    spread_per_day = (expensive_means - cheap_means).rename("spread")
    expensive_max = df.groupby("day")["da"].max().rename("top_max")

    day_idx = df["day"]
    spread_arr = day_idx.map(spread_per_day).fillna(0.0).to_numpy()
    top_max_arr = day_idx.map(expensive_max).fillna(0.0).to_numpy()
    expensive_threshold_arr = day_idx.map(expensive_means).fillna(0.0).to_numpy()

    pv_daily = pv.groupby(_local_day(pv.index)).sum()
    pv_forecast = (
        pv_daily.shift(1)
        .rolling(tune.pv_forecast_window_days, min_periods=1)
        .mean()
        .fillna(0.0)
    )
    pv_forecast_arr = day_idx.map(pv_forecast).fillna(0.0).to_numpy()

    if override_price is None:
        override_arr = np.full(len(df), np.nan)
        pct_low_arr = np.full(len(df), -np.inf)
        pct_high_arr = np.full(len(df), np.inf)
    else:
        override_arr = override_price.to_numpy()
        pct_low_arr, pct_high_arr = _rolling_percentile_thresholds(
            override_price,
            tune.im_percentile_window_days,
            tune.im_low_percentile,
            tune.im_high_percentile,
        )

    return DispatchContext(
        is_cheap=is_cheap.astype(np.bool_),
        is_expensive=is_expensive.astype(np.bool_),
        spread=spread_arr,
        top_max=top_max_arr,
        exp_threshold=expensive_threshold_arr,
        pv_forecast=pv_forecast_arr,
        hour=_local_hour(df.index),
        override=override_arr,
        pct_low=pct_low_arr,
        pct_high=pct_high_arr,
    )


def day_ahead_arbitrage(
    consumption: pd.Series,
    pv: pd.Series,
    da_price: pd.Series,
    spec: BatterySpec,
    tune: DispatchTuning | None = None,
    allow_grid_export: bool = True,
    precomputed_context: DispatchContext | None = None,
) -> pd.Series:
    """Dag-rank-arbitrage op day-ahead met ZP-skip, spread-aware idling,
    negatieve-prijs-absorptie en een rolling-percentiel override op day-ahead.
    Zie `DispatchTuning` voor parameters.

    `allow_grid_export=False` (post-saldering): ontladen tot net_load, de
    batterij stuurt geen energie naar het net.

    `precomputed_context` laat een caller (`run_sweep`) de contextbouw uit de
    per-capaciteit-loop hijsen. Prijzen en ZP variëren niet over capaciteiten.
    """
    tune = tune or DispatchTuning()
    ctx = precomputed_context or _build_dispatch_context(
        da_price, pv, tune, override_price=da_price
    )
    usable_kwh, max_charge_q, max_discharge_q, one_way = _spec_scalars(spec)

    out = _dispatch_loop_jit(
        consumption.to_numpy(),
        pv.to_numpy(),
        da_price.to_numpy(),
        ctx.override,
        ctx.pct_low,
        ctx.pct_high,
        ctx.is_cheap,
        ctx.is_expensive,
        ctx.spread,
        ctx.top_max,
        ctx.exp_threshold,
        ctx.pv_forecast,
        ctx.hour,
        usable_kwh,
        max_charge_q,
        max_discharge_q,
        one_way,
        tune.min_spread_eur_kwh,
        tune.pv_skip_hour_local,
        tune.pv_skip_room_factor,
        tune.negative_price_charge_max_soc_frac,
        tune.pct_charge_max_soc_frac,
        tune.pct_discharge_min_soc_frac,
        tune.adaptive_discharge_floor,
        bool(allow_grid_export),
        # Day-ahead-only: geen onbalansbonus, dus percentiel-override volgt
        # de saldering-vlag voor grid export.
        False,
    )
    return pd.Series(out, index=consumption.index, name="battery_ac_kwh")


def imbalance_aware(
    consumption: pd.Series,
    pv: pd.Series,
    da_price: pd.Series,
    imbalance_price: pd.Series,
    spec: BatterySpec,
    tune: DispatchTuning | None = None,
    allow_grid_export: bool = True,
    precomputed_context: DispatchContext | None = None,
) -> pd.Series:
    """Dag-rank-arbitrage plus een percentiel-override op onbalansprijzen.
    De override pakt de absolute onbalans-extremen die geen day-ahead-ranking ziet.

    `allow_grid_export=False` capt ontladen op net_load (post-saldering).
    `precomputed_context` laat `run_sweep` de context delen over alle capaciteiten.
    """
    tune = tune or DispatchTuning()
    ctx = precomputed_context or _build_dispatch_context(
        da_price, pv, tune, override_price=imbalance_price
    )
    usable_kwh, max_charge_q, max_discharge_q, one_way = _spec_scalars(spec)

    out = _dispatch_loop_jit(
        consumption.to_numpy(),
        pv.to_numpy(),
        da_price.to_numpy(),
        ctx.override,
        ctx.pct_low,
        ctx.pct_high,
        ctx.is_cheap,
        ctx.is_expensive,
        ctx.spread,
        ctx.top_max,
        ctx.exp_threshold,
        ctx.pv_forecast,
        ctx.hour,
        usable_kwh,
        max_charge_q,
        max_discharge_q,
        one_way,
        tune.min_spread_eur_kwh,
        tune.pv_skip_hour_local,
        tune.pv_skip_room_factor,
        tune.negative_price_charge_max_soc_frac,
        tune.pct_charge_max_soc_frac,
        tune.pct_discharge_min_soc_frac,
        tune.adaptive_discharge_floor,
        bool(allow_grid_export),
        # Onbalans-trading (Frank): bonus op kWh boven net_load betaalt
        # round-trip-verlies plus terugleverpremie ruim als onbalans top-tail haalt.
        True,
    )
    return pd.Series(out, index=consumption.index, name="battery_ac_kwh")


@njit(cache=True)
def _perfect_foresight_loop(
    cons_arr: np.ndarray,
    pv_arr: np.ndarray,
    price_arr: np.ndarray,
    low_price_threshold: float,
    high_price_threshold: float,
    usable_kwh: float,
    max_charge_q: float,
    max_discharge_q: float,
    one_way_eff: float,
    allow_grid_export: bool,
) -> np.ndarray:
    """JIT-loop voor `perfect_foresight`. Drempels worden buiten de loop
    eenmaal berekend en als scalar doorgegeven."""
    n = cons_arr.shape[0]
    out = np.zeros(n)
    soc = 0.0
    for i in range(n):
        net_load = cons_arr[i] - pv_arr[i]
        if net_load < 0.0:
            soc, ac = charge_step(
                soc, -net_load, max_charge_q, usable_kwh, one_way_eff
            )
            out[i] = ac
            continue
        price = price_arr[i]
        if price <= low_price_threshold:
            soc, ac = charge_step(
                soc, max_charge_q, max_charge_q, usable_kwh, one_way_eff
            )
            out[i] = ac
        elif price >= high_price_threshold:
            target = max_discharge_q if allow_grid_export else net_load
            soc, ac = discharge_step(soc, target, max_discharge_q, one_way_eff)
            out[i] = -ac
        elif net_load > 0.0:
            soc, ac = discharge_step(soc, net_load, max_discharge_q, one_way_eff)
            out[i] = -ac
    return out


def perfect_foresight(
    consumption: pd.Series,
    pv: pd.Series,
    price: pd.Series,
    spec: BatterySpec,
    allow_grid_export: bool = True,
) -> pd.Series:
    """Greedy bovengrens: laden in goedkoopste kwartieren, ontladen in duurste.
    Alleen begrensd door capaciteit en vermogen, niet door dagstructuur.

    Niet realistisch (niemand heeft globale voorzicht), maar zet het plafond.
    """
    price_arr = price.to_numpy()
    low_price_threshold = float(np.percentile(price_arr, 25))
    high_price_threshold = float(np.percentile(price_arr, 75))
    usable_kwh, max_charge_q, max_discharge_q, one_way = _spec_scalars(spec)
    out = _perfect_foresight_loop(
        consumption.to_numpy(),
        pv.to_numpy(),
        price_arr,
        low_price_threshold,
        high_price_threshold,
        usable_kwh,
        max_charge_q,
        max_discharge_q,
        one_way,
        bool(allow_grid_export),
    )
    return pd.Series(out, index=consumption.index, name="battery_ac_kwh")


def optimal_lp(
    consumption: pd.Series,
    pv: pd.Series,
    import_price: pd.Series,
    export_price: pd.Series,
    spec: BatterySpec,
) -> pd.Series:
    """Globaal optimale batterij-dispatch via lineair programmeren.

    Minimaliseert Σ (import[t]·import_prijs[t] − export[t]·export_prijs[t]) over
    de hele serie met als voorwaarden:
      - SoC ∈ [0, bruikbare_capaciteit]
      - Laden / ontladen ∈ [0, max_kwh_per_kwartier]
      - Energiebalans: load - pv + laden - ontladen = import - export
      - SoC-dynamiek: soc[t+1] = soc[t] + laden[t]·η − ontladen[t]/η
      - soc[0] = soc[N] = 0  (geen gratis energie aan jaarrand)

    η = sqrt(round_trip_efficiency). Omdat laden en ontladen beide rendement
    kosten, doet de LP nooit beide tegelijk in optimum.

    Opgelost als één globaal LP via HiGHS (scipy). ~175k variabelen en ~70k
    gelijkheidsbeperkingen voor één jaar QH; HiGHS lost dat in seconden op.
    """
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix

    n = len(consumption)
    cons_arr = consumption.to_numpy(dtype=np.float64)
    pv_arr = pv.to_numpy(dtype=np.float64)
    imp_p = import_price.to_numpy(dtype=np.float64)
    exp_p = export_price.to_numpy(dtype=np.float64)

    eta = math.sqrt(spec.round_trip_efficiency)
    inv_eta = 1.0 / eta
    cap_kwh = spec.usable_kwh
    max_q = spec.max_charge_kwh_per_quarter()

    # Variabelvolgorde:
    #   [laden[0..n-1], ontladen[0..n-1], import[0..n-1], export[0..n-1], soc[0..n]]
    idx_chg = 0
    idx_dis = n
    idx_imp = 2 * n
    idx_exp = 3 * n
    idx_soc = 4 * n
    n_vars = 5 * n + 1

    # Doelfunctie: Σ import[t]·imp_p[t] - export[t]·exp_p[t]
    c = np.zeros(n_vars)
    c[idx_imp:idx_imp + n] = imp_p
    c[idx_exp:idx_exp + n] = -exp_p

    # Grenzen.
    bounds = [(0.0, max_q)] * n + [(0.0, max_q)] * n
    bounds += [(0.0, None)] * n + [(0.0, None)] * n
    bounds += [(0.0, cap_kwh)] * (n + 1)
    bounds[idx_soc] = (0.0, 0.0)
    bounds[idx_soc + n] = (0.0, 0.0)

    # Gelijkheden (sparse), per t twee: energiebalans en SoC-dynamiek.
    rows = np.empty(8 * n, dtype=np.int64)
    cols = np.empty(8 * n, dtype=np.int64)
    vals = np.empty(8 * n, dtype=np.float64)

    t_arr = np.arange(n, dtype=np.int64)
    rows[0 * n:1 * n] = t_arr
    cols[0 * n:1 * n] = idx_chg + t_arr
    vals[0 * n:1 * n] = -1.0
    rows[1 * n:2 * n] = t_arr
    cols[1 * n:2 * n] = idx_dis + t_arr
    vals[1 * n:2 * n] = 1.0
    rows[2 * n:3 * n] = t_arr
    cols[2 * n:3 * n] = idx_imp + t_arr
    vals[2 * n:3 * n] = 1.0
    rows[3 * n:4 * n] = t_arr
    cols[3 * n:4 * n] = idx_exp + t_arr
    vals[3 * n:4 * n] = -1.0

    rows[4 * n:5 * n] = n + t_arr
    cols[4 * n:5 * n] = idx_chg + t_arr
    vals[4 * n:5 * n] = -eta
    rows[5 * n:6 * n] = n + t_arr
    cols[5 * n:6 * n] = idx_dis + t_arr
    vals[5 * n:6 * n] = inv_eta
    rows[6 * n:7 * n] = n + t_arr
    cols[6 * n:7 * n] = idx_soc + t_arr
    vals[6 * n:7 * n] = -1.0
    rows[7 * n:8 * n] = n + t_arr
    cols[7 * n:8 * n] = idx_soc + t_arr + 1
    vals[7 * n:8 * n] = 1.0

    A_eq = csr_matrix((vals, (rows, cols)), shape=(2 * n, n_vars))
    b_eq = np.concatenate([cons_arr - pv_arr, np.zeros(n)])

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP solver faalde: {res.message}")

    chg = res.x[idx_chg:idx_chg + n]
    dis = res.x[idx_dis:idx_dis + n]
    # Met η<1 én normale prijsstructuren is gelijktijdig laden en ontladen
    # nooit optimaal. Pathologische tariefcombinaties (bv. negatieve import en
    # negatieve export tegelijk) zouden een spin kunnen toelaten; controleer
    # achteraf zodat we hier geen onfysisch optimum exporteren.
    co_active = float(np.dot(chg, dis))
    if co_active > 1e-6:
        raise RuntimeError(
            "LP-oplossing laadt en ontlaadt gelijktijdig "
            f"({co_active:.3f} kWh² overlap); prijsstructuur niet realistisch."
        )
    dispatch = chg - dis  # AC-zijde: positief=laden, negatief=ontladen
    return pd.Series(dispatch, index=consumption.index, name="battery_ac_kwh")


StrategyFn = Callable[..., pd.Series]
