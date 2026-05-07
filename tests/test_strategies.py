"""Tests voor de heuristische dispatchstrategieën.

Elke test isoleert één regel met een klein synthetisch prijs- en loadvenster.
"""


import numpy as np
import pandas as pd

from sim.battery import BatterySpec
from sim.strategies import (
    DispatchTuning,
    day_ahead_arbitrage,
    imbalance_aware,
)


def _quarter_index(n: int, start: str = "2025-06-01 00:00") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq="15min", tz="UTC")


def _zero_series(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(0.0, index=idx)


def _spec() -> BatterySpec:
    # Kleine batterij zodat dispatch per kwartier makkelijk te volgen is.
    return BatterySpec(
        capacity_kwh=10.0,
        usable_fraction=1.0,
        max_charge_kw=4.0,  # 1.0 kWh per kwartier
        max_discharge_kw=4.0,
        round_trip_efficiency=1.0,
    )


def test_negative_price_absorption_overrides_daily_rank() -> None:
    """Een `da < 0` kwartier moet laden, ook buiten de daggoedkoop-set."""
    idx = _quarter_index(96)  # één kalenderdag
    cons = pd.Series(0.5, index=idx)  # constante 0.5 kWh/kwartier load
    pv = _zero_series(idx)
    # Day-ahead curve: vlak €0.10, behalve kwartier 50 met -€0.05.
    # Door andere kwartieren goedkoper te maken valt die niet in de goedkoopste 16.
    da = pd.Series(0.10, index=idx)
    da.iloc[:16] = 0.05  # de dagrank-goedkoop-set
    da.iloc[50] = -0.05  # negatieve prijs, later op de dag
    im = da.copy()  # onbalans == day-ahead, dus geen override

    tune = DispatchTuning(
        # Zet spread-idle en percentieloverride uit.
        min_spread_eur_kwh=0.0,
        im_low_percentile=0.0,
        im_high_percentile=100.0,
        pv_skip_room_factor=99.0,
    )
    dispatch = imbalance_aware(cons, pv, da, im, _spec(), tune=tune, allow_grid_export=False)

    # Kwartier 50 moet laden door de negatieve-prijsregel.
    assert dispatch.iloc[50] > 0.0, (
        f"verwachtte laden op negatieve-prijs kwartier, kreeg {dispatch.iloc[50]}"
    )


def test_flat_day_idles_with_spread_filter() -> None:
    """Bij te kleine dagspread mag de batterij niet cyclen."""
    idx = _quarter_index(96)
    cons = pd.Series(0.5, index=idx)
    pv = _zero_series(idx)
    # Dagspread van €0.01 ligt ruim onder de standaarddrempel van €0.05.
    da = pd.Series(np.linspace(0.10, 0.11, 96), index=idx)
    im = da.copy()

    tune = DispatchTuning(
        min_spread_eur_kwh=0.05,
        im_low_percentile=0.0,
        im_high_percentile=100.0,
        pv_skip_room_factor=99.0,
        negative_price_charge_max_soc_frac=-1.0,  # uit
    )
    dispatch = imbalance_aware(cons, pv, da, im, _spec(), tune=tune, allow_grid_export=False)

    # Op een vlakke dag mag de batterij niet uit het net laden.
    assert (dispatch <= 0.0).all(), (
        "laden vuurde op vlakke dag; verwachtte idle"
    )
    # Ontladen moet ook nul zijn; anders betaal je rendementsverlies voor niets.
    assert (dispatch == 0.0).all(), (
        "ontladen vuurde op vlakke dag; verwachtte idle"
    )


def test_pv_aware_morning_charge_skipped() -> None:
    """Sla netladen voor zonsopkomst over als de PV-forecast genoeg is.

    Zowel dag-rank als morning-skip werken nu in lokale tijd
    (Europe/Amsterdam). Indices 96+16..96+24 = UTC 04:00-06:00 op dag 1 =
    local 06:00-08:00 (CEST), ruim binnen `pv_skip_hour_local=9`.
    """
    idx = _quarter_index(96 * 2)
    cons = pd.Series(0.5, index=idx)
    pv = pd.Series(0.0, index=idx)
    pv.iloc[24:64] = 0.5  # 20 kWh PV op dag 0 → forecast voor dag 1 ≥ 20.
    da = pd.Series(0.20, index=idx)
    morning_start = 96 + 16  # UTC 04:00 op dag 1 = local 06:00 (CEST)
    morning_end = morning_start + 8  # UTC 06:00 op dag 1 = local 08:00 (CEST)
    da.iloc[morning_start:morning_end] = 0.05
    im = da.copy()

    tune = DispatchTuning(
        min_spread_eur_kwh=0.05,
        im_low_percentile=0.0,
        im_high_percentile=100.0,
        pv_skip_room_factor=1.0,
        pv_skip_hour_local=9,
        pv_forecast_window_days=7,
        negative_price_charge_max_soc_frac=-1.0,
    )

    dispatch = imbalance_aware(cons, pv, da, im, _spec(), tune=tune, allow_grid_export=False)

    # Geen positieve dispatch (= geen netladen) in het morning-venster.
    morning_window = dispatch.iloc[morning_start:morning_end]
    assert (morning_window <= 0.0).all(), (
        f"verwachtte geen netladen in pre-sunrise kwartieren van dag 1, "
        f"kreeg {morning_window.tolist()}"
    )


def test_imbalance_percentile_override_charges_at_low_extreme() -> None:
    """Bij een laag onbalanspercentiel moet de batterij laden."""
    # 60 dagen data. Day-ahead is constant €0.20, dus geen dagrank-winst.
    # Eén diepe onbalansdip moet onvoorwaardelijk laden triggeren.
    idx = _quarter_index(96 * 60)
    cons = pd.Series(0.5, index=idx)
    pv = _zero_series(idx)
    da = pd.Series(0.20, index=idx)
    im = pd.Series(0.20, index=idx)
    # Voeg ruis toe voor een goed gedefinieerd rollend percentiel.
    rng = np.random.default_rng(0)
    im = im + rng.normal(0, 0.02, size=len(im))
    target_idx = 96 * 45 + 12
    im.iloc[target_idx] = -0.50

    tune = DispatchTuning(
        min_spread_eur_kwh=99.0,  # zet het dagrank-arbitragepad uit
        im_low_percentile=2.0,
        im_high_percentile=99.0,
        im_percentile_window_days=30,
        negative_price_charge_max_soc_frac=-1.0,
        pv_skip_room_factor=99.0,
    )

    dispatch = imbalance_aware(cons, pv, da, im, _spec(), tune=tune, allow_grid_export=False)

    # Het override-kwartier moet de batterij laden.
    assert dispatch.iloc[target_idx] > 0.0, (
        f"verwachtte percentiel-override laden op idx {target_idx}, "
        f"kreeg {dispatch.iloc[target_idx]}"
    )


def test_day_ahead_arbitrage_does_not_force_grid_export_in_postsaldering() -> None:
    """`day_ahead_arbitrage` mag post-saldering geen netexport forceren."""
    idx = _quarter_index(96 * 60)
    cons = pd.Series(0.0, index=idx)  # geen load, dus ontladen zou exporteren
    pv = _zero_series(idx)
    # Plaats een hoge day-ahead piek waar de override zou triggeren.
    rng = np.random.default_rng(1)
    da = pd.Series(0.10, index=idx) + rng.normal(0, 0.005, size=len(idx))
    da.iloc[96 * 45 + 30] = 5.0  # grote piek

    tune = DispatchTuning(
        min_spread_eur_kwh=99.0,
        im_low_percentile=2.0,
        im_high_percentile=99.0,
        negative_price_charge_max_soc_frac=-1.0,
        pv_skip_room_factor=99.0,
    )
    dispatch = day_ahead_arbitrage(cons, pv, da, _spec(), tune=tune, allow_grid_export=False)

    # Netto-load is overal nul, dus geen kwartier mag negatieve dispatch hebben.
    assert (dispatch >= 0.0).all(), (
        "day_ahead_arbitrage exporteerde naar net post-saldering; "
        "percentieloverride moet allow_grid_export=False respecteren"
    )
