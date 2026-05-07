"""Historische EPEX day-ahead (NL) en TenneT onbalansprijzen.

Day-ahead, in volgorde van voorkeur:
- ENTSO-E API. Vereist `entsoe_api_key` in TOML. Cached naar parquet bij eerste hit.
- EnergyZero publieke API (geen auth). Uurlijks NL day-ahead, ex BTW.
- Synthetisch als laatste redmiddel: gekalibreerde stochastische serie.

Onbalans, in volgorde van voorkeur:
- ENTSO-E API. 15-min Long/Short per biedzone NL, samengevat tot één
  "mid" serie per kwartier.
- TenneT publications API. Vereist TENNET_API_KEY (gratis, op aanvraag).
  15-min CSV per kwartier, regulatiestaat-bewust.
- Synthetisch: day-ahead + scheve ruis + 5% zware-staart shocks.
"""


import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import Settings


TENNET_API_URL = "https://api.tennet.eu/publications/v1/settlement-prices"


@dataclass(frozen=True)
class Prices:
    """Alle prijzen in EUR/kWh op dezelfde 15-min UTC-index als de loadserie."""

    day_ahead: pd.Series  # EPEX uurlijks ge-ffilled naar kwartier
    imbalance: pd.Series  # 15-min onbalans (echt of synthetisch)
    source: str  # day-ahead bron: "entsoe" | "energyzero" | "synthetic"
    imbalance_source: str  # "entsoe" | "tennet" | "synthetic"


def fetch_or_synthesize(settings: Settings, start: datetime, end: datetime) -> Prices:
    """Haal day-ahead en onbalans onafhankelijk op met eigen fallbacks."""
    da, da_source = _fetch_day_ahead(settings, start, end)
    if da is None:
        # Beide synthetisch, met dezelfde RNG voor coherentie.
        return _synthesize(start, end)
    imb, imb_source = _fetch_imbalance(settings, start, end, da)
    return Prices(
        day_ahead=da, imbalance=imb, source=da_source, imbalance_source=imb_source
    )


def _fetch_day_ahead(
    settings: Settings, start: datetime, end: datetime
) -> tuple[pd.Series | None, str]:
    if settings.entsoe_api_key:
        try:
            return _fetch_entsoe_da(settings, start, end), "entsoe"
        except Exception as e:  # noqa: BLE001
            print(f"[prices] ENTSO-E fetch failed ({e}); trying EnergyZero.")
    try:
        return _fetch_energyzero_da(settings, start, end), "energyzero"
    except Exception as e:  # noqa: BLE001
        print(f"[prices] EnergyZero fetch failed ({e}); falling back to synthetic.")
    return None, "synthetic"


def _fetch_imbalance(
    settings: Settings, start: datetime, end: datetime, da: pd.Series
) -> tuple[pd.Series, str]:
    if settings.entsoe_api_key:
        try:
            imb = _fetch_entsoe_imbalance(settings, start, end)
            imb = imb.reindex(da.index).ffill().bfill()
            return imb, "entsoe"
        except Exception as e:  # noqa: BLE001
            print(f"[prices] ENTSO-E imbalance fetch failed ({e}); trying TenneT.")
    # TenneT publiceert alleen NL onbalans. Voor andere zones zou dat
    # NL-data labellen als de gevraagde zone; sla over en gebruik synthetisch.
    if settings.entsoe_zone != "NL":
        print(
            f"[prices] TenneT is NL-only; entsoe_zone={settings.entsoe_zone} "
            "valt terug op synthetische onbalans."
        )
        return synthesize_imbalance(da), "synthetic"
    try:
        imb = _fetch_tennet_imbalance(settings, start, end)
        imb = imb.reindex(da.index).ffill().bfill()
        return imb, "tennet"
    except Exception as e:  # noqa: BLE001
        print(f"[prices] TenneT imbalance fetch failed ({e}); using synthetic imbalance.")
        return synthesize_imbalance(da), "synthetic"


def _fetch_entsoe_imbalance(
    settings: Settings, start: datetime, end: datetime
) -> pd.Series:
    """15-min onbalansprijzen van ENTSO-E voor `settings.entsoe_zone`.

    `query_imbalance_prices` levert een DataFrame met `Long` en `Short` in
    EUR/MWh. NL kent één prijs, dus we nemen het midden als per-kwartier
    proxy. Conservatief en behoudt de spread t.o.v. day-ahead. Gecached.
    """
    cache = _cache_path(settings, start, end, _zone_kind(settings.entsoe_zone, "imbalance_entsoe"))
    if cache.exists():
        return pd.read_parquet(cache)["imbalance"]

    from entsoe import EntsoePandasClient  # lazy import
    from bs4 import XMLParsedAsHTMLWarning

    client = EntsoePandasClient(api_key=settings.entsoe_api_key)
    parts: list[pd.DataFrame] = []
    cur = _to_utc(start)
    end_ts = _to_utc(end)
    while cur < end_ts:
        nxt = min(cur + pd.Timedelta(days=31), end_ts)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
            df = client.query_imbalance_prices(
                settings.entsoe_zone, start=cur, end=nxt
            )
        if not df.empty:
            parts.append(df)
        cur = nxt

    if not parts:
        raise RuntimeError(
            f"ENTSO-E gaf geen onbalansprijzen voor {settings.entsoe_zone} "
            f"in venster {start} → {end}."
        )
    full = pd.concat(parts).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    missing = {"Long", "Short"} - set(full.columns)
    if missing:
        raise RuntimeError(
            "ENTSO-E onbalans-respons mist verwachte kolommen "
            f"{sorted(missing)}; kreeg {sorted(full.columns)}. "
            "Mogelijk is de entsoe-py versie veranderd."
        )
    long_p = full["Long"].astype("float64")
    short_p = full["Short"].astype("float64")
    mid_eur_mwh = (long_p + short_p) / 2.0
    imb = (mid_eur_mwh / 1000.0).rename("imbalance").tz_convert("UTC")
    imb.to_frame("imbalance").to_parquet(cache)
    return imb


def _cache_path(settings: Settings, start: datetime, end: datetime, kind: str) -> Path:
    tag = f"{start:%Y%m%d}_{end:%Y%m%d}_{kind}.parquet"
    return settings.cache_dir / tag


def _zone_kind(zone: str, base: str) -> str:
    """Cache-naam suffix voor een zone. NL is back-compat; andere zones krijgen
    een suffix zodat caches niet door elkaar lopen."""
    return base if zone == "NL" else f"{base}_{zone}"


def _to_utc(ts: datetime) -> pd.Timestamp:
    """Forceer een datetime / pd.Timestamp / str naar UTC pd.Timestamp."""
    t = pd.Timestamp(ts)
    return t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")


def _fetch_entsoe_da(settings: Settings, start: datetime, end: datetime) -> pd.Series:
    from entsoe import EntsoePandasClient  # lazy import

    da_cache = _cache_path(settings, start, end, _zone_kind(settings.entsoe_zone, "da"))
    if da_cache.exists():
        da_eur_mwh = pd.read_parquet(da_cache)["price"]
    else:
        client = EntsoePandasClient(api_key=settings.entsoe_api_key)
        start_ts = _to_utc(start)
        end_ts = _to_utc(end)
        da_eur_mwh = client.query_day_ahead_prices(
            settings.entsoe_zone, start=start_ts, end=end_ts,
        )
        da_eur_mwh.to_frame("price").to_parquet(da_cache)

    da_eur_kwh = da_eur_mwh / 1000.0
    da_eur_kwh = da_eur_kwh.tz_convert("UTC")

    # Uurlijks naar 15-min forward-fill.
    idx = pd.date_range(start=start, end=end, freq="15min", tz="UTC", inclusive="left")
    return da_eur_kwh.reindex(idx, method="ffill")


def _fetch_energyzero_da(settings: Settings, start: datetime, end: datetime) -> pd.Series:
    """Haal NL uurlijks day-ahead op van de EnergyZero publieke API.

    Endpoint: https://api.energyzero.nl/v1/energyprices
      interval=4   = uurlijks
      usageType=1  = elektriciteit
      inclBtw=false = ruwe commodity (ex BTW), in EUR/kWh
    """
    cache = _cache_path(settings, start, end, "ez")
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        from_dt = _to_utc(start).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        till_dt = _to_utc(end).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        resp = requests.get(
            "https://api.energyzero.nl/v1/energyprices",
            params={
                "fromDate": from_dt,
                "tillDate": till_dt,
                "interval": 4,
                "usageType": 1,
                "inclBtw": "false",
            },
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        rows = body.get("Prices", [])
        if not rows:
            raise RuntimeError("EnergyZero gaf geen prijzen terug voor dit venster")
        df = pd.DataFrame(rows)
        df["readingDate"] = pd.to_datetime(df["readingDate"], utc=True)
        df = df.set_index("readingDate")[["price"]]
        df.to_parquet(cache)

    hourly = df["price"].astype("float64").rename("day_ahead")

    idx = pd.date_range(start=start, end=end, freq="15min", tz="UTC", inclusive="left")
    return hourly.reindex(idx, method="ffill")


def _fetch_tennet_imbalance(
    settings: Settings, start: datetime, end: datetime
) -> pd.Series:
    """Haal echte NL onbalansprijzen op via de TenneT publications API.

    Endpoint: ``https://api.tennet.eu/publications/v1/settlement-prices``.
    De ``apikey`` header is verplicht. Gratis sleutel via
    https://developer.tennet.eu/.

    Levert een 15-min EUR/kWh serie. Cached op
    ``cache/<start>_<end>_imbalance.parquet``.
    """
    cache = _cache_path(settings, start, end, "imbalance")
    if cache.exists():
        return pd.read_parquet(cache)["imbalance"]

    api_key = os.environ.get("TENNET_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TENNET_API_KEY niet gezet; geen echte onbalansprijzen mogelijk "
            "(sleutel aanvragen via https://developer.tennet.eu/)."
        )

    headers = {
        "apikey": api_key,
        "Accept": "text/csv",
        "user-agent": "thuisbat-sim",
    }

    # API limiteert request-grootte; per maand chunken om binnen de limieten te blijven.
    parsed: list[pd.DataFrame] = []
    cur = _to_utc(start).tz_convert("Europe/Amsterdam").normalize()
    end_ams = _to_utc(end).tz_convert("Europe/Amsterdam")
    while cur < end_ams:
        nxt = min(cur + pd.Timedelta(days=30), end_ams)
        params = {
            "date_from": cur.strftime("%d-%m-%Y %H:%M:%S"),
            "date_to": nxt.strftime("%d-%m-%Y %H:%M:%S"),
        }
        resp = requests.get(TENNET_API_URL, headers=headers, params=params, timeout=60)
        resp.raise_for_status()
        df = _parse_settlement_csv(resp.text)
        if df.empty:
            raise RuntimeError(f"TenneT gaf lege CSV terug voor {cur}..{nxt}")
        parsed.append(df)
        cur = nxt

    full = pd.concat(parsed).sort_index()
    full = full[~full.index.duplicated(keep="first")]

    imb = _settlement_to_single_price(full).tz_convert("UTC")
    imb.to_frame("imbalance").to_parquet(cache)
    return imb


def _parse_settlement_csv(csv_text: str) -> pd.DataFrame:
    """Parse een TenneT settlement-prices CSV naar een 15-min Europe/Amsterdam frame.

    De CSV bevat één rij per ISP (1..96). Elke rij heeft de ISP-datum in
    ``Timeinterval Start Loc`` plus periodenummer; we reconstrueren de tijd
    als ``datum + (Isp-1) * 15min`` lokale tijd. DST-overgangen via
    localize: voorjaarssprong shift, najaarsdubbel valt naar NaT.
    """
    df = pd.read_csv(StringIO(csv_text))
    if df.empty:
        return df

    base_local = pd.to_datetime(df["Timeinterval Start Loc"].str.split("T").str[0])
    base_local = base_local.dt.tz_localize(
        "Europe/Amsterdam", ambiguous="NaT", nonexistent="shift_forward"
    )
    ts = base_local + (df["Isp"].astype("int64") - 1) * pd.Timedelta(minutes=15)
    df = df.assign(timestamp=ts).dropna(subset=["timestamp"])
    return df.set_index("timestamp").sort_index()


def _settlement_to_single_price(df: pd.DataFrame) -> pd.Series:
    """Reduceer TenneT settlement-prices kolommen tot één €/kWh serie.

    NL hanteert een symmetrisch (single-price) regime. Welke prijs BRP's
    betalen/ontvangen wordt bepaald door de regulatiestaat:

      - state >= 1 (systeem short, opregeling): beide richtingen klaren op
        ``Price Shortage`` (afname-prijs, typisch hoog).
      - state <= -1 (systeem long, neerregeling): beide richtingen klaren op
        ``Price Surplus`` (invoed-prijs, typisch laag).
      - state == 0 (geen regulering): geen echte settlement; gemiddelde als
        defensieve default.

    State == 2 is "beide richtingen geactiveerd" binnen een kwartier; voor de
    single-price proxy is shortage de conservatieve keus.

    Bron: €/MWh, hier omgezet naar €/kWh.
    """
    if "Regulation State" not in df.columns:
        raise RuntimeError(
            "TenneT settlement-prices CSV mist kolom 'Regulation State'; "
            f"kreeg {list(df.columns)}"
        )
    state = df["Regulation State"].astype("int64")
    shortage = df["Price Shortage"].astype("float64")
    surplus = df["Price Surplus"].astype("float64")
    midprice = (shortage + surplus) / 2.0

    price = midprice.copy()
    price[state >= 1] = shortage[state >= 1]
    price[state <= -1] = surplus[state <= -1]

    return (price / 1000.0).rename("imbalance")


def _synthesize(start: datetime, end: datetime) -> Prices:
    """Aannemelijke synthetische NL day-ahead, gekalibreerd op 2024-2025.

    Diurnale vorm: laag 02-05, middagdip 11-14 (zon-overschot), avondpiek
    17-20. Jaarlijks gemiddelde ~€0.10/kWh commodity, dagspread ~€0.30/kWh.
    Alleen voor smoke-tests. Met ENTSO-E API-key wordt dit overschreven.
    """
    rng = np.random.default_rng(seed=42)
    idx = pd.date_range(start=start, end=end, freq="15min", tz="UTC", inclusive="left")
    n = len(idx)

    h = idx.hour + idx.minute / 60.0  # uur (UTC, close enough voor de vorm)
    diurnal = (
        0.09
        + 0.06 * np.sin((h - 6) / 24 * 2 * np.pi)
        - 0.18 * np.exp(-((h - 12.5) ** 2) / 6.0)  # middagdip (zon-overschot)
        + 0.28 * np.exp(-((h - 18.5) ** 2) / 4.0)  # avondpiek
    )
    seasonal = 0.03 * np.cos((idx.dayofyear - 15) / 365 * 2 * np.pi)
    noise = rng.normal(0, 0.07, size=n)
    da = pd.Series(diurnal + seasonal + noise, index=idx, name="day_ahead").clip(lower=-0.12)

    imbalance = synthesize_imbalance(da, rng=rng)
    return Prices(
        day_ahead=da, imbalance=imbalance, source="synthetic", imbalance_source="synthetic"
    )


def synthesize_imbalance(
    day_ahead: pd.Series, rng: np.random.Generator | None = None
) -> pd.Series:
    """Onbalans rekent per kwartier af met een veel bredere spread dan day-ahead.

    Model: onbalans = day-ahead + scheve ruis, met ~5% kwartieren met extreme
    uitslag (|delta| > €0.30/kWh).
    """
    if rng is None:
        rng = np.random.default_rng(seed=7)
    n = len(day_ahead)
    base_noise = rng.normal(0, 0.05, size=n)
    # Zware-staart shocks 5% van de tijd.
    shock_mask = rng.random(n) < 0.05
    shocks = rng.normal(0, 0.50, size=n) * shock_mask
    delta = base_noise + shocks
    return (day_ahead + delta).rename("imbalance")
