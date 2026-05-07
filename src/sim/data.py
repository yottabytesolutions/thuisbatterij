"""Laad huishoudelijk verbruik en ZP-productie uit QuestDB; resample en gap-fill.

Alleen lezen. Ruwe QuestDB-tabellen (`stroom`, `solar`, `solar_inverters`) worden
nooit gewijzigd. Geschoonde series leven in memory of in de lokale parquet-cache.

Gap-fill-model:
  De kWh-tellers (UsageCounter*, OutputCounter*, ProductionWattHours) tellen
  door tijdens uitvallen; alleen de seconde-niveau power-metingen ontbreken.
  Voor elke gap-dag kennen we daarom de echte dagtotalen via teller-deltas.
  We nemen het maandgemiddelde dagprofiel en schalen het zodat de dagintegraal
  matcht met de gemeten teller-delta. Dat behoudt de echte energiestromen.
"""


from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .questdb import QuestDB

# 1 Hz verwacht. Dagen onder deze drempel zijn "stuk" en worden vervangen
# door een geschaalde reconstructie (zie gap_fill_scaled_shape).
GOOD_DAY_MIN_SAMPLES = 80_000

QUARTER = "15min"

# Bump bij elke logica-wijziging die de output van `load_series` beïnvloedt
# (bv. teller-filter, gap-fill, schaalmethode). Oude parquet-caches matchen
# dan niet meer en worden automatisch opnieuw opgebouwd.
LOAD_CACHE_VERSION = 2


@dataclass(frozen=True)
class LoadSeries:
    """Op 15 min geresampelde energieserie, kWh per bucket."""

    consumption_kwh: pd.Series  # huishoudelijk verbruik (bruto, vóór ZP)
    pv_kwh: pd.Series  # ZP-productie
    grid_import_kwh: pd.Series  # wat de meter zag importeren, per kwartier
    grid_export_kwh: pd.Series  # wat de meter zag exporteren, per kwartier
    gap_filled_index: pd.DatetimeIndex  # welke buckets via gap-fill kwamen


def daily_counter_totals(
    db: QuestDB, start: datetime, end: datetime
) -> pd.DataFrame:
    """Echte dagtotalen uit de kWh-tellers. Geldig ook op gap-dagen.

    Tellers stijgen monotoon, dus dagen zonder ruwe rijen hebben toch een
    welbepaald dagtotaal: het verschil tussen de eerste teller op de volgende
    dag en de laatste op de vorige, verdeeld over de tussenliggende dagen.
    We benaderen dat via per-dag min/max ffill/bfill en dag-op-dag delta's.
    """
    # Filter 0-rijen om min() te beschermen tegen sentinel-rijen die de
    # ingestor soms emit bij cold-start. Per kant filteren via OR; we splitten
    # in pandas per kolom hieronder.
    sql = f"""
    SELECT
      to_timezone(timestamp, 'UTC') AS ts,
      UsageCounter1, UsageCounter2,
      OutputCounter1, OutputCounter2
    FROM stroom
    WHERE timestamp >= '{start.strftime("%Y-%m-%d %H:%M:%S")}'
      AND timestamp <  '{end.strftime("%Y-%m-%d %H:%M:%S")}'
      AND (
        UsageCounter1 + UsageCounter2 > 0
        OR OutputCounter1 + OutputCounter2 > 0
      )
    """
    raw = db.query(sql)
    raw["ts"] = pd.to_datetime(raw["ts"], utc=True)
    raw = raw.set_index("ts").sort_index()

    full_days = pd.date_range(
        start=pd.Timestamp(start).normalize(),
        end=pd.Timestamp(end).normalize() - pd.Timedelta(days=1),
        freq="D",
        tz="UTC",
    )
    return pd.DataFrame(
        {
            "import_kwh": _delta_per_day(
                raw["UsageCounter1"] + raw["UsageCounter2"], full_days
            ),
            "export_kwh": _delta_per_day(
                raw["OutputCounter1"] + raw["OutputCounter2"], full_days
            ),
        }
    )


def _delta_per_day(counter: pd.Series, full_days: pd.DatetimeIndex) -> pd.Series:
    """Dagtotaal uit een monotoon-stijgende kWh-teller.

    Filter 0-rijen per kant: de WHERE-clausule houdt rijen waarop één van
    beide kanten een waarde heeft, maar de andere kant kan nog 0 zijn op
    diezelfde rij. Dat zou min() naar 0 trekken. Filter dus per teller.
    """
    nonzero = counter[counter > 0]
    daily = nonzero.resample("D").agg(["min", "max"]).reindex(full_days)
    delta = daily["max"] - daily["min"]
    return _fill_missing_day_totals(delta, daily["max"], daily["min"])


def _fill_missing_day_totals(
    daily_delta: pd.Series, daily_max: pd.Series, daily_min: pd.Series
) -> pd.Series:
    """Verdeel teller-delta's over reeksen volledig-ontbrekende dagen.

    Per aaneengesloten NaN-reeks: energie = next_min - prev_max, gelijk verdeeld.
    Dagen met 0.0 delta blijven ongemoeid (echte "geen flow" dagen).

    O(n) via één ffill/bfill in plaats van per-gat scan.
    """
    out = daily_delta.copy()
    # prev_max[i] = laatste niet-NaN max strikt vóór i.
    prev_max = daily_max.ffill().shift(1)
    # next_min[i] = eerste niet-NaN min op of na i.
    next_min = daily_min.bfill()

    is_nan = daily_delta.isna().to_numpy()
    n = len(out)
    i = 0
    while i < n:
        if is_nan[i]:
            j = i
            while j < n and is_nan[j]:
                j += 1
            pm = prev_max.iloc[i] if i < n else float("nan")
            nm = next_min.iloc[j] if j < n else float("nan")
            if not pd.isna(pm) and not pd.isna(nm):
                bridge = max(0.0, float(nm) - float(pm))
                out.iloc[i:j] = bridge / (j - i)
            else:
                out.iloc[i:j] = 0.0
            i = j
        else:
            i += 1
    return out


def daily_pv_total(db: QuestDB, start: datetime, end: datetime) -> pd.Series:
    """ZP-productie per dag uit de ProductionWattHours-teller.

    Ontbrekende dagen overbruggen door de teller-delta tussen de omringende
    goede metingen te verdelen, net als bij de import/export-tellers.
    """
    sql = f"""
    SELECT timestamp, ProductionWattHours
    FROM solar
    WHERE timestamp >= '{start.strftime("%Y-%m-%d %H:%M:%S")}'
      AND timestamp <  '{end.strftime("%Y-%m-%d %H:%M:%S")}'
      AND ProductionWattHours > 0
    """
    raw = db.query(sql)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.set_index("timestamp").sort_index()

    full_days = pd.date_range(
        start=pd.Timestamp(start).normalize(),
        end=pd.Timestamp(end).normalize() - pd.Timedelta(days=1),
        freq="D",
        tz="UTC",
    )
    return (_delta_per_day(raw["ProductionWattHours"], full_days) / 1000.0).rename("pv_kwh")


def load_high_res_grid(
    db: QuestDB, start: datetime, end: datetime
) -> tuple[pd.Series, pd.Series, pd.DatetimeIndex]:
    """Haal instantane TotalPowerUsage / TotalPowerOutput op, resample naar 15-min kWh.
    Geeft (import_kwh, export_kwh, bad_days_index) terug.

    Slechte dagen = dagen met te weinig samples of helemaal geen (volledige uitval).
    """
    sql = f"""
    SELECT timestamp, TotalPowerUsage, TotalPowerOutput
    FROM stroom
    WHERE timestamp >= '{start.strftime("%Y-%m-%d %H:%M:%S")}'
      AND timestamp <  '{end.strftime("%Y-%m-%d %H:%M:%S")}'
    """
    raw = db.query(sql)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.set_index("timestamp").sort_index()

    full_days = pd.date_range(
        start=pd.Timestamp(start).normalize(),
        end=pd.Timestamp(end).normalize() - pd.Timedelta(days=1),
        freq="D",
        tz="UTC",
    )
    samples_per_day = raw.resample("D").size().reindex(full_days, fill_value=0)
    bad_days = samples_per_day[samples_per_day < GOOD_DAY_MIN_SAMPLES].index

    imp = (raw["TotalPowerUsage"].astype("float64").resample(QUARTER).mean() * 0.25 / 1000.0)
    exp = (raw["TotalPowerOutput"].astype("float64").resample(QUARTER).mean() * 0.25 / 1000.0)
    return imp.rename("grid_import_kwh"), exp.rename("grid_export_kwh"), pd.DatetimeIndex(bad_days)


def load_high_res_pv(
    db: QuestDB, start: datetime, end: datetime
) -> tuple[pd.Series, pd.DatetimeIndex]:
    """Resample ZP-vermogen naar 15-min kWh en markeer dagen met sparse/ontbrekende dekking.

    Omvormers loggen meestal om de paar seconden; dagen met minder dan 50
    rijen als gap-dag behandelen (zelfde shape-fill als bij grid).
    """
    sql = f"""
    SELECT timestamp, ProductionWatt
    FROM solar
    WHERE timestamp >= '{start.strftime("%Y-%m-%d %H:%M:%S")}'
      AND timestamp <  '{end.strftime("%Y-%m-%d %H:%M:%S")}'
      AND ProductionWatt IS NOT NULL
    """
    raw = db.query(sql)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.set_index("timestamp").sort_index()

    full_days = pd.date_range(
        start=pd.Timestamp(start).normalize(),
        end=pd.Timestamp(end).normalize() - pd.Timedelta(days=1),
        freq="D",
        tz="UTC",
    )
    # Omvormer logt overdag elke paar seconden (~500-700/dag zomer, ~150-300 winter).
    # 50 is een conservatieve "is er überhaupt data" drempel.
    samples_per_day = raw.resample("D").size().reindex(full_days, fill_value=0)
    bad_days = samples_per_day[samples_per_day < 50].index

    pv = (
        raw["ProductionWatt"].astype("float64").resample(QUARTER).mean().fillna(0.0)
        * 0.25
        / 1000.0
    ).rename("pv_kwh")
    return pv, pd.DatetimeIndex(bad_days)


def gap_fill_scaled_shape(
    series: pd.Series,
    bad_days: pd.DatetimeIndex,
    daily_targets_kwh: pd.Series,
) -> tuple[pd.Series, pd.DatetimeIndex]:
    """Vervang gap-dagen met een geschaalde maandvorm waarvan de integraal
    gelijk is aan daily_targets_kwh.

    Stappen:
      1. Bereken per maand, per kwartier-van-dag het gemiddelde profiel uit goede dagen.
      2. Voor elke gap-dag: pak het maandprofiel, normaliseer op som 1, schaal
         met het echte dagtotaal uit `daily_targets_kwh`.
      3. Stik terug in de serie.
    """
    if len(bad_days) == 0:
        return series, pd.DatetimeIndex([])

    bad_days_set = {pd.Timestamp(day).date() for day in bad_days}

    df = series.to_frame("v").copy()
    df["date"] = df.index.date
    df["month"] = df.index.month
    df["qod"] = df.index.hour * 4 + df.index.minute // 15
    df["is_bad"] = df["date"].isin(bad_days_set)

    good = df[~df["is_bad"]]
    shape = good.groupby(["month", "qod"])["v"].mean()
    # Normaliseer elk maandprofiel (96 buckets) op som 1. Maanden zonder
    # bruikbare energie krijgen expliciet een uniform profiel, zonder globale
    # NumPy-waarschuwingen te onderdrukken.
    uniform = 1.0 / 96
    shape_per_month = shape.unstack("qod")  # index=month, columns=qod
    month_sums = shape_per_month.sum(axis=1)
    valid_months = month_sums.gt(0.0)
    shape_per_month.loc[valid_months] = shape_per_month.loc[valid_months].div(
        month_sums.loc[valid_months], axis=0
    )
    shape_per_month.loc[~valid_months] = uniform
    shape_per_month = shape_per_month.fillna(uniform)

    targets_by_date = {
        timestamp.date(): float(target_kwh)
        for timestamp, target_kwh in daily_targets_kwh.items()
    }

    bad_mask = df["is_bad"]
    bad_idx = df.index[bad_mask]
    if len(bad_idx) == 0:
        return series, pd.DatetimeIndex([])

    lookup = pd.MultiIndex.from_arrays(
        [bad_idx.month, bad_idx.hour * 4 + bad_idx.minute // 15], names=["month", "qod"]
    )
    weights = shape_per_month.stack().reindex(lookup, fill_value=uniform).to_numpy()
    targets = pd.Series(bad_idx.date, index=bad_idx).map(targets_by_date).fillna(0.0).to_numpy()
    df.loc[bad_idx, "v"] = targets * weights
    return df["v"].rename(series.name), pd.DatetimeIndex(bad_idx)


def load_series(
    db: QuestDB,
    start: datetime,
    end: datetime,
    cache_dir: Path | None = None,
) -> LoadSeries:
    """Top-level loader: high-res power binnenhalen en gap-fillen op basis van
    teller-afgeleide dagtotalen.

    Alle operaties zijn read-only tegen QuestDB. Met `cache_dir` wordt de
    uiteindelijke 15-min serie weggeschreven naar
    `<cache_dir>/load_<venster>.parquet` en daarna van daar geserveerd. Een
    afgesloten venster verandert niet, dus de cache is veilig herbruikbaar.
    """
    cache_path = (
        _load_cache_path(cache_dir, start, end) if cache_dir is not None else None
    )
    if cache_path is not None and cache_path.exists():
        df = pd.read_parquet(cache_path)
        gap_filled_index = pd.DatetimeIndex(
            pd.read_parquet(cache_path.with_suffix(".gap.parquet"))["timestamp"]
        )
        return LoadSeries(
            consumption_kwh=df["consumption_kwh"],
            pv_kwh=df["pv_kwh"],
            grid_import_kwh=df["grid_import_kwh"],
            grid_export_kwh=df["grid_export_kwh"],
            gap_filled_index=gap_filled_index,
        )
    imp, exp, grid_bad_days = load_high_res_grid(db, start, end)
    pv, pv_bad_days = load_high_res_pv(db, start, end)

    daily_targets = daily_counter_totals(db, start, end)
    pv_daily = daily_pv_total(db, start, end)

    # Lijn uit op gemeenschappelijke 15-min UTC-index.
    idx = pd.date_range(start=start, end=end, freq=QUARTER, tz="UTC", inclusive="left")
    imp = imp.reindex(idx).fillna(0.0)
    exp = exp.reindex(idx).fillna(0.0)
    pv = pv.reindex(idx).fillna(0.0)

    # Gap-fill elke serie met zijn eigen teller-afgeleide dagtotaal.
    imp_filled, gap_idx = gap_fill_scaled_shape(
        imp, grid_bad_days, daily_targets["import_kwh"]
    )
    exp_filled, _ = gap_fill_scaled_shape(
        exp, grid_bad_days, daily_targets["export_kwh"]
    )
    pv_filled, pv_gap_idx = gap_fill_scaled_shape(pv, pv_bad_days, pv_daily)

    consumption_filled = (imp_filled - exp_filled + pv_filled).clip(lower=0.0).rename(
        "consumption_kwh"
    )

    all_gaps = gap_idx.union(pv_gap_idx)
    result = LoadSeries(
        consumption_kwh=consumption_filled.astype("float64"),
        pv_kwh=pv_filled.astype("float64"),
        grid_import_kwh=imp_filled.astype("float64"),
        grid_export_kwh=exp_filled.astype("float64"),
        gap_filled_index=all_gaps,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "consumption_kwh": result.consumption_kwh,
                "pv_kwh": result.pv_kwh,
                "grid_import_kwh": result.grid_import_kwh,
                "grid_export_kwh": result.grid_export_kwh,
            }
        ).to_parquet(cache_path)
        pd.DataFrame({"timestamp": all_gaps}).to_parquet(
            cache_path.with_suffix(".gap.parquet")
        )

    return result


def _load_cache_path(cache_dir: Path, start: datetime, end: datetime) -> Path:
    return (
        cache_dir
        / f"load_{start:%Y%m%d}_{end:%Y%m%d}_v{LOAD_CACHE_VERSION}.parquet"
    )


def summary(ls: LoadSeries) -> dict[str, float]:
    """Jaartotalen voor sanity-check tegen bekende getallen."""
    hours = len(ls.consumption_kwh) * 0.25
    annual_factor = 8760.0 / hours if hours else 0.0
    return {
        "consumption_kwh_total": float(ls.consumption_kwh.sum()),
        "pv_kwh_total": float(ls.pv_kwh.sum()),
        "consumption_kwh_annualized": float(ls.consumption_kwh.sum() * annual_factor),
        "pv_kwh_annualized": float(ls.pv_kwh.sum() * annual_factor),
        "gap_filled_buckets": float(len(ls.gap_filled_index)),
        "buckets_total": float(len(ls.consumption_kwh)),
    }


