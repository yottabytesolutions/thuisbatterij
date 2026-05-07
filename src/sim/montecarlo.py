"""Meerjaarlijkse Monte Carlo over historische ENTSO-E NL prijzen.

De jaarlijkse run beantwoordt "wat bespaart de gebruiker in 2025-2026". Deze
module beantwoordt "wat is de realistische *verdeling* van jaarlijkse
besparingen over alle prijsregimes sinds 2015".

Aanpak:

  1. Speel de werkelijke load (mei 2025 tot mei 2026) af tegen elk
     historisch jaar, kalender-uitgelijnd op (maand, dag, uur, minuut). 11
     jaren in cache (2015 tot 2025) leveren elk een deterministische
     scenario-set op.

  2. Bootstrap-resample N samples uit die 11 jaren met teruglegging en
     bereken gemiddelde / p10 / p50 / p90 / std per scenario. De deterministische
     simulaties worden in-memory gecached; bootstrap is puur numpy-indexing.
     `--monte-carlo 100` en `--monte-carlo 10_000` kosten daardoor evenveel.

Onbalansprijzen voor Frank-scenario's komen uit historische ENTSO-E-cache zodra
die voor de mei→mei-spanne aanwezig is. Alleen ontbrekende spannes vallen terug
op synthetische onbalans uit het jaarlijkse day-ahead-verloop.
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .battery import BatterySpec
from .config import Settings
from .data import LoadSeries
from .economics import TariffParams
from .prices import Prices, _fetch_entsoe_imbalance, synthesize_imbalance
from .simulate import run_scenario


def _default_scenarios(tariffs: dict[str, TariffParams]) -> list[tuple[str, str, str]]:
    """Bouw dezelfde scenariolijst als het hoofdrapport uit een tariefregistry.
    Alleen scenario's met aanwezige tarieven worden opgenomen.
    """
    out: list[tuple[str, str, str]] = []
    fixed = next(
        (tariff_key for tariff_key in ("baseline-fixed", "fixed") if tariff_key in tariffs),
        None,
    )
    tibber = next(
        (tariff_key for tariff_key in ("tibber", "tibber-day-ahead") if tariff_key in tariffs),
        None,
    )
    frank = next(
        (tariff_key for tariff_key in ("frank", "frank-imbalance") if tariff_key in tariffs),
        None,
    )
    perfect = next(
        (tariff_key for tariff_key in ("perfect-foresight", "perfect") if tariff_key in tariffs),
        None,
    )

    if fixed:
        out.append((fixed, "no_battery", fixed))
        out.append((f"{fixed}-postsaldering", "no_battery", f"{fixed}-postsaldering"))
        out.append(("battery-pv-only", "pv_self", fixed))
    if tibber:
        out.append(("dynamic-no-battery", "no_battery", tibber))
        out.append((tibber, "day_ahead", tibber))
        out.append(
            ("dynamic-no-battery-postsaldering", "no_battery", f"{tibber}-postsaldering")
        )
        out.append((f"{tibber}-postsaldering", "day_ahead", f"{tibber}-postsaldering"))
        if f"{tibber}-curtail-postsaldering" in tariffs:
            out.append(
                (
                    f"{tibber}-curtail-postsaldering",
                    "day_ahead",
                    f"{tibber}-curtail-postsaldering",
                )
            )
            out.append(
                (
                    "dynamic-curtail-no-battery-postsaldering",
                    "no_battery",
                    f"{tibber}-curtail-postsaldering",
                )
            )
    if frank:
        out.append((frank, "imbalance", frank))
        out.append((f"{frank}-postsaldering", "imbalance", f"{frank}-postsaldering"))
    if perfect:
        out.append((f"{perfect}-postsaldering", "perfect", f"{perfect}-postsaldering"))
    return out


@dataclass(frozen=True)
class YearResult:
    """Deterministische uitkomst van alle scenario's tegen één jaar prijzen."""

    year: int  # startjaar van de mei-Y tot mei-(Y+1) spanne
    da_source: str  # herkomst voor traceerbaarheid (bv. "entsoe-2018")
    imbalance_source: str
    annual_cost_by_scenario: dict[str, float]


@dataclass(frozen=True)
class ScenarioStats:
    name: str
    mean_eur: float
    std_eur: float
    p10_eur: float
    p50_eur: float
    p90_eur: float
    min_eur: float
    max_eur: float
    n_samples: int


@dataclass(frozen=True)
class MonteCarloResult:
    n_samples: int
    years_used: list[int]
    year_results: list[YearResult]
    sample_year_picks: list[int]  # welk jaar elke van N samples trok
    scenario_stats: dict[str, ScenarioStats]
    walltime_seconds: float
    workers: int


# Ontdekken en prijsconstructie.


def discover_price_years(cache_dir: Path) -> list[int]:
    """Geef startjaren terug waarvoor een mei Y tot mei Y+1 spanne mogelijk is.

    Een spanne vereist de tweede helft van jaar Y plus de eerste vier maanden
    van Y+1, dus beide `da_NL_Y.parquet` en `da_NL_(Y+1).parquet` moeten bestaan.
    De 2025 tot 2026 spanne is een speciaal geval: er is geen 2026 cache, maar
    `20250501_20260501_da.parquet` dekt het volledige live venster.
    """
    years_cached = sorted(
        int(cache_path.stem.split("_")[-1])
        for cache_path in cache_dir.glob("da_NL_*.parquet")
    )
    spans = [year for year in years_cached if (year + 1) in years_cached]
    live = cache_dir / "20250501_20260501_da.parquet"
    if live.exists() and 2025 not in spans:
        spans.append(2025)
    return sorted(set(spans))


def _load_calendar_year_da(cache_dir: Path, year: int) -> pd.Series:
    """Lees da_NL_<jaar>.parquet en geef EUR/kWh terug met UTC-index."""
    df = pd.read_parquet(cache_dir / f"da_NL_{year}.parquet")
    s = df["price"].astype("float64") / 1000.0  # EUR/MWh -> EUR/kWh
    return s.tz_convert("UTC").rename("day_ahead")


def _build_da_for_span(
    cache_dir: Path, start_year: int, target_index: pd.DatetimeIndex
) -> pd.Series:
    """Bouw een 15-min UTC day-ahead serie die past bij `target_index`.

    Voor start_year=Y is de brondata mei Y tot mei Y+1. Elk target-tijdstip
    krijgt de prijs van het historische tijdstip met dezelfde (maand, dag, uur,
    minuut). Robuust tegen DST-drift en schrikkeldagen.
    """
    if start_year == 2025:
        # Live cache voor het load-venster. EUR/MWh, uurlijks, Europe/Amsterdam.
        live = pd.read_parquet(cache_dir / "20250501_20260501_da.parquet")["price"]
        hist = (live.astype("float64") / 1000.0).tz_convert("UTC")
    else:
        # Combineer de twee kalenderjaarbestanden voor mei Y tot mei Y+1.
        hist = pd.concat(
            [
                _load_calendar_year_da(cache_dir, start_year),
                _load_calendar_year_da(cache_dir, start_year + 1),
            ]
        ).sort_index()
        hist = hist[~hist.index.duplicated(keep="first")]

    # Forward-fill naar een continu 15-min UTC grid dat precies het mei-mei
    # venster dekt. Geen buffer: de calendar-key lookup pakt de eerste hit per
    # (maand, dag, uur, minuut) en buffer-dagen zouden de echte rand maskeren.
    load_first = target_index[0]
    load_last = target_index[-1]
    hist_first = load_first.replace(year=start_year)
    hist_last = load_last.replace(year=start_year + 1)
    hist_grid = pd.date_range(
        start=hist_first,
        end=hist_last + pd.Timedelta("15min"),
        freq="15min",
        tz="UTC",
        inclusive="left",
    )
    hist_15min = hist.reindex(hist_grid, method="ffill").ffill().bfill()

    # Calendar-key lookup: (maand, dag, uur, minuut) is gelijk in load- en
    # historisch jaar (beide UTC).
    keys = (
        hist_15min.index.month.values * 100_00_00
        + hist_15min.index.day.values * 10_000
        + hist_15min.index.hour.values * 100
        + hist_15min.index.minute.values
    )
    lookup = pd.Series(hist_15min.values, index=keys)
    # Eerste hit wint bij duplicaten (DST-fallback uur komt tweemaal voor).
    lookup = lookup[~lookup.index.duplicated(keep="first")]

    target_keys = (
        target_index.month.values * 100_00_00
        + target_index.day.values * 10_000
        + target_index.hour.values * 100
        + target_index.minute.values
    )
    aligned = lookup.reindex(target_keys)
    if aligned.isna().any():
        # Veiligheidsnet voor DST-rand-minuten die niet exact matchen.
        aligned = aligned.ffill().bfill()

    return pd.Series(
        aligned.values.astype("float64"), index=target_index, name="day_ahead"
    )


def _imbalance_cache_path(cache_dir: Path, start_year: int) -> Path:
    return cache_dir / f"{start_year}0501_{start_year + 1}0501_imbalance_entsoe.parquet"


def _build_entsoe_imbalance_for_span(
    cache_dir: Path,
    start_year: int,
    target_index: pd.DatetimeIndex,
    entsoe_api_key: str | None = None,
) -> pd.Series | None:
    """Bouw een 15-min UTC ENTSO-E-onbalansserie voor de mei→mei-spanne.

    De normale prijsloader cached ENTSO-E-onbalans als
    `<start>_<end>_imbalance_entsoe.parquet`. Monte Carlo hergebruikt die cache
    en past dezelfde calendar-key alignment toe als voor day-ahead, zodat het
    actuele loadprofiel tegen het historische prijsregime wordt afgespeeld.
    """
    cache = _imbalance_cache_path(cache_dir, start_year)
    if not cache.exists():
        if not entsoe_api_key:
            return None
        settings = Settings(cache_dir=cache_dir, entsoe_api_key=entsoe_api_key)
        start = datetime(start_year, 5, 1, tzinfo=timezone.utc)
        end = datetime(start_year + 1, 5, 1, tzinfo=timezone.utc)
        try:
            _fetch_entsoe_imbalance(settings, start, end)
        except Exception as e:  # noqa: BLE001
            print(
                f"[mc] ENTSO-E onbalans {start_year}-{start_year + 1} "
                f"niet beschikbaar ({e}); gebruik synthetisch."
            )
            return None

    hist = pd.read_parquet(cache)["imbalance"].astype("float64").tz_convert("UTC")
    hist = hist[~hist.index.duplicated(keep="first")].sort_index()

    load_first = target_index[0]
    load_last = target_index[-1]
    hist_first = load_first.replace(year=start_year)
    hist_last = load_last.replace(year=start_year + 1)
    hist_grid = pd.date_range(
        start=hist_first,
        end=hist_last + pd.Timedelta("15min"),
        freq="15min",
        tz="UTC",
        inclusive="left",
    )
    hist_15min = hist.reindex(hist_grid).ffill().bfill()

    keys = (
        hist_15min.index.month.values * 100_00_00
        + hist_15min.index.day.values * 10_000
        + hist_15min.index.hour.values * 100
        + hist_15min.index.minute.values
    )
    lookup = pd.Series(hist_15min.values, index=keys)
    lookup = lookup[~lookup.index.duplicated(keep="first")]

    target_keys = (
        target_index.month.values * 100_00_00
        + target_index.day.values * 10_000
        + target_index.hour.values * 100
        + target_index.minute.values
    )
    aligned = lookup.reindex(target_keys)
    if aligned.isna().any():
        aligned = aligned.ffill().bfill()

    return pd.Series(
        aligned.values.astype("float64"), index=target_index, name="imbalance"
    )


def build_prices_for_year(
    cache_dir: Path,
    year: int,
    target_index: pd.DatetimeIndex,
    entsoe_api_key: str | None = None,
) -> Prices:
    """Pak day-ahead serie + historische of synthetische onbalans in als `Prices`."""
    day_ahead = _build_da_for_span(cache_dir, year, target_index)
    imbalance = _build_entsoe_imbalance_for_span(
        cache_dir, year, target_index, entsoe_api_key=entsoe_api_key
    )
    imbalance_source = "entsoe" if imbalance is not None else "synthetic"
    if imbalance is None:
        imbalance = synthesize_imbalance(day_ahead)
    return Prices(
        day_ahead=day_ahead,
        imbalance=imbalance.astype("float64"),
        source=f"entsoe-{year}",
        imbalance_source=imbalance_source,
    )


# Worker (ProcessPoolExecutor).


def _run_year_batch(
    args: tuple[
        int, str, LoadSeries, Prices, BatterySpec,
        list[tuple[str, str, str]], dict[str, TariffParams],
    ],
) -> YearResult:
    """Eén worker = één jaar volledige scenario-sweep.

    Pickle-kosten schalen met LoadSeries + Prices (~1-2 MB). Door alle
    scenario's per jaar in dezelfde worker te doen amortiseren we dat over
    ~45 ms compute elk, ruim boven de pickle-overhead.
    """
    year, da_source, load, prices, spec, scenarios, tariffs = args
    annual_cost_by_scenario: dict[str, float] = {}
    for scenario_name, strategy_kind, tariff_key in scenarios:
        result = run_scenario(
            scenario_name,
            strategy_kind,
            load,
            prices,
            tariffs[tariff_key],
            spec,
            include_detail=False,  # detail-frames blazen de return-payload op
        )
        annual_cost_by_scenario[scenario_name] = result.annual_cost_eur
    return YearResult(
        year=year,
        da_source=da_source,
        imbalance_source=prices.imbalance_source,
        annual_cost_by_scenario=annual_cost_by_scenario,
    )


# Publieke entry point.


def run_monte_carlo(
    load: LoadSeries,
    spec: BatterySpec,
    cache_dir: Path,
    tariffs: dict[str, TariffParams],
    *,
    n_samples: int,
    workers: int,
    seed: int = 42,
    entsoe_api_key: str | None = None,
) -> MonteCarloResult:
    """Draai alle historische jaren deterministisch en bootstrap N samples.

    Elk historisch jaar wordt precies één keer gesimuleerd. De N samples
    indexen in die tabel met teruglegging, dus geen extra rekentijd.
    """
    target_index = load.consumption_kwh.index
    years = discover_price_years(cache_dir)
    if not years:
        raise RuntimeError(
            "Geen ENTSO-E NL day-ahead jaarcaches gevonden in "
            f"{cache_dir}; verwachte bestanden: da_NL_<jaar>.parquet."
        )

    print(
        f"[mc] gevonden: {len(years)} historische jaren {years[0]}-{years[-1]} "
        f"({len(years)} mei→mei spannes)"
    )

    # Bouw alle jaarprijzen in het parent-proces; goedkoop en voorkomt dubbele I/O.
    print("[mc] bouw per-jaar prijsseries ...")
    scenarios = _default_scenarios(tariffs)
    tasks = []
    for year in years:
        prices = build_prices_for_year(
            cache_dir, year, target_index, entsoe_api_key=entsoe_api_key
        )
        tasks.append((year, prices.source, load, prices, spec, scenarios, tariffs))

    print(
        f"[mc] {len(tasks)} jaar × {len(scenarios)} scenario's "
        f"= {len(tasks) * len(scenarios)} sims over {workers} worker(s) ..."
    )
    start_time = time.perf_counter()
    year_results: list[YearResult] = []
    if workers <= 1:
        for year_task in tasks:
            print(f"[mc]   jaar {year_task[0]} ...")
            year_results.append(_run_year_batch(year_task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_year_batch, year_task): year_task[0]
                for year_task in tasks
            }
            for future in as_completed(futures):
                year = futures[future]
                year_results.append(future.result())
                print(f"[mc]   jaar {year} klaar")
    walltime = time.perf_counter() - start_time

    year_results.sort(key=lambda year_result: year_result.year)
    random_generator = np.random.default_rng(seed)
    sample_picks = random_generator.choice(len(years), size=n_samples, replace=True)

    # Statistieken per scenario uit de N samples.
    stats: dict[str, ScenarioStats] = {}
    for scenario_name, _, _ in scenarios:
        per_year = np.array(
            [
                year_result.annual_cost_by_scenario[scenario_name]
                for year_result in year_results
            ],
            dtype="float64",
        )
        sampled = per_year[sample_picks]
        stats[scenario_name] = ScenarioStats(
            name=scenario_name,
            mean_eur=float(np.mean(sampled)),
            std_eur=float(np.std(sampled, ddof=1)) if len(sampled) > 1 else 0.0,
            p10_eur=float(np.percentile(sampled, 10)),
            p50_eur=float(np.percentile(sampled, 50)),
            p90_eur=float(np.percentile(sampled, 90)),
            min_eur=float(np.min(sampled)),
            max_eur=float(np.max(sampled)),
            n_samples=len(sampled),
        )

    return MonteCarloResult(
        n_samples=n_samples,
        years_used=years,
        year_results=year_results,
        sample_year_picks=[int(sample_pick) for sample_pick in sample_picks.tolist()],
        scenario_stats=stats,
        walltime_seconds=walltime,
        workers=workers,
    )
