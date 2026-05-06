"""Hoofdingang: data laden, prijzen ophalen, scenario's draaien, rapport schrijven."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

from .battery import BatterySpec
from .config import load_settings
from .data import LoadSeries, load_series, summary
from .economics import TariffParams, build_predefined
from .montecarlo import run_monte_carlo
from .prices import Prices, fetch_or_synthesize
from .questdb import QuestDB
from .report import render, render_monte_carlo, render_sweep
from .simulate import ScenarioResult, run_scenario
from .sweep import run_sweep
from .userconfig import load_user_config


def _parse_utc_datetime(value: str) -> datetime:
    """Parseer ISO-datum/-tijd en behoud eventuele offset correct in UTC."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _run_one(
    args: tuple[str, str, TariffParams, BatterySpec, LoadSeries, Prices],
) -> ScenarioResult:
    """Worker-entry. Moet top-level zijn voor ProcessPoolExecutor-pickling."""
    name, kind, tariff, spec, load, prices = args
    return run_scenario(name, kind, load, prices, tariff, spec)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Thuisbatterij-scenariosimulator")
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--capacity", type=float, default=None, help="Nominale capaciteit in kWh")
    p.add_argument("--max-power", type=float, default=None, help="AC laad/ontlaadvermogen in kW")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Pad naar user TOML. Default: $THUISBAT_CONFIG, "
        "config/user.toml of config/user.example.toml.",
    )
    p.add_argument(
        "--sweep-capacity",
        action="store_true",
        help="Draai gevoeligheidsanalyse over [5, 8, 10, 12, 15, 20, 25, 30] kWh "
        "en schrijf output/sensitivity.md.",
    )
    p.add_argument(
        "--monte-carlo",
        type=int,
        default=0,
        metavar="N",
        help="Speel load af tegen elk historisch jaar in cache "
        "(cache/da_NL_*.parquet) en bootstrap N samples voor mean/p10/p90/std "
        "per scenario. Schrijft output/monte_carlo.md.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Aantal worker-processen. Default: cpu_count-1. 1 schakelt parallel uit.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    user = load_user_config(args.config)
    settings = load_settings(user)
    db = QuestDB(settings.questdb_url)

    start_str = args.start or user.simulation.start
    end_str = args.end or user.simulation.end
    start = _parse_utc_datetime(start_str)
    end = _parse_utc_datetime(end_str)

    capacity = args.capacity if args.capacity is not None else user.battery.capacity_kwh
    max_power = args.max_power if args.max_power is not None else user.battery.max_charge_kw

    tariffs = build_predefined(
        user.contracts,
        user.grid,
        user.saldering,
        # Curtailment-scenario voor "tibber" als die bestaat.
        curtail_pairs={"tibber": -0.02} if "tibber" in user.contracts else {},
    )

    print(f"[load] data uit QuestDB {start} -> {end} (cache: {settings.cache_dir})")
    load = load_series(db, start, end, cache_dir=settings.cache_dir)
    print("[load]", summary(load))

    print("[prices] ophalen of synthetiseren ...")
    prices = fetch_or_synthesize(settings, start, end)
    print(f"[prices] day-ahead source: {prices.source}")
    print(f"[prices] imbalance source: {prices.imbalance_source}")
    using_synthetic = prices.source == "synthetic"

    if args.monte_carlo > 0:
        spec = BatterySpec(
            capacity_kwh=capacity,
            max_charge_kw=max_power,
            max_discharge_kw=max_power,
        )
        mc = run_monte_carlo(
            load,
            spec,
            settings.cache_dir,
            tariffs,
            n_samples=args.monte_carlo,
            workers=args.workers,
            entsoe_api_key=settings.entsoe_api_key,
        )
        out = render_monte_carlo(mc, settings.output_dir)
        print(f"[done] {out}")
        print(
            f"[mc] walltime {mc.walltime_seconds:.1f}s, "
            f"{len(mc.years_used)} years × {len(mc.year_results[0].annual_cost_by_scenario)} "
            f"scenarios = {len(mc.years_used) * len(mc.year_results[0].annual_cost_by_scenario)} sims, "
            f"bootstrapped to N={mc.n_samples}"
        )
        for name, st in sorted(mc.scenario_stats.items(), key=lambda kv: kv[1].mean_eur):
            print(
                f"  {name:42s} mean €{st.mean_eur:7,.0f}  "
                f"p10 €{st.p10_eur:7,.0f}  p90 €{st.p90_eur:7,.0f}  "
                f"std €{st.std_eur:5,.0f}"
            )
        return

    if args.sweep_capacity:
        print(f"[sweep] capaciteits-sweep met {args.workers} worker(s) ...")
        sweep = run_sweep(load, prices, tariffs, workers=args.workers)
        out = render_sweep(
            sweep,
            settings.output_dir,
            using_synthetic_prices=using_synthetic,
        )
        print(f"[done] {out}")
        for r in sweep.rows:
            print(
                f"  {r.capacity_kwh:5.1f} kWh  "
                f"capex €{r.capex_eur:6,.0f}  "
                f"savings €{r.avg_annual_savings_eur:5,.0f}/yr  "
                f"payback {r.payback_years:5.1f} yr"
            )
        return

    spec = BatterySpec(
        capacity_kwh=capacity,
        max_charge_kw=max_power,
        max_discharge_kw=max_power,
    )

    # (display_name, strategy_kind, tariff_key). De tariff_key is de
    # contract-display_name (met optionele suffix), opgezocht in de registry
    # die uit de user-config is gebouwd.
    has_tibber = "tibber" in user.contracts
    has_frank = "frank" in user.contracts
    has_perfect = "perfect" in user.contracts
    fixed_name = user.contracts["fixed"].display_name if "fixed" in user.contracts else "baseline-fixed"
    tibber_name = user.contracts["tibber"].display_name if has_tibber else None
    frank_name = user.contracts["frank"].display_name if has_frank else None
    perfect_name = user.contracts["perfect"].display_name if has_perfect else None

    scenarios: list[tuple[str, str, str]] = [
        # Saldering-tijdperk (huidig contract):
        (fixed_name, "no_battery", fixed_name),
        (f"{fixed_name}-postsaldering", "no_battery", f"{fixed_name}-postsaldering"),
        ("battery-pv-only", "pv_self", fixed_name),
    ]
    if has_tibber:
        scenarios += [
            ("dynamic-no-battery", "no_battery", tibber_name),
            (tibber_name, "day_ahead", tibber_name),
            (
                "dynamic-no-battery-postsaldering",
                "no_battery",
                f"{tibber_name}-postsaldering",
            ),
            (
                f"{tibber_name}-postsaldering",
                "day_ahead",
                f"{tibber_name}-postsaldering",
            ),
            (
                f"{tibber_name}-curtail-postsaldering",
                "day_ahead",
                f"{tibber_name}-curtail-postsaldering",
            ),
            (
                "dynamic-curtail-no-battery-postsaldering",
                "no_battery",
                f"{tibber_name}-curtail-postsaldering",
            ),
            # Globaal optimale LP-dispatch.
            (f"{tibber_name}-lp-saldering", "lp", tibber_name),
            (f"{tibber_name}-lp-postsaldering", "lp", f"{tibber_name}-postsaldering"),
            (f"{tibber_name}-perfect-saldering", "perfect", tibber_name),
        ]
    if has_frank:
        scenarios += [
            (frank_name, "imbalance", frank_name),
            (f"{frank_name}-postsaldering", "imbalance", f"{frank_name}-postsaldering"),
        ]
    if has_perfect:
        scenarios += [
            (
                f"{perfect_name}-postsaldering",
                "perfect",
                f"{perfect_name}-postsaldering",
            ),
        ]

    print(f"[run] {len(scenarios)} scenario's over {args.workers} worker(s) ...")
    tasks = [
        (name, kind, tariffs[key], spec, load, prices) for name, kind, key in scenarios
    ]
    results: list[ScenarioResult] = []
    if args.workers <= 1:
        for t in tasks:
            print(f"[run] {t[0]} ({t[1]}) ...")
            results.append(_run_one(t))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_one, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                name = futures[fut]
                results.append(fut.result())
                print(f"[run] {name} done")

    out = render(
        results, load, settings.output_dir, using_synthetic_prices=using_synthetic
    )
    print(f"[done] {out}")
    for r in sorted(results, key=lambda x: x.annual_cost_eur):
        print(f"  {r.name:24s} €{r.annual_cost_eur:8,.0f}/yr")


if __name__ == "__main__":
    main()
