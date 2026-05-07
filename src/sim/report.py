"""Schrijf markdown-samenvatting en per-scenario CSV-traces."""


from pathlib import Path

from .data import LoadSeries
from .montecarlo import MonteCarloResult, ScenarioStats
from .simulate import ScenarioResult
from .sweep import (
    DEFAULT_MARGINAL_PAYBACK_THRESHOLD_YEARS,
    HORIZON_YEARS,
    SweepResult,
    lowest_tco_row,
    roi_optimal_floor,
)


def render(
    results: list[ScenarioResult],
    load: LoadSeries,
    output_dir: Path,
    *,
    using_synthetic_prices: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Baseline = duurste scenario (geen batterij, vast tarief).
    baseline = max(results, key=lambda result: result.annual_cost_eur)

    lines: list[str] = []
    lines.append("# Simulatieresultaten")
    lines.append("")
    if using_synthetic_prices:
        lines.append(
            "> **Let op:** op synthetische prijzen (geen live bron). "
            "Absolute getallen zijn indicatief; relatieve rangschikking is nog informatief."
        )
        lines.append("")
    lines.append(f"Venster: {load.consumption_kwh.index.min()} → {load.consumption_kwh.index.max()}")
    lines.append("")
    lines.append(
        f"Totaal verbruik: {load.consumption_kwh.sum():.0f} kWh, "
        f"ZP: {load.pv_kwh.sum():.0f} kWh, "
        f"gap-fill buckets: {len(load.gap_filled_index)} "
        f"({len(load.gap_filled_index) / max(1, len(load.consumption_kwh)):.1%})"
    )
    lines.append("")

    lines.append("## Jaarkosten per scenario")
    lines.append("")
    lines.append("| Scenario | Jaarkosten (€) | vs baseline | Net-import (kWh) | Net-export (kWh) | Batterij-doorzet (kWh) |")
    lines.append("|---|---|---|---|---|---|")
    for scenario_result in sorted(
        results, key=lambda result: result.annual_cost_eur, reverse=True
    ):
        delta = scenario_result.annual_cost_eur - baseline.annual_cost_eur
        lines.append(
            f"| {scenario_result.name} | {scenario_result.annual_cost_eur:,.0f} | "
            f"{delta:+,.0f} | "
            f"{scenario_result.breakdown['grid_import_kwh_total']:,.0f} | "
            f"{scenario_result.breakdown['grid_export_kwh_total']:,.0f} | "
            f"{scenario_result.breakdown['battery_throughput_kwh']:,.0f} |"
        )
    lines.append("")

    lines.append("## Cashflow-uitsplitsing")
    lines.append("")
    lines.append("| Scenario | Importkosten | Exportopbrengst | Onbalans-extra | Vaste kosten | Fees/heffingen |")
    lines.append("|---|---|---|---|---|---|")
    for scenario_result in results:
        breakdown = scenario_result.breakdown
        lines.append(
            f"| {scenario_result.name} | {breakdown['import_cost']:,.0f} | "
            f"{breakdown['export_revenue']:,.0f} | "
            f"{breakdown['imbalance_extra']:,.0f} | "
            f"{breakdown['standing_charges']:,.0f} | "
            f"{breakdown['service_fees_and_penalties']:,.0f} |"
        )
    lines.append("")

    lines.append("## Jaarbesparing vs huidige situatie (baseline-fixed, saldering aan)")
    lines.append("")
    current_baseline = next(
        result for result in results if result.name == "baseline-fixed"
    )
    for scenario_result in results:
        if scenario_result.name == "baseline-fixed":
            continue
        savings = current_baseline.annual_cost_eur - scenario_result.annual_cost_eur
        lines.append(f"- **{scenario_result.name}**: {savings:+,.0f} €/yr")
    lines.append("")

    # 15-jarige mix: 1 jaar saldering + 14 jaar post-saldering.
    # Vergelijk per strategie tegen de no-battery mix.
    saldering_era = {
        result.name: result for result in results if "postsaldering" not in result.name
    }
    post_era = {
        result.name: result for result in results if "postsaldering" in result.name
    }
    pairs = [
        ("tibber-day-ahead", "tibber-day-ahead-postsaldering", "Tibber day-ahead"),
        ("frank-imbalance", "frank-imbalance-postsaldering", "Frank imbalance"),
    ]
    base_pre = saldering_era["baseline-fixed"].annual_cost_eur
    base_post = post_era["baseline-fixed-postsaldering"].annual_cost_eur
    base_blended = base_pre * 1 + base_post * 14
    lines.append(
        "## 15-jarige mix-outlook (1 jr saldering + 14 jr post)"
    )
    lines.append("")
    lines.append(
        f"Baseline (geen batterij, vast→dynamisch in 2027): **€{base_blended:,.0f}** totaal over 15 jr"
    )
    lines.append("")
    lines.append("| Strategie | 1 jr saldering | 14 jr post | 15 jr totaal | vs. baseline |")
    lines.append("|---|---|---|---|---|")
    for pre_scenario_name, post_scenario_name, label in pairs:
        if pre_scenario_name not in saldering_era or post_scenario_name not in post_era:
            continue
        pre_cost = saldering_era[pre_scenario_name].annual_cost_eur
        post_cost = post_era[post_scenario_name].annual_cost_eur
        total = pre_cost + 14 * post_cost
        delta = base_blended - total
        lines.append(
            f"| {label} | €{pre_cost:,.0f} | €{post_cost:,.0f}/yr × 14 | €{total:,.0f} | {delta:+,.0f} |"
        )
    lines.append("")

    out = output_dir / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")

    # Per-scenario CSV.
    for scenario_result in results:
        slim = scenario_result.detail.copy()
        slim.index.name = "timestamp"
        slim.to_csv(output_dir / f"{scenario_result.name}.csv", float_format="%.4f")

    return out


def _fmt_years(years: float) -> str:
    return f"{years:.1f}" if years != float("inf") else "∞"


def render_sweep(
    sweep: SweepResult,
    output_dir: Path,
    *,
    using_synthetic_prices: bool,
    marginal_threshold_years: float = DEFAULT_MARGINAL_PAYBACK_THRESHOLD_YEARS,
) -> Path:
    """Schrijf de capaciteits-gevoeligheidstabel naar `output/sensitivity.md`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = sweep.rows
    floor = roi_optimal_floor(rows, threshold_years=marginal_threshold_years)
    best = min(rows, key=lambda row: row.payback_years)
    lowest_capex_row = min(rows, key=lambda row: row.capex_eur)
    tco_best = lowest_tco_row(rows)
    best_tco_payback = min(rows, key=lambda row: row.tco_payback_years)

    lines: list[str] = []
    lines.append("# Capaciteits-gevoeligheidsanalyse")
    lines.append("")
    if using_synthetic_prices:
        lines.append(
            "> **Let op:** synthetische prijzen. Absolute payback indicatief; "
            "relatieve rangschikking blijft informatief."
        )
        lines.append("")
    lines.append(
        "Strategie: `frank-imbalance` (saldering aan, jr 1) → "
        "`frank-imbalance-postsaldering` (saldering uit, jr 2-15)."
    )
    lines.append("")
    lines.append(
        f"Geen-batterij baseline: €{sweep.baseline_pre_eur:,.0f}/jr saldering, "
        f"€{sweep.baseline_post_eur:,.0f}/jr post → "
        f"€{sweep.baseline_blended_eur:,.0f} over 15 jr."
    )
    lines.append("")
    lines.append(
        "Capex-model: cellen €100/kWh + €630 vast (BMS + balancer + rack + "
        "installatie). De €630 is sunk ongeacht grootte; marginale cel-payback "
        "gebruikt alleen de per-kWh-cost (€100)."
    )
    lines.append("")
    lines.append(
        "| Capaciteit (kWh) | Capex (€) | Pre €/jr | Post €/jr | "
        "15-jr mix (€) | Gem. besparing (€/jr) | Payback (jr) | "
        "€/kWh geïnstalleerd | Marginaal €/jr·kWh | Marginale kWh payback (jr) |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    for sweep_row in rows:
        lines.append(
            f"| {sweep_row.capacity_kwh:.1f} | {sweep_row.capex_eur:,.0f} | "
            f"{sweep_row.pre_annual_eur:,.0f} | {sweep_row.post_annual_eur:,.0f} | "
            f"{sweep_row.blended_15yr_eur:,.0f} | "
            f"{sweep_row.avg_annual_savings_eur:,.0f} | "
            f"{_fmt_years(sweep_row.payback_years)} | "
            f"{sweep_row.eur_per_kwh_installed:,.0f} | "
            f"{sweep_row.marginal_savings_eur_per_kwh:,.1f} | "
            f"{_fmt_years(sweep_row.marginal_payback_years)} |"
        )
    lines.append("")
    lines.append(
        "Marginale kolommen vergelijken elke rij met de vorige; eerste rij "
        "tegen no-battery. Marginale kWh-payback rekent alleen €100/kWh "
        "cellen, niet de vaste overhead."
    )
    lines.append("")

    lines.append(f"## Veroudering en {HORIZON_YEARS}-jaars TCO")
    lines.append("")
    lines.append(
        "Cycluslevensduur uit power-law fit op 6000 cycli naar 80% bij 80% DoD "
        "en 0.5C referentie. Kalenderondergrens 14 jr bij typische "
        "meterkasttemperatuur. Vervangingskosten: €100/kWh cellen + €350 BMS "
        "elke 2e swap. TCO = capex + nominale som van vervangingen."
    )
    lines.append("")
    lines.append(
        "| Capaciteit (kWh) | EFC/jr | Piek C-rate | Jaren tot EOL | "
        "Vervangingen in 15 jr | Vervangingskost (€) | "
        f"**TCO {HORIZON_YEARS} jr (€)** | TCO-payback (jr) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for sweep_row in rows:
        lines.append(
            f"| {sweep_row.capacity_kwh:.1f} | {sweep_row.annual_efc:,.0f} | "
            f"{sweep_row.peak_c_rate:.2f} | {_fmt_years(sweep_row.years_to_eol)} | "
            f"{sweep_row.replacements_in_horizon} | "
            f"{sweep_row.replacement_cost_eur:,.0f} | "
            f"**{sweep_row.tco_15yr_eur:,.0f}** | "
            f"{_fmt_years(sweep_row.tco_payback_years)} |"
        )
    lines.append("")
    lines.append(
        f"**Beste totale payback (capex-only):** {best.capacity_kwh:.1f} kWh op "
        f"{_fmt_years(best.payback_years)} jr "
        f"(€{best.avg_annual_savings_eur:,.0f}/jr gem. besparing, "
        f"€{best.capex_eur:,.0f} capex)."
    )
    lines.append("")
    lines.append(
        f"**Laagste capex:** {lowest_capex_row.capacity_kwh:.1f} kWh op "
        f"€{lowest_capex_row.capex_eur:,.0f}. Met "
        f"{lowest_capex_row.replacements_in_horizon} vervanging(en) over 15 jr "
        f"is de TCO €{lowest_capex_row.tco_15yr_eur:,.0f}."
    )
    lines.append("")
    lines.append(
        f"**Laagste 15-jr TCO na besparingen:** {tco_best.capacity_kwh:.1f} kWh "
        f"(TCO €{tco_best.tco_15yr_eur:,.0f}, "
        f"€{tco_best.avg_annual_savings_eur:,.0f}/jr gem. besparing, "
        f"TCO-payback {_fmt_years(tco_best.tco_payback_years)} jr, "
        f"{tco_best.replacements_in_horizon} vervanging(en))."
    )
    lines.append("")
    if tco_best.capacity_kwh != lowest_capex_row.capacity_kwh:
        lines.append(
            "Laagste capex en laagste TCO verschillen: vervangingen eten de "
            "besparing van het kleinere pack op."
        )
    else:
        lines.append(
            "Laagste capex valt samen met laagste TCO. Geen formaat in de "
            "sweep heeft vervanging vóór de horizon nodig."
        )
    lines.append("")
    lines.append(
        f"**Beste TCO-payback:** {best_tco_payback.capacity_kwh:.1f} kWh op "
        f"{_fmt_years(best_tco_payback.tco_payback_years)} jr "
        f"(TCO €{best_tco_payback.tco_15yr_eur:,.0f})."
    )
    lines.append("")
    lines.append(
        f"**RoI-optimale capaciteit ({marginal_threshold_years:.0f}-jr "
        f"marginale drempel):** {floor.capacity_kwh:.1f} kWh, grootste maat "
        f"waarvan de laatste-kWh payback nog ≤ {marginal_threshold_years:.0f} jr is "
        f"({_fmt_years(floor.marginal_payback_years)} jr in deze rij). "
        f"Capex €{floor.capex_eur:,.0f}, "
        f"gem. besparing €{floor.avg_annual_savings_eur:,.0f}/jr, "
        f"totale payback {_fmt_years(floor.payback_years)} jr."
    )
    lines.append("")
    lines.append(
        "Boven dit formaat kost extra capaciteit meer dan het terugverdient "
        f"binnen {marginal_threshold_years:.0f} jr arbitrage. Eronder laat je "
        "winst liggen."
    )
    lines.append("")

    out = output_dir / "sensitivity.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def render_monte_carlo(mc: MonteCarloResult, output_dir: Path) -> Path:
    """Schrijf de Monte Carlo verdelingstabel naar `output/monte_carlo.md`."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pre_baseline = mc.scenario_stats.get("baseline-fixed")
    post_baseline = mc.scenario_stats.get("baseline-fixed-postsaldering")
    imbalance_sources = sorted(
        {year_result.imbalance_source for year_result in mc.year_results}
    )
    if imbalance_sources == ["entsoe"]:
        imbalance_note = "Onbalans komt uit historische ENTSO-E-cache."
    elif imbalance_sources == ["synthetic"]:
        imbalance_note = "Onbalans is gesynthetiseerd uit het day-ahead-verloop."
    else:
        imbalance_note = (
            "Onbalans is deels historisch uit ENTSO-E-cache en deels gesynthetiseerd "
            "uit het day-ahead-verloop."
        )

    def _is_post(scenario_name: str) -> bool:
        return "postsaldering" in scenario_name

    def _baseline_for(scenario_name: str) -> ScenarioStats | None:
        return post_baseline if _is_post(scenario_name) else pre_baseline

    lines: list[str] = []
    lines.append("# Monte Carlo over historische prijsjaren")
    lines.append("")
    lines.append(
        f"Load: mei 2025 tot mei 2026 (constant). Prijzen: ENTSO-E NL "
        f"day-ahead afgespeeld vanuit {len(mc.years_used)} historische "
        f"kalenderjaren ({mc.years_used[0]} tot {mc.years_used[-1]}), "
        f"kalender-uitgelijnd op (maand, dag, uur, minuut). "
        f"{imbalance_note}"
    )
    lines.append("")
    lines.append(
        f"Bootstrap: **N={mc.n_samples}** samples met teruglegging uit de "
        f"{len(mc.year_results)}-jarige deterministische pool. Walltime: "
        f"**{mc.walltime_seconds:.1f}s** over {mc.workers} worker(s) "
        f"({len(mc.year_results) * len(mc.year_results[0].annual_cost_by_scenario)} "
        f"onderliggende scenario-runs)."
    )
    lines.append("")

    lines.append("## Jaarkostenverdeling per scenario")
    lines.append("")
    lines.append(
        "| Scenario | Gem. (€/jr) | Std (€) | p10 | p50 | p90 | min | max | Robuust vs niets-doen |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for scenario_name, _, _ in [
        ("baseline-fixed", "", ""),
        ("dynamic-no-battery", "", ""),
        ("battery-pv-only", "", ""),
        ("tibber-day-ahead", "", ""),
        ("frank-imbalance", "", ""),
        ("baseline-fixed-postsaldering", "", ""),
        ("dynamic-no-battery-postsaldering", "", ""),
        ("tibber-day-ahead-postsaldering", "", ""),
        ("tibber-curtail-postsaldering", "", ""),
        ("dynamic-curtail-no-battery-postsaldering", "", ""),
        ("frank-imbalance-postsaldering", "", ""),
        ("perfect-foresight-postsaldering", "", ""),
    ]:
        if scenario_name not in mc.scenario_stats:
            continue
        scenario_stats = mc.scenario_stats[scenario_name]
        baseline_stats = _baseline_for(scenario_name)
        # "Robuust" = scenario p90 ligt onder de gem. van de baseline. Betekent:
        # zelfs in de ongelukkige 10% van de prijsjaren ben je goedkoper dan
        # wat baseline gemiddeld doet.
        if baseline_stats is None or scenario_name == baseline_stats.name:
            robust = "-"
        elif scenario_stats.p90_eur < baseline_stats.mean_eur:
            robust = (
                f"✓ p90 < gem. {baseline_stats.name} "
                f"(€{baseline_stats.mean_eur:,.0f})"
            )
        else:
            robust = (
                f"✗ p90 (€{scenario_stats.p90_eur:,.0f}) ≥ "
                f"gem. {baseline_stats.name}"
            )
        lines.append(
            f"| {scenario_name} | {scenario_stats.mean_eur:,.0f} | "
            f"{scenario_stats.std_eur:,.0f} | {scenario_stats.p10_eur:,.0f} | "
            f"{scenario_stats.p50_eur:,.0f} | {scenario_stats.p90_eur:,.0f} | "
            f"{scenario_stats.min_eur:,.0f} | {scenario_stats.max_eur:,.0f} | "
            f"{robust} |"
        )
    lines.append("")

    # Per-jaar uitsplitsing: zichtbaar welke jaren de staart drijven.
    lines.append("## Per-jaar deterministische resultaten")
    lines.append("")
    scenario_names = [scenario_name for scenario_name, _, _ in (
        ("baseline-fixed", "", ""),
        ("dynamic-no-battery", "", ""),
        ("battery-pv-only", "", ""),
        ("tibber-day-ahead", "", ""),
        ("frank-imbalance", "", ""),
        ("baseline-fixed-postsaldering", "", ""),
        ("dynamic-no-battery-postsaldering", "", ""),
        ("tibber-day-ahead-postsaldering", "", ""),
        ("tibber-curtail-postsaldering", "", ""),
        ("dynamic-curtail-no-battery-postsaldering", "", ""),
        ("frank-imbalance-postsaldering", "", ""),
        ("perfect-foresight-postsaldering", "", ""),
    ) if scenario_name in mc.scenario_stats]
    lines.append("Jaarkosten (€) per prijsjaar en scenario.")
    lines.append("")
    header = "| Jaar | " + " | ".join(scenario_names) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(scenario_names) + 1))
    for year_result in mc.year_results:
        cells = [f"{year_result.year}-{(year_result.year + 1) % 100:02d}"]
        for scenario_name in scenario_names:
            cells.append(f"{year_result.annual_cost_by_scenario[scenario_name]:,.0f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Rangschikking op gemiddelde kosten.
    lines.append("## Scenario's gerangschikt op gem. kosten")
    lines.append("")
    ranked = sorted(mc.scenario_stats.values(), key=lambda stats: stats.mean_eur)
    lines.append("| Rang | Scenario | Gem. (€/jr) | Spread p90-p10 (€) |")
    lines.append("|---|---|---|---|")
    for rank, scenario_stats in enumerate(ranked, 1):
        spread = scenario_stats.p90_eur - scenario_stats.p10_eur
        lines.append(
            f"| {rank} | {scenario_stats.name} | "
            f"{scenario_stats.mean_eur:,.0f} | {spread:,.0f} |"
        )
    lines.append("")
    lines.append(
        "Spread p90-p10 meet de prijsgevoeligheid. Smal = robuust tegen het "
        "historische prijsregime. Breed = de uitkomst hangt sterk af van welk "
        "prijsjaar je raakt."
    )
    lines.append("")

    out = output_dir / "monte_carlo.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
