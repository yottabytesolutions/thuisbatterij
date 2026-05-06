"""Capaciteits-gevoeligheidsanalyse voor de batterij.

Draait de `frank-imbalance` strategie over een reeks nominale capaciteiten en
berekent simpele terugverdientijd tegen de no-battery baseline op een
15-jarige horizon (1 jaar saldering + 14 jaar post-saldering).

Capex-model:
  capex_delta = capacity_kwh * €100   (CN LFP-cellen, marginaal)
              + €630                  (BMS + balancer + rack + installatie)

Marginale kolommen tonen waar extra kWh's zichzelf niet meer terugverdienen:
een marginale cel kost €100, dus break-even zodra die kWh ≥ €100/drempeljaren
per jaar oplevert.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from .aging import AgingModel, CycleProfile, replacement_cost, replacement_schedule, years_to_eol
from .battery import BatterySpec
from .data import LoadSeries
from .economics import TariffParams
from .prices import Prices
from .simulate import ScenarioResult, run_scenario
from .strategies import DispatchTuning, _build_dispatch_context

SWEEP_CAPACITIES_KWH: tuple[float, ...] = (5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0)

CELLS_EUR_PER_KWH = 100.0
FIXED_OVERHEAD_EUR = 630.0  # BMS + balancer + rack + installatie

SALDERING_YEARS = 1
POST_YEARS = 14
HORIZON_YEARS = SALDERING_YEARS + POST_YEARS

USABLE_FRACTION = 0.9
MAX_POWER_KW = 5.0

DEFAULT_MARGINAL_PAYBACK_THRESHOLD_YEARS = 7.0


@dataclass(frozen=True)
class SweepRow:
    capacity_kwh: float
    capex_eur: float
    pre_annual_eur: float
    post_annual_eur: float
    blended_15yr_eur: float
    avg_annual_savings_eur: float
    payback_years: float
    eur_per_kwh_installed: float
    # Marginaal versus vorige rij (of versus (0 kWh, €0) voor de eerste rij;
    # "geen batterij" is het impliciete nulpunt).
    marginal_savings_eur_per_kwh: float
    marginal_payback_years: float
    # Veroudering / TCO. Horizon = HORIZON_YEARS (15 jaar).
    annual_efc: float
    peak_c_rate: float
    years_to_eol: float
    replacements_in_horizon: int
    replacement_cost_eur: float
    tco_15yr_eur: float
    tco_payback_years: float


@dataclass(frozen=True)
class SweepResult:
    rows: list[SweepRow]
    baseline_pre_eur: float
    baseline_post_eur: float
    baseline_blended_eur: float


def capex_for(capacity_kwh: float) -> float:
    return capacity_kwh * CELLS_EUR_PER_KWH + FIXED_OVERHEAD_EUR


def _payback_years(capex_eur: float, avg_annual_savings_eur: float) -> float:
    if avg_annual_savings_eur <= 0:
        return float("inf")
    return capex_eur / avg_annual_savings_eur


def _marginal_payback_years(marginal_savings_eur_per_kwh: float) -> float:
    """Jaren voor 1 extra kWh aan cellen (€100) om zichzelf terug te verdienen
    uit zijn marginale besparing. Vaste overhead is sunk."""
    if marginal_savings_eur_per_kwh <= 0:
        return float("inf")
    return CELLS_EUR_PER_KWH / marginal_savings_eur_per_kwh


def _resolve_sweep_tariffs(
    tariffs: dict[str, TariffParams],
) -> tuple[str, str, str, str, str]:
    """Kies baseline- en arbitrage-tarieven uit de registry.

    Voorkeur voor `frank-imbalance`. Valt terug op `tibber-day-ahead`.
    """
    fixed = next((k for k in ("baseline-fixed", "fixed") if k in tariffs), None)
    if fixed is None:
        raise RuntimeError("Geen vast-tarief contract in user-config; kan niet sweepen.")
    fixed_post = f"{fixed}-postsaldering"
    arb = next(
        (
            k
            for k in (
                "frank-imbalance",
                "frank",
                "tibber-day-ahead",
                "tibber",
            )
            if k in tariffs
        ),
        None,
    )
    if arb is None:
        raise RuntimeError("Geen dynamisch contract in user-config; kan niet sweepen.")
    arb_post = f"{arb}-postsaldering"
    arb_kind = "imbalance" if tariffs[arb].imbalance_trading else "day_ahead"
    return fixed, fixed_post, arb, arb_post, arb_kind


def _run_baselines(
    load: LoadSeries, prices: Prices, tariffs: dict[str, TariffParams]
) -> tuple[float, float]:
    """Capaciteits-onafhankelijke no-battery baselines (saldering + post)."""
    dummy_spec = BatterySpec(
        capacity_kwh=1.0,
        usable_fraction=USABLE_FRACTION,
        max_charge_kw=MAX_POWER_KW,
        max_discharge_kw=MAX_POWER_KW,
    )
    fixed, fixed_post, _, _, _ = _resolve_sweep_tariffs(tariffs)
    base_pre = run_scenario(
        fixed, "no_battery", load, prices, tariffs[fixed], dummy_spec
    ).annual_cost_eur
    base_post = run_scenario(
        fixed_post, "no_battery", load, prices, tariffs[fixed_post], dummy_spec
    ).annual_cost_eur
    return base_pre, base_post


def _run_capacity(
    capacity_kwh: float,
    load: LoadSeries,
    prices: Prices,
    tariffs: dict[str, TariffParams],
    precomputed_context: dict | None = None,
) -> tuple[ScenarioResult, ScenarioResult]:
    """Draai arbitrage-strategie pre/post saldering bij gegeven capaciteit."""
    spec = BatterySpec(
        capacity_kwh=capacity_kwh,
        usable_fraction=USABLE_FRACTION,
        max_charge_kw=MAX_POWER_KW,
        max_discharge_kw=MAX_POWER_KW,
    )
    _, _, arb, arb_post, arb_kind = _resolve_sweep_tariffs(tariffs)
    pre = run_scenario(
        arb, arb_kind, load, prices, tariffs[arb], spec,
        include_detail=False, precomputed_context=precomputed_context,
    )
    post = run_scenario(
        arb_post, arb_kind, load, prices, tariffs[arb_post], spec,
        include_detail=False, precomputed_context=precomputed_context,
    )
    return pre, post


def _run_capacity_packed(
    args: tuple[float, LoadSeries, Prices, dict[str, TariffParams], dict | None],
) -> tuple[float, ScenarioResult, ScenarioResult]:
    """Pickle-friendly worker wrapper for ProcessPoolExecutor."""
    capacity_kwh, load, prices, tariffs, ctx = args
    pre, post = _run_capacity(capacity_kwh, load, prices, tariffs, precomputed_context=ctx)
    return capacity_kwh, pre, post


def _cycle_profile(
    capacity_kwh: float, pre: ScenarioResult, post: ScenarioResult
) -> CycleProfile:
    """Horizon-gewogen cyclusprofiel uit een (pre, post)-saldering paar."""
    pre_throughput = pre.breakdown["battery_throughput_kwh"]
    post_throughput = post.breakdown["battery_throughput_kwh"]
    weighted_throughput = (
        SALDERING_YEARS * pre_throughput + POST_YEARS * post_throughput
    ) / HORIZON_YEARS
    peak = max(
        pre.breakdown["battery_peak_quarter_kwh"],
        post.breakdown["battery_peak_quarter_kwh"],
    )
    return CycleProfile(
        capacity_kwh=capacity_kwh,
        usable_fraction=USABLE_FRACTION,
        annual_throughput_kwh=weighted_throughput,
        peak_quarter_kwh=peak,
    )


def run_sweep(
    load: LoadSeries,
    prices: Prices,
    tariffs: dict[str, TariffParams],
    aging_model: AgingModel = AgingModel(),
    workers: int = 1,
) -> SweepResult:
    """Sweep frank-imbalance over SWEEP_CAPACITIES_KWH.

    No-battery baselines draaien één keer (capaciteits-onafhankelijk). Marginale
    kolommen worden berekend tegen de vorige rij; de eerste rij neemt
    (0 kWh, €0) als nulpunt.

    `workers > 1` parallelliseert via ProcessPoolExecutor.
    """
    base_pre, base_post = _run_baselines(load, prices, tariffs)
    base_blended = SALDERING_YEARS * base_pre + POST_YEARS * base_post

    # Dispatch-context hijsen uit de per-capaciteit-loop. Prijzen en ZP zijn
    # gelijk, dus rolling percentielen, dag-ranks en ZP-forecast hoeven maar
    # één keer berekend te worden.
    ctx = _build_dispatch_context(
        prices.day_ahead, load.pv_kwh, DispatchTuning(), override_price=prices.imbalance
    )

    if workers > 1 and len(SWEEP_CAPACITIES_KWH) > 1:
        tasks = [(cap, load, prices, tariffs, ctx) for cap in SWEEP_CAPACITIES_KWH]
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            packed = list(pool.map(_run_capacity_packed, tasks))
        capacity_results: dict[float, tuple[ScenarioResult, ScenarioResult]] = {
            cap: (pre, post) for cap, pre, post in packed
        }
    else:
        capacity_results = {
            cap: _run_capacity(cap, load, prices, tariffs, precomputed_context=ctx)
            for cap in SWEEP_CAPACITIES_KWH
        }

    rows: list[SweepRow] = []
    prev_capacity_kwh = 0.0
    prev_savings_eur = 0.0

    for cap in SWEEP_CAPACITIES_KWH:
        pre_result, post_result = capacity_results[cap]
        pre = pre_result.annual_cost_eur
        post = post_result.annual_cost_eur
        blended = SALDERING_YEARS * pre + POST_YEARS * post
        savings_total = base_blended - blended
        avg_annual_savings = savings_total / HORIZON_YEARS
        capex = capex_for(cap)

        d_cap = cap - prev_capacity_kwh
        d_sav = avg_annual_savings - prev_savings_eur
        marginal = d_sav / d_cap if d_cap > 0 else 0.0

        profile = _cycle_profile(cap, pre_result, post_result)
        eol = years_to_eol(profile, aging_model)
        schedule = replacement_schedule(eol, HORIZON_YEARS)
        repl = replacement_cost(profile, schedule, aging_model)
        tco_total = capex + repl

        rows.append(
            SweepRow(
                capacity_kwh=cap,
                capex_eur=capex,
                pre_annual_eur=pre,
                post_annual_eur=post,
                blended_15yr_eur=blended,
                avg_annual_savings_eur=avg_annual_savings,
                payback_years=_payback_years(capex, avg_annual_savings),
                eur_per_kwh_installed=capex / cap,
                marginal_savings_eur_per_kwh=marginal,
                marginal_payback_years=_marginal_payback_years(marginal),
                annual_efc=profile.annual_efc,
                peak_c_rate=profile.peak_c_rate,
                years_to_eol=eol,
                replacements_in_horizon=len(schedule),
                replacement_cost_eur=repl,
                tco_15yr_eur=tco_total,
                tco_payback_years=_payback_years(tco_total, avg_annual_savings),
            )
        )

        prev_capacity_kwh = cap
        prev_savings_eur = avg_annual_savings

    return SweepResult(
        rows=rows,
        baseline_pre_eur=base_pre,
        baseline_post_eur=base_post,
        baseline_blended_eur=base_blended,
    )


def lowest_tco_row(rows: list[SweepRow]) -> SweepRow:
    """Capaciteit met laagste 15-jaars TCO (capex + vervanging - besparing)."""
    if not rows:
        raise ValueError("geen rijen om uit te kiezen")
    return min(rows, key=lambda r: r.tco_15yr_eur - HORIZON_YEARS * r.avg_annual_savings_eur)


def roi_optimal_floor(
    rows: list[SweepRow],
    threshold_years: float = DEFAULT_MARGINAL_PAYBACK_THRESHOLD_YEARS,
) -> SweepRow:
    """Grootste capaciteit waarvan de marginale kWh binnen `threshold_years`
    terugverdient. Daarboven faalt de volgende kWh de drempel; daaronder
    laat je geld liggen.
    """
    if not rows:
        raise ValueError("geen rijen om uit te kiezen")
    eligible = [r for r in rows if r.marginal_payback_years <= threshold_years]
    if not eligible:
        return rows[0]
    return max(eligible, key=lambda r: r.capacity_kwh)
