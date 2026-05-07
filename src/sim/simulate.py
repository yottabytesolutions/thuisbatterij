"""Voer één strategie uit over het jaar en boek de cashflow."""


import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .battery import BatterySpec
from .data import LoadSeries
from .economics import TariffParams, export_revenue_price, import_retail_price
from .prices import Prices
from .strategies import (
    DispatchContext,
    day_ahead_arbitrage,
    imbalance_aware,
    no_battery_dispatch,
    optimal_lp,
    perfect_foresight,
    pv_self_consume,
)


@dataclass(frozen=True)
class ScenarioResult:
    """Domeinmodel voor één scenario-uitkomst.

    `strategy_kind` en `tariff` zijn de identificerende dimensies waarop het
    rapport scenario's terugvindt. Eerder kwam dat uit de naam-string; die is
    nu louter cosmetisch.
    """

    name: str
    strategy_kind: str
    tariff: TariffParams
    annual_cost_eur: float
    breakdown: dict[str, float]
    detail: pd.DataFrame


def _as_series(value, index: pd.Index) -> pd.Series:
    """`export_revenue_price` geeft scalar of Series; lift altijd naar Series."""
    if isinstance(value, pd.Series):
        return value
    return pd.Series(value, index=index)


def _settle_price(tariff: TariffParams, prices: Prices) -> pd.Series:
    """Afrekenprijs per kwartier: day-ahead voor dynamisch, vaste commodity anders."""
    if tariff.is_dynamic:
        return prices.day_ahead
    return pd.Series(
        tariff.fixed_commodity_eur_kwh, index=prices.day_ahead.index, name="commodity"
    )


def _imbalance_revenue_share(
    dispatch: pd.Series, prices: Prices, tariff: TariffParams
) -> float:
    """Deel van Frank's onbalans-P&L op de batterij-dispatch.

    Frank dispatcht de batterij als BRP-afwijking van de DA-nominatie en
    settlet die afwijking (`dispatch` per kwartier) op onbalans. Met
    `share=0` valt de bonus weg, met `share=1` interpoleert de totale
    cashflow naar volledige onbalans-arbitrage.
    """
    delta = prices.imbalance - prices.day_ahead
    margin = float((-dispatch * delta).sum())
    return margin * tariff.imbalance_revenue_share_to_user


def run_scenario(
    name: str,
    strategy_kind: str,
    load: LoadSeries,
    prices: Prices,
    tariff: TariffParams,
    spec: BatterySpec,
    include_detail: bool = True,
    precomputed_context: DispatchContext | None = None,
) -> ScenarioResult:
    cons = load.consumption_kwh
    pv = load.pv_kwh

    allow_grid_export = tariff.saldering_active

    if strategy_kind == "no_battery":
        dispatch = no_battery_dispatch(cons, pv)
    elif strategy_kind == "pv_self":
        dispatch = pv_self_consume(cons, pv, spec)
    elif strategy_kind == "day_ahead":
        dispatch = day_ahead_arbitrage(
            cons, pv, prices.day_ahead, spec,
            allow_grid_export=allow_grid_export,
            precomputed_context=precomputed_context,
        )
    elif strategy_kind == "imbalance":
        dispatch = imbalance_aware(
            cons,
            pv,
            prices.day_ahead,
            prices.imbalance,
            spec,
            allow_grid_export=allow_grid_export,
            precomputed_context=precomputed_context,
        )
    elif strategy_kind == "perfect":
        # Perfect-foresight optimaliseert tegen de prijs die het contract
        # daadwerkelijk afrekent: day-ahead voor dynamisch. Onbalans is een
        # losse stroom in de cashflow hieronder.
        dispatch = perfect_foresight(
            cons, pv, prices.day_ahead, spec, allow_grid_export=allow_grid_export
        )
    elif strategy_kind == "lp":
        # Globale optimum: absolute ondergrens gegeven capaciteit en rendement.
        settle = _settle_price(tariff, prices)
        imp_price_arr = import_retail_price(settle, tariff)
        exp_price_arr = _as_series(export_revenue_price(settle, tariff), settle.index)
        dispatch = optimal_lp(cons, pv, imp_price_arr, exp_price_arr, spec)
    else:
        raise ValueError(f"unknown strategy {strategy_kind}")

    # Netto netstroom: dispatch > 0 = laden (meer import), < 0 = ontladen.
    net_grid = cons - pv + dispatch

    # ZP-curtailment bij pass-through post-saldering onder de drempel: snij
    # productie terug tot wat huis + batterij verbruiken. Strategie heeft ZP-
    # overschot al opgenomen in stap 1, dus geen verloren laadkans.
    if (
        not tariff.saldering_active
        and tariff.pass_through_negative_export
        and tariff.pv_curtail_threshold_eur_kwh > -math.inf
    ):
        will_curtail = (net_grid < 0) & (
            prices.day_ahead < tariff.pv_curtail_threshold_eur_kwh
        )
        curtailed_kwh = (-net_grid).where(will_curtail, 0.0).clip(lower=0.0)
        net_grid = net_grid.where(~will_curtail, 0.0)
    else:
        curtailed_kwh = pd.Series(0.0, index=net_grid.index)

    grid_import = net_grid.clip(lower=0.0).rename("grid_import_kwh")
    grid_export = (-net_grid).clip(lower=0.0).rename("grid_export_kwh")

    settle_price = _settle_price(tariff, prices)
    import_price = import_retail_price(settle_price, tariff)
    export_price = _as_series(export_revenue_price(settle_price, tariff), settle_price.index)

    cost_import = (grid_import * import_price).sum()
    revenue_export = (grid_export * export_price).sum()

    imbalance_extra = (
        _imbalance_revenue_share(dispatch, prices, tariff)
        if tariff.imbalance_trading
        else 0.0
    )

    standing = tariff.standing_yearly_eur
    vermindering = tariff.vermindering_energiebelasting_yearly_eur
    fees = tariff.service_fees_yearly_eur + tariff.terugleverkosten_yearly_eur

    total = (
        cost_import
        - revenue_export
        - imbalance_extra
        + standing
        - vermindering
        + fees
    )

    if include_detail:
        detail = pd.DataFrame(
            {
                "consumption_kwh": cons,
                "pv_kwh": pv,
                "battery_ac_kwh": dispatch,
                "grid_import_kwh": grid_import,
                "grid_export_kwh": grid_export,
                "import_price_eur_kwh": import_price,
                "export_price_eur_kwh": export_price,
                "day_ahead_eur_kwh": prices.day_ahead,
                "imbalance_eur_kwh": prices.imbalance,
            }
        )
    else:
        detail = pd.DataFrame()

    throughput = float(np.abs(dispatch).sum())
    peak_quarter = float(np.abs(dispatch).max()) if len(dispatch) else 0.0
    usable_kwh = spec.usable_kwh
    annual_efc = throughput / (2.0 * usable_kwh) if usable_kwh > 0 else 0.0

    breakdown = {
        "import_cost": float(cost_import),
        "export_revenue": float(revenue_export),
        "imbalance_extra": float(imbalance_extra),
        "standing_charges": float(standing),
        "vermindering_energiebelasting": float(vermindering),
        "service_fees_and_penalties": float(fees),
        "grid_import_kwh_total": float(grid_import.sum()),
        "grid_export_kwh_total": float(grid_export.sum()),
        "pv_curtailed_kwh_total": float(curtailed_kwh.sum()),
        "battery_throughput_kwh": throughput,
        "battery_peak_quarter_kwh": peak_quarter,
        "annual_efc": annual_efc,
    }

    return ScenarioResult(
        name=name,
        strategy_kind=strategy_kind,
        tariff=tariff,
        annual_cost_eur=float(total),
        breakdown=breakdown,
        detail=detail,
    )
