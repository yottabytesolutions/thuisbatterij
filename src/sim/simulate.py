"""Voer één strategie uit over het jaar en boek de cashflow."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .battery import BatterySpec
from .data import LoadSeries
from .economics import TariffParams, export_revenue_price, import_retail_price
from .prices import Prices
from .strategies import (
    day_ahead_arbitrage,
    imbalance_aware,
    no_battery_dispatch,
    optimal_lp,
    perfect_foresight,
    pv_self_consume,
)


@dataclass
class ScenarioResult:
    name: str
    annual_cost_eur: float
    breakdown: dict[str, float]
    detail: pd.DataFrame  # detail per kwartier, handig voor grafieken


def run_scenario(
    name: str,
    strategy_kind: str,
    load: LoadSeries,
    prices: Prices,
    tariff: TariffParams,
    spec: BatterySpec,
    include_detail: bool = True,
    precomputed_context: dict | None = None,
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
        # Globale optimum via lineair programmeren: absolute ondergrens van
        # de rekening gegeven capaciteit, vermogen en round-trip-rendement.
        if tariff.is_dynamic:
            settle = prices.day_ahead
        else:
            settle = pd.Series(
                tariff.fixed_commodity_eur_kwh, index=prices.day_ahead.index
            )
        imp_price_arr = import_retail_price(settle, tariff)
        exp_price_arr = export_revenue_price(settle, tariff)
        if not isinstance(exp_price_arr, pd.Series):
            exp_price_arr = pd.Series(exp_price_arr, index=settle.index)
        dispatch = optimal_lp(cons, pv, imp_price_arr, exp_price_arr, spec)
    else:
        raise ValueError(f"unknown strategy {strategy_kind}")

    # Netto netstroom per kwartier (kWh):
    #   net_grid = verbruik - pv + dispatch
    #   dispatch > 0: laden vanaf AC, meer import / minder export
    #   dispatch < 0: ontladen naar AC, minder import / meer export
    net_grid = cons - pv + dispatch

    # ZP-curtailment: bij (a) saldering uit, (b) pass-through-contract,
    # (c) export-richting en (d) commodity onder de drempel, snijden we de
    # productie terug tot wat huis + batterij verbruiken. net_grid wordt op
    # nul geklemd in plaats van negatief te gaan.
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

    # Afrekenmarkt: dynamisch op day-ahead, vast op fixed commodity.
    if tariff.is_dynamic:
        settle_price = prices.day_ahead
    else:
        settle_price = pd.Series(
            tariff.fixed_commodity_eur_kwh, index=prices.day_ahead.index, name="commodity"
        )

    import_price = import_retail_price(settle_price, tariff)
    export_price = export_revenue_price(settle_price, tariff)
    if not isinstance(export_price, pd.Series):
        export_price = pd.Series(export_price, index=settle_price.index)

    cost_import = (grid_import * import_price).sum()
    revenue_export = (grid_export * export_price).sum()

    # Onbalans-P&L op de delta (im - da) elk kwartier waarin de batterij actief was.
    # Tekens: ontladen wint (im - da) per kWh, laden wint (da - im).
    # Revenue-share is symmetrisch: slechte trades kosten ook geld.
    imbalance_extra = 0.0
    if tariff.imbalance_trading:
        delta = prices.imbalance - prices.day_ahead
        net_im = (
            (dispatch < 0).astype("float64") * (-dispatch) * delta
            + (dispatch > 0).astype("float64") * dispatch * (-delta)
        ).sum()
        imbalance_extra = float(net_im) * tariff.imbalance_revenue_share_to_user

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
        name=name, annual_cost_eur=float(total), breakdown=breakdown, detail=detail
    )
