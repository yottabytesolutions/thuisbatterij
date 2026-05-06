"""Tarief- en belastingberekening, los van de dispatch-beslissing.

Retail-elektriciteit (consument) per kWh:
  import_prijs = commodity
                 + leverancier_opslag
                 + energiebelasting
                 + ode
                 + transport
                 + btw
  export_prijs (saldering aan) = import_prijs - leverancier_export_opslag
  export_prijs (saldering uit, vaste vloer) = max(0, post_sal_premie) - leverancier_export_opslag
  export_prijs (saldering uit, pass-through) = commodity - leverancier_export_opslag

Veel termen zijn vlak per kWh en hangen niet af van wanneer je verbruikt.
De arbitragewaarde van een batterij zit dus puur in het commodity-deel.
Voor totaalbedragen moet je de vlakke termen wel meerekenen.

Deze module is generiek. Alle numerieke kalibratiewaarden komen uit de
geladen `UserConfig` (`config/user.toml`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .userconfig import ContractConfig, GridConfig, SalderingConfig


@dataclass(frozen=True)
class TariffParams:
    """Runtime-parameters voor één leverancier/contract, volledig gemergd.

    Bouw via `tariff_from_config(...)` vanuit een `UserConfig`. Niet direct
    construeren tenzij je test. Het veld-aantal is groot.
    """

    name: str
    supplier_markup_eur_kwh: float
    supplier_export_markup_eur_kwh: float
    energiebelasting_eur_kwh: float
    ode_eur_kwh: float
    transport_eur_kwh: float
    btw_rate: float
    fixed_commodity_eur_kwh: float
    standing_yearly_eur: float
    vermindering_energiebelasting_yearly_eur: float
    is_dynamic: bool
    saldering_active: bool
    post_saldering_export_premium_eur_kwh: float
    terugleverkosten_yearly_eur: float
    imbalance_trading: bool
    imbalance_revenue_share_to_user: float
    service_fees_yearly_eur: float
    pass_through_negative_export: bool
    # ZP-curtailmentdrempel (€/kWh, ex BTW commodity). Default -inf = nooit afregelen.
    pv_curtail_threshold_eur_kwh: float = -math.inf


def tariff_from_config(
    contract: ContractConfig,
    grid: GridConfig,
    saldering: SalderingConfig,
    *,
    saldering_active: bool,
    pv_curtail_threshold_eur_kwh: float = -math.inf,
    name_override: str | None = None,
) -> TariffParams:
    """Voeg contract, jurisdictie en saldering-status samen tot één TariffParams."""
    suffix = "" if saldering_active else "-postsaldering"
    return TariffParams(
        name=(name_override or contract.display_name) + suffix,
        supplier_markup_eur_kwh=contract.supplier_markup_eur_kwh,
        supplier_export_markup_eur_kwh=contract.supplier_export_markup_eur_kwh,
        energiebelasting_eur_kwh=grid.energiebelasting_eur_kwh,
        ode_eur_kwh=0.0,
        transport_eur_kwh=grid.transport_eur_kwh,
        btw_rate=grid.btw_rate,
        fixed_commodity_eur_kwh=contract.commodity_eur_kwh,
        standing_yearly_eur=contract.standing_yearly_eur,
        vermindering_energiebelasting_yearly_eur=grid.vermindering_energiebelasting_yearly_eur,
        is_dynamic=contract.is_dynamic,
        saldering_active=saldering_active,
        post_saldering_export_premium_eur_kwh=saldering.post_saldering_export_premium_eur_kwh,
        terugleverkosten_yearly_eur=contract.terugleverkosten_yearly_eur,
        imbalance_trading=contract.imbalance_trading,
        imbalance_revenue_share_to_user=contract.imbalance_revenue_share_to_user,
        service_fees_yearly_eur=contract.service_fees_yearly_eur,
        pass_through_negative_export=contract.pass_through_negative_export,
        pv_curtail_threshold_eur_kwh=pv_curtail_threshold_eur_kwh,
    )


def import_retail_price(commodity_eur_kwh, t: TariffParams):
    """€/kWh om nu 1 kWh van het net te importeren.

    Accepteert scalar float of pd.Series/np.ndarray. Pandas en NumPy
    broadcasten transparant: één C-level call in plaats van 35.000 Python-calls.
    """
    base = (
        commodity_eur_kwh
        + t.supplier_markup_eur_kwh
        + t.energiebelasting_eur_kwh
        + t.ode_eur_kwh
        + t.transport_eur_kwh
    )
    return base * (1.0 + t.btw_rate)


def export_revenue_price(commodity_eur_kwh, t: TariffParams):
    """€/kWh ontvangen bij teruglevering.

    Drie regimes:
      - saldering aan: export wordt verrekend op retail (tot end_date).
      - saldering uit, vaste vloer: export levert de terugleverpremie op.
      - saldering uit, pass-through: export volgt de live commodity-prijs,
        inclusief negatieve uren.

    In alle drie wordt de export-opslag van de leverancier afgetrokken.

    Vectorbaar: werkt op scalar, Series of ndarray.
    """
    if t.saldering_active:
        return import_retail_price(commodity_eur_kwh, t) - t.supplier_export_markup_eur_kwh
    if t.pass_through_negative_export:
        return commodity_eur_kwh - t.supplier_export_markup_eur_kwh
    premium = max(0.0, t.post_saldering_export_premium_eur_kwh) - t.supplier_export_markup_eur_kwh
    if hasattr(commodity_eur_kwh, "__len__"):
        import numpy as np
        return np.full(len(commodity_eur_kwh), premium)
    return premium


def build_predefined(
    contracts: dict[str, ContractConfig],
    grid: GridConfig,
    saldering: SalderingConfig,
    *,
    curtail_pairs: dict[str, float] | None = None,
) -> dict[str, TariffParams]:
    """Bouw de runtime-tariefregistry uit een geladen UserConfig.

    Voor elk contract `c` in `contracts` worden twee entries gegenereerd:
      - `<c.display_name>` (saldering aan)
      - `<c.display_name>-postsaldering` (saldering uit)

    `curtail_pairs` voegt een variant toe met ZP-curtailmentdrempel.
    Bijvoorbeeld `{"tibber": -0.02}` levert `tibber-curtail-postsaldering`.
    Curtailment werkt alleen post-saldering met `pass_through_negative_export=True`.
    """
    out: dict[str, TariffParams] = {}
    for key, c in contracts.items():
        on = tariff_from_config(c, grid, saldering, saldering_active=True)
        off = tariff_from_config(c, grid, saldering, saldering_active=False)
        out[on.name] = on
        out[off.name] = off
    for key, threshold in (curtail_pairs or {}).items():
        if key not in contracts:
            continue
        c = contracts[key]
        out[f"{c.display_name}-curtail-postsaldering"] = tariff_from_config(
            c,
            grid,
            saldering,
            saldering_active=False,
            pv_curtail_threshold_eur_kwh=threshold,
            name_override=f"{c.display_name}-curtail",
        )
    return out
