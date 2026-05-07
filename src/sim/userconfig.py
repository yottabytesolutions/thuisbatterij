"""Gebruikersconfiguratie geladen uit `config/user.toml`.

Elke waarde hier hoort bij het werkelijke energiecontract, lokale belastingregime
of gekozen batterij van de gebruiker. Defaults zijn NL 2026-typisch zodat een
verse clone redelijke getallen geeft. Voor echt gebruik: overschrijf via
`config/user.toml`.
"""


import datetime as _dt
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from .battery import BatterySpec


@dataclass(frozen=True)
class GridConfig:
    energiebelasting_eur_kwh: float = 0.0996
    transport_eur_kwh: float = 0.025
    btw_rate: float = 0.21
    vermindering_energiebelasting_yearly_eur: float = 634.38


@dataclass(frozen=True)
class SalderingConfig:
    end_date: _dt.date = _dt.date(2027, 1, 1)
    post_saldering_export_premium_eur_kwh: float = 0.07


@dataclass(frozen=True)
class ContractConfig:
    """Eén leverancier of tarief. Wordt gestapeld op `GridConfig` om
    effectieve import- en exportprijzen op te bouwen."""

    display_name: str
    is_dynamic: bool = False
    commodity_eur_kwh: float = 0.0  # alleen relevant bij is_dynamic=False
    supplier_markup_eur_kwh: float = 0.0
    supplier_export_markup_eur_kwh: float = 0.0
    standing_yearly_eur: float = 0.0
    terugleverkosten_yearly_eur: float = 0.0
    service_fees_yearly_eur: float = 0.0
    imbalance_trading: bool = False
    imbalance_revenue_share_to_user: float = 1.0
    pass_through_negative_export: bool = False


@dataclass(frozen=True)
class SimulationConfig:
    start: str = "2025-05-01"
    end: str = "2026-05-01"
    questdb_url: str = "http://localhost:9000"
    entsoe_api_key: str | None = None
    entsoe_zone: str = "NL"


@dataclass(frozen=True)
class UserConfig:
    grid: GridConfig = field(default_factory=GridConfig)
    saldering: SalderingConfig = field(default_factory=SalderingConfig)
    contracts: dict[str, ContractConfig] = field(default_factory=dict)
    battery: BatterySpec = field(default_factory=BatterySpec)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


def _filter_known(cls, src: dict) -> dict:
    known = {field.name for field in fields(cls)}
    return {
        key: value
        for key, value in src.items()
        if key in known
    }


def _coerce_date(v) -> _dt.date:
    if isinstance(v, _dt.date):
        return v
    if isinstance(v, _dt.datetime):
        return v.date()
    return _dt.date.fromisoformat(str(v))


def load_user_config(path: Path | str | None = None) -> UserConfig:
    """Laad gebruikersconfig uit TOML.

    Bij `path=None`:
      1. omgevingsvariabele `$THUISBAT_CONFIG`
      2. `./config/user.toml`
      3. `./config/user.example.toml` (fallback)

    Ontbrekend bestand is geen fout. Defaults worden dan gebruikt.
    """
    import os

    if path is None:
        env = os.environ.get("THUISBAT_CONFIG")
        if env:
            path = Path(env)
        else:
            for candidate in (Path("config/user.toml"), Path("config/user.example.toml")):
                if candidate.exists():
                    path = candidate
                    break
    if path is None or not Path(path).exists():
        return UserConfig()

    raw = tomllib.loads(Path(path).read_text())

    grid = GridConfig(**_filter_known(GridConfig, raw.get("grid", {})))

    sal_raw = raw.get("saldering", {})
    if "end_date" in sal_raw:
        sal_raw = dict(sal_raw)
        sal_raw["end_date"] = _coerce_date(sal_raw["end_date"])
    saldering = SalderingConfig(**_filter_known(SalderingConfig, sal_raw))

    contracts: dict[str, ContractConfig] = {}
    for contract_key, contract_body in raw.get("contract", {}).items():
        contract_body = dict(contract_body)
        contract_body.setdefault("display_name", contract_key)
        contracts[contract_key] = ContractConfig(
            **_filter_known(ContractConfig, contract_body)
        )

    bat_raw = raw.get("battery", {})
    battery = BatterySpec(**_filter_known(BatterySpec, bat_raw)) if bat_raw else BatterySpec()

    sim = SimulationConfig(**_filter_known(SimulationConfig, raw.get("simulation", {})))

    return UserConfig(
        grid=grid,
        saldering=saldering,
        contracts=contracts,
        battery=battery,
        simulation=sim,
    )
