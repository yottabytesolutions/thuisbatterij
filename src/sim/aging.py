"""Veroudering en vervangingskostenmodel voor de batterij.

Cycluslevensduur hangt af van DoD en gemiddelde C-rate. We gebruiken een
power-law correctie op een referentie van 80% DoD, 0.5C, 25°C voor moderne
LFP-cellen. De kalenderondergrens reflecteert plank-aging bij meterkast-
temperatuur (15-20°C); boven 25°C zakt dit snel.

Bewust simpel: 80% SoH is een garantie-/gezondheidsdrempel, geen sloopgrens.
Vervanging gebeurt pas bij een praktische retirementsdrempel; tussen die twee
draait het pack door met minder bruikbare capaciteit. Throughput en piek-C-rate
komen uit de dispatch-serie van de simulator.
"""


import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AgingModel:
    reference_cycles_to_80pct: int = 6000
    calendar_life_years: float = 14.0
    warranty_capacity_fraction: float = 0.80
    retirement_capacity_fraction: float = 0.70
    dod_exponent: float = 1.7
    c_rate_exponent: float = 0.5
    cell_replacement_cost_eur_per_kwh: float = 100.0
    bms_replacement_cost_eur: float = 350.0
    bms_replacement_interval: int = 2


@dataclass(frozen=True)
class CycleProfile:
    capacity_kwh: float
    usable_fraction: float
    annual_throughput_kwh: float
    peak_quarter_kwh: float

    @property
    def usable_kwh(self) -> float:
        return self.capacity_kwh * self.usable_fraction

    @property
    def annual_efc(self) -> float:
        if self.usable_kwh <= 0:
            return 0.0
        return self.annual_throughput_kwh / (2.0 * self.usable_kwh)

    @property
    def peak_c_rate(self) -> float:
        if self.capacity_kwh <= 0:
            return 0.0
        return self.peak_quarter_kwh * 4.0 / self.capacity_kwh


def adjusted_cycle_life(profile: CycleProfile, model: AgingModel = AgingModel()) -> float:
    """Cycli tot 80% capaciteit bij de DoD en piek-C-rate uit het profiel.

    Power-law multipliers op een 6000-cycli referentie bij 80% DoD, 0.5C.
    """
    dod_pct = profile.usable_fraction * 100.0
    dod_factor = (80.0 / dod_pct) ** model.dod_exponent if dod_pct > 0 else 0.0
    c_rate = max(profile.peak_c_rate, 1e-6)
    c_factor = (0.5 / c_rate) ** model.c_rate_exponent
    return model.reference_cycles_to_80pct * dod_factor * c_factor


def years_to_warranty_threshold(profile: CycleProfile, model: AgingModel = AgingModel()) -> float:
    """Jaren tot de cellen op de warranty/health-drempel zitten.

    Het minimum van cyclus-gedreven levensduur (cycli / jaarlijkse EFC) en
    kalenderlevensduur. Dit is meestal 80% restcapaciteit: relevant voor
    garantie en performance, maar geen automatische vervangingsdatum.
    """
    cycles = adjusted_cycle_life(profile, model)
    annual = profile.annual_efc
    cycle_life = cycles / annual if annual > 0 else float("inf")
    return min(cycle_life, model.calendar_life_years)


def retirement_multiplier(model: AgingModel = AgingModel()) -> float:
    """Schaalfactor van 80%-drempel naar praktische retirementsdrempel.

    Zonder cel-specifieke fadecurve nemen we na de warranty-drempel lineaire
    capaciteitsfade aan. 80% → 70% betekent dan grofweg 1,5× de tijd/cycli tot
    80%. De factor is begrensd zodat onzinnige configuraties niet exploderen.
    """
    warranty_loss = 1.0 - model.warranty_capacity_fraction
    retirement_loss = 1.0 - model.retirement_capacity_fraction
    if warranty_loss <= 0 or retirement_loss <= warranty_loss:
        return 1.0
    return round(retirement_loss / warranty_loss, 6)


def years_to_eol(profile: CycleProfile, model: AgingModel = AgingModel()) -> float:
    """Jaren tot praktische cel-retirement, niet tot 80% SoH.

    Een thuisbatterij onder 80% van oorspronkelijke capaciteit is doorgaans nog
    bruikbaar; de gebruiker merkt vooral minder kWh venster. Daarom plant TCO
    pas een cel-swap bij `retirement_capacity_fraction`.
    """
    return years_to_warranty_threshold(profile, model) * retirement_multiplier(model)


def end_of_horizon_capacity_fraction(
    profile: CycleProfile,
    horizon_years: int = 15,
    model: AgingModel = AgingModel(),
) -> float:
    """Geschatte restcapaciteit zonder tussentijdse cel-swap."""
    warranty_years = years_to_warranty_threshold(profile, model)
    if warranty_years <= 0 or math.isinf(warranty_years):
        return 1.0
    annual_fade = (1.0 - model.warranty_capacity_fraction) / warranty_years
    return max(model.retirement_capacity_fraction, 1.0 - annual_fade * horizon_years)


def replacement_schedule(years_to_eol: float, horizon_years: int = 15) -> list[int]:
    """Jaren (1-geïndexeerd) binnen de horizon waarop een cel-swap valt.

    Een vervanging op jaar N reset de klok; de volgende swap landt op
    ⌈2 × years_to_eol⌉, dan ⌈3 × years_to_eol⌉, enz., zolang het jaar strikt
    minder is dan de horizon. Een falen op de horizon-rand triggert geen swap.
    """
    if years_to_eol <= 0:
        return []
    schedule: list[int] = []
    n = 1
    while True:
        year = math.ceil(years_to_eol * n)
        if year >= horizon_years:
            break
        schedule.append(year)
        n += 1
    return schedule


def replacement_cost(
    profile: CycleProfile,
    schedule: list[int],
    model: AgingModel = AgingModel(),
) -> float:
    """Nominale kosten van alle swaps in het schema.

    Elke swap betaalt voor cellen (€/kWh × capaciteit). Elke N-de swap
    betaalt ook voor een BMS-vervanging.
    """
    if not schedule:
        return 0.0
    cell_cost = profile.capacity_kwh * model.cell_replacement_cost_eur_per_kwh
    bms_count = len(schedule) // model.bms_replacement_interval
    return len(schedule) * cell_cost + bms_count * model.bms_replacement_cost_eur


def tco(
    capex_eur: float,
    profile: CycleProfile,
    horizon_years: int = 15,
    model: AgingModel = AgingModel(),
) -> tuple[float, list[int], float]:
    """Capex + vervangingskosten over de horizon.

    Geeft (tco_eur, replacement_schedule, replacement_cost_eur) terug.
    """
    eol = years_to_eol(profile, model)
    schedule = replacement_schedule(eol, horizon_years)
    repl = replacement_cost(profile, schedule, model)
    return capex_eur + repl, schedule, repl
