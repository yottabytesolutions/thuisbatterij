"""Batterij-toestand."""


from dataclasses import dataclass


@dataclass
class BatterySpec:
    """Fysieke en operationele parameters van de batterij."""

    capacity_kwh: float = 28.7
    usable_fraction: float = 0.9
    max_charge_kw: float = 5.0
    max_discharge_kw: float = 5.0
    round_trip_efficiency: float = 0.88

    @property
    def usable_kwh(self) -> float:
        return self.capacity_kwh * self.usable_fraction

    @property
    def one_way_efficiency(self) -> float:
        # Symmetrisch: laadrendement * ontlaadrendement = round_trip.
        return self.round_trip_efficiency**0.5

    def max_charge_kwh_per_quarter(self) -> float:
        return self.max_charge_kw * 0.25

    def max_discharge_kwh_per_quarter(self) -> float:
        return self.max_discharge_kw * 0.25


@dataclass
class BatteryState:
    """Toestand van de batterij tijdens een simulatie."""

    soc_kwh: float

    def charge(self, ac_kwh: float, spec: BatterySpec) -> tuple[float, float]:
        """Laad de batterij vanaf AC-zijde. Geeft (werkelijk_ac_kwh, opgeslagen_dc) terug."""
        if ac_kwh <= 0:
            return 0.0, 0.0
        ac_max = spec.max_charge_kwh_per_quarter()
        ac_kwh = min(ac_kwh, ac_max)
        eff = spec.one_way_efficiency
        dc = ac_kwh * eff
        room = spec.usable_kwh - self.soc_kwh
        if dc > room:
            dc = room
            ac_kwh = dc / eff
        self.soc_kwh += dc
        return ac_kwh, dc

    def discharge(self, ac_kwh_target: float, spec: BatterySpec) -> tuple[float, float]:
        """Ontlaad richting AC-zijde. Geeft (werkelijk_ac_kwh, opgenomen_dc) terug."""
        if ac_kwh_target <= 0:
            return 0.0, 0.0
        ac_max = spec.max_discharge_kwh_per_quarter()
        ac_kwh_target = min(ac_kwh_target, ac_max)
        eff = spec.one_way_efficiency
        dc_needed = ac_kwh_target / eff
        if dc_needed > self.soc_kwh:
            dc_needed = self.soc_kwh
            ac_kwh_target = dc_needed * eff
        self.soc_kwh -= dc_needed
        return ac_kwh_target, dc_needed
