"""Batterij-spec en pure laad/ontlaad-stappen.

`charge_step` en `discharge_step` zijn de canonieke math: scalar-in, scalar-uit,
JIT-gecompileerd. `BatteryState` is een mutabele wrapper voor ad-hoc gebruik en
tests; productie gebruikt de pure functies direct.
"""


from dataclasses import dataclass

from numba import njit


@dataclass(frozen=True)
class BatterySpec:
    """Fysieke en operationele parameters van de batterij."""

    capacity_kwh: float = 28.7
    usable_fraction: float = 0.9
    max_charge_kw: float = 5.0
    max_discharge_kw: float = 5.0
    round_trip_efficiency: float = 0.88

    def __post_init__(self) -> None:
        if self.capacity_kwh <= 0:
            raise ValueError(f"capacity_kwh must be > 0, got {self.capacity_kwh}")
        if not 0 < self.usable_fraction <= 1:
            raise ValueError(
                f"usable_fraction must be in (0, 1], got {self.usable_fraction}"
            )
        if self.max_charge_kw <= 0:
            raise ValueError(f"max_charge_kw must be > 0, got {self.max_charge_kw}")
        if self.max_discharge_kw <= 0:
            raise ValueError(
                f"max_discharge_kw must be > 0, got {self.max_discharge_kw}"
            )
        if not 0 < self.round_trip_efficiency <= 1:
            raise ValueError(
                f"round_trip_efficiency must be in (0, 1], got {self.round_trip_efficiency}"
            )

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


@njit(cache=True)
def charge_step(
    soc: float, ac_kwh: float, max_q: float, usable: float, one_way: float
) -> tuple[float, float]:
    """Pure laadstap. Geeft (nieuwe_soc, werkelijk_ac_kwh) terug."""
    if ac_kwh <= 0.0:
        return soc, 0.0
    ac = ac_kwh if ac_kwh < max_q else max_q
    dc = ac * one_way
    room = usable - soc
    if dc > room:
        dc = room
        ac = dc / one_way if one_way > 0.0 else 0.0
    return soc + dc, ac


@njit(cache=True)
def discharge_step(
    soc: float, ac_target: float, max_q: float, one_way: float
) -> tuple[float, float]:
    """Pure ontlaadstap. Geeft (nieuwe_soc, werkelijk_ac_kwh) terug."""
    if ac_target <= 0.0:
        return soc, 0.0
    ac = ac_target if ac_target < max_q else max_q
    dc_needed = ac / one_way if one_way > 0.0 else 0.0
    if dc_needed > soc:
        dc_needed = soc
        ac = dc_needed * one_way
    return soc - dc_needed, ac


@dataclass
class BatteryState:
    """Mutabele wrapper over `charge_step` / `discharge_step` voor ad-hoc gebruik.

    Productie gebruikt de pure stap-functies direct (sneller, picklebaar in
    workers). Deze klasse bestaat voor scripting en de tests.
    """

    soc_kwh: float

    def charge(self, ac_kwh: float, spec: BatterySpec) -> tuple[float, float]:
        new_soc, ac = charge_step(
            self.soc_kwh,
            ac_kwh,
            spec.max_charge_kwh_per_quarter(),
            spec.usable_kwh,
            spec.one_way_efficiency,
        )
        dc = new_soc - self.soc_kwh
        self.soc_kwh = new_soc
        return ac, dc

    def discharge(self, ac_target: float, spec: BatterySpec) -> tuple[float, float]:
        new_soc, ac = discharge_step(
            self.soc_kwh,
            ac_target,
            spec.max_discharge_kwh_per_quarter(),
            spec.one_way_efficiency,
        )
        dc = self.soc_kwh - new_soc
        self.soc_kwh = new_soc
        return ac, dc
