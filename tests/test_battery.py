"""Tests voor de batterij-state-machine."""

from sim.battery import BatterySpec, BatteryState


def test_charge_respects_capacity() -> None:
    spec = BatterySpec(capacity_kwh=10, usable_fraction=1.0, max_charge_kw=100)
    state = BatteryState(soc_kwh=0)
    ac, dc = state.charge(20.0, spec)
    # AC wordt begrensd door max_charge_kwh_per_quarter; 20 past dus.
    # DC wordt begrensd op 10 door de capaciteit.
    assert state.soc_kwh == 10
    assert dc == 10
    assert ac < 20  # alleen genoeg AC om 10 kWh ruimte te vullen


def test_discharge_respects_soc() -> None:
    spec = BatterySpec(capacity_kwh=10, usable_fraction=1.0, max_discharge_kw=100)
    state = BatteryState(soc_kwh=5)
    ac, dc = state.discharge(20.0, spec)
    assert state.soc_kwh == 0
    assert dc == 5
    assert ac < 5  # geleverde AC is dc * rendement


def test_round_trip() -> None:
    spec = BatterySpec(
        capacity_kwh=100, usable_fraction=1.0, max_charge_kw=400, max_discharge_kw=400
    )
    state = BatteryState(soc_kwh=0)
    ac_in, _ = state.charge(10.0, spec)
    ac_out, _ = state.discharge(100.0, spec)
    # Round-trip: ac_out / ac_in moet dicht bij round_trip_efficiency liggen.
    assert abs(ac_out / ac_in - 0.88) < 0.01
