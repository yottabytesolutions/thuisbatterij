"""Tests voor het verouderings- en vervangingskostenmodel."""

import math

import pytest

from sim.aging import (
    AgingModel,
    CycleProfile,
    adjusted_cycle_life,
    end_of_horizon_capacity_fraction,
    replacement_cost,
    replacement_schedule,
    retirement_multiplier,
    tco,
    years_to_warranty_threshold,
    years_to_eol,
)


def _profile(
    capacity_kwh: float = 20.0,
    usable_fraction: float = 0.9,
    annual_throughput_kwh: float = 4000.0,
    peak_quarter_kwh: float = 1.25,
) -> CycleProfile:
    return CycleProfile(
        capacity_kwh=capacity_kwh,
        usable_fraction=usable_fraction,
        annual_throughput_kwh=annual_throughput_kwh,
        peak_quarter_kwh=peak_quarter_kwh,
    )


def test_annual_efc_uses_usable_kwh() -> None:
    p = _profile(capacity_kwh=10, usable_fraction=0.9, annual_throughput_kwh=900)
    # bruikbaar = 9 kWh, doorzet / (2*bruikbaar) = 900 / 18 = 50 EFC
    assert p.annual_efc == pytest.approx(50.0)


def test_peak_c_rate_kwh_per_quarter() -> None:
    p = _profile(capacity_kwh=10, peak_quarter_kwh=1.25)
    # 1.25 kWh per kwartier = 5 kW piek; 5 kW / 10 kWh = 0.5 C
    assert p.peak_c_rate == pytest.approx(0.5)


def test_dod_curve_penalises_higher_dod() -> None:
    model = AgingModel()
    shallow = _profile(usable_fraction=0.5, peak_quarter_kwh=0.625)  # 0.5C
    deep = _profile(usable_fraction=0.9, peak_quarter_kwh=0.625)
    assert adjusted_cycle_life(shallow, model) > adjusted_cycle_life(deep, model)


def test_c_rate_curve_penalises_high_c_rate() -> None:
    model = AgingModel()
    slow = _profile(capacity_kwh=20, peak_quarter_kwh=0.5)   # 0.1C
    fast = _profile(capacity_kwh=20, peak_quarter_kwh=5.0)   # 1.0C
    assert adjusted_cycle_life(slow, model) > adjusted_cycle_life(fast, model)


def test_reference_point_matches_constant() -> None:
    """Bij 80% DoD en 0.5C moet het model het referentieaantal cycli geven."""
    model = AgingModel()
    p = _profile(capacity_kwh=10, usable_fraction=0.8, peak_quarter_kwh=1.25)
    # peak_quarter 1.25 kWh × 4 / 10 kWh = 0.5C
    assert adjusted_cycle_life(p, model) == pytest.approx(model.reference_cycles_to_80pct)


def test_calendar_floor_caps_warranty_threshold() -> None:
    model = AgingModel(calendar_life_years=10.0)
    p = _profile(annual_throughput_kwh=100.0)  # ~3 EFC/jr, cyclusleven ruim langer
    assert years_to_warranty_threshold(p, model) == pytest.approx(10.0)


def test_eol_extends_beyond_80pct_health_threshold() -> None:
    model = AgingModel(calendar_life_years=10.0)
    p = _profile(annual_throughput_kwh=100.0)
    assert retirement_multiplier(model) == pytest.approx(1.5)
    assert years_to_eol(p, model) == pytest.approx(15.0)


def test_cycle_floor_dominates_when_thrashed() -> None:
    model = AgingModel(calendar_life_years=14.0)
    # 5 kWh batterij, 10000 kWh doorzet, 1.25 kWh/kwartier: zwaar cyclen.
    p = _profile(
        capacity_kwh=5.0,
        usable_fraction=0.9,
        annual_throughput_kwh=10000.0,
        peak_quarter_kwh=1.25,
    )
    assert years_to_warranty_threshold(p, model) < model.calendar_life_years


def test_end_of_horizon_capacity_fraction_derates_instead_of_scrapping() -> None:
    model = AgingModel(calendar_life_years=10.0)
    p = _profile(annual_throughput_kwh=100.0)
    assert end_of_horizon_capacity_fraction(p, horizon_years=10, model=model) == pytest.approx(0.8)
    assert end_of_horizon_capacity_fraction(p, horizon_years=15, model=model) == pytest.approx(0.7)


def test_replacement_schedule_short_life() -> None:
    # Levensduur 4 jaar, horizon 15 jaar: swaps in jaar 4, 8, 12.
    assert replacement_schedule(4.0, horizon_years=15) == [4, 8, 12]


def test_replacement_schedule_survives_horizon() -> None:
    # Levensduur langer dan horizon: geen swaps.
    assert replacement_schedule(20.0, horizon_years=15) == []


def test_replacement_schedule_rounds_up() -> None:
    # 3.5 jaar: swaps op 4, 7, 11, 14.
    assert replacement_schedule(3.5, horizon_years=15) == [4, 7, 11, 14]


def test_replacement_schedule_zero_or_negative_returns_empty() -> None:
    assert replacement_schedule(0.0) == []
    assert replacement_schedule(-1.0) == []


def test_replacement_cost_includes_cells_and_periodic_bms() -> None:
    model = AgingModel(
        cell_replacement_cost_eur_per_kwh=100.0,
        bms_replacement_cost_eur=350.0,
        bms_replacement_interval=2,
    )
    p = _profile(capacity_kwh=10.0)
    # 3 swaps: 3 × 1000 cellen + 1 BMS, elke 2e swap, is 3350.
    cost = replacement_cost(p, [4, 8, 12], model)
    assert cost == pytest.approx(3 * 1000.0 + 1 * 350.0)


def test_replacement_cost_no_swaps_zero() -> None:
    assert replacement_cost(_profile(), [], AgingModel()) == 0.0


def test_tco_combines_capex_and_replacements() -> None:
    model = AgingModel(calendar_life_years=4.0)  # forceer drie swaps in 15 jr
    p = _profile(capacity_kwh=10.0)
    total, schedule, repl = tco(2000.0, p, horizon_years=15, model=model)
    assert schedule == [6, 12]
    assert repl == pytest.approx(2 * 1000.0 + 350.0)
    assert total == pytest.approx(2000.0 + repl)


def test_tco_no_replacements_returns_capex() -> None:
    model = AgingModel(calendar_life_years=20.0)
    p = _profile(annual_throughput_kwh=100.0)  # cyclusleven is lang
    total, schedule, repl = tco(5000.0, p, horizon_years=15, model=model)
    assert schedule == []
    assert repl == 0.0
    assert total == pytest.approx(5000.0)


def test_zero_capacity_is_handled() -> None:
    p = _profile(capacity_kwh=0.0)
    # Voorkom deling door nul. EFC = 0, cyclusvloer = inf, kalender wint.
    assert math.isfinite(years_to_eol(p))
