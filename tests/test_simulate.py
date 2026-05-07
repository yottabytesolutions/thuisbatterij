"""Tests voor `simulate.run_scenario` en de onbalans-bonus-formule."""


import pandas as pd
import pytest

from sim.battery import BatterySpec
from sim.data import LoadSeries
from sim.economics import TariffParams
from sim.prices import Prices
from sim.simulate import _imbalance_revenue_share, run_scenario


def _quarter_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-06-01", periods=n, freq="15min", tz="UTC")


def _frank_like_tariff(*, imbalance_revenue_share_to_user: float) -> TariffParams:
    """Minimale Frank-achtige TariffParams voor unit-tests.

    Geen toeslagen of vaste kosten zodat de math van de onbalans-bonus
    geïsoleerd kan worden.
    """
    return TariffParams(
        name="frank-test",
        supplier_markup_eur_kwh=0.0,
        supplier_export_markup_eur_kwh=0.0,
        energiebelasting_eur_kwh=0.0,
        ode_eur_kwh=0.0,
        transport_eur_kwh=0.0,
        btw_rate=0.0,
        fixed_commodity_eur_kwh=0.0,
        standing_yearly_eur=0.0,
        vermindering_energiebelasting_yearly_eur=0.0,
        is_dynamic=True,
        saldering_active=False,
        post_saldering_export_premium_eur_kwh=0.0,
        terugleverkosten_yearly_eur=0.0,
        imbalance_trading=True,
        imbalance_revenue_share_to_user=imbalance_revenue_share_to_user,
        service_fees_yearly_eur=0.0,
        pass_through_negative_export=True,
    )


def _round_trip_prices() -> Prices:
    """Twee kwartieren: laden goedkoop (DA=IM=0.05), ontladen duur (DA=0.20, IM=0.50)."""
    idx = _quarter_index(2)
    da = pd.Series([0.05, 0.20], index=idx, name="day_ahead")
    im = pd.Series([0.05, 0.50], index=idx, name="imbalance")
    return Prices(day_ahead=da, imbalance=im, source="test", imbalance_source="test")


def test_imbalance_share_round_trip_at_full_share() -> None:
    """Bij volledige share captureert de gebruiker de IM-arbitrage: 0.50 - 0.05."""
    prices = _round_trip_prices()
    dispatch = pd.Series([+1.0, -1.0], index=prices.day_ahead.index)

    bonus = _imbalance_revenue_share(
        dispatch, prices, _frank_like_tariff(imbalance_revenue_share_to_user=1.0)
    )

    assert bonus == pytest.approx(0.30)


def test_imbalance_share_round_trip_at_zero_share_yields_no_bonus() -> None:
    """Zonder revenue-share is de bonus nul; DA-arbitrage zit elders in de boekhouding."""
    prices = _round_trip_prices()
    dispatch = pd.Series([+1.0, -1.0], index=prices.day_ahead.index)

    bonus = _imbalance_revenue_share(
        dispatch, prices, _frank_like_tariff(imbalance_revenue_share_to_user=0.0)
    )

    assert bonus == 0.0


def test_imbalance_share_pv_charge_still_credits_brp_deviation() -> None:
    """Laden vanuit ZP telt: de meter-afwijking is `dispatch`, ongeacht ZP."""
    idx = _quarter_index(1)
    prices = Prices(
        day_ahead=pd.Series([0.20], index=idx, name="day_ahead"),
        imbalance=pd.Series([0.05], index=idx, name="imbalance"),
        source="test",
        imbalance_source="test",
    )
    dispatch = pd.Series([+1.0], index=idx)  # 1 kWh laden, vanuit ZP of net.

    bonus = _imbalance_revenue_share(
        dispatch, prices, _frank_like_tariff(imbalance_revenue_share_to_user=1.0)
    )

    # Frank dispatcht 1 kWh met DA=0.20 en IM=0.05 → margin = da - im = 0.15.
    assert bonus == pytest.approx(0.15)


def test_run_scenario_no_dispatch_implies_no_imbalance_bonus() -> None:
    """End-to-end sanity: zonder dispatch is `imbalance_extra` nul, zelfs als
    `imbalance_trading=True`. Toont aan dat de helper alleen op echte dispatch
    geld toekent en niet op de inherente (load - pv) flow.
    """
    prices = _round_trip_prices()
    idx = prices.day_ahead.index
    zero = pd.Series(0.0, index=idx)
    load = LoadSeries(
        consumption_kwh=zero,
        pv_kwh=zero,
        grid_import_kwh=zero,
        grid_export_kwh=zero,
        gap_filled_index=pd.DatetimeIndex([]),
    )
    spec = BatterySpec(
        capacity_kwh=10.0,
        usable_fraction=1.0,
        max_charge_kw=4.0,
        max_discharge_kw=4.0,
        round_trip_efficiency=1.0,
    )
    tariff = _frank_like_tariff(imbalance_revenue_share_to_user=1.0)

    result = run_scenario("nb", "no_battery", load, prices, tariff, spec)

    assert result.breakdown["imbalance_extra"] == 0.0
    assert result.annual_cost_eur == pytest.approx(0.0)
