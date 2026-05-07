"""Tests voor onbalansprijs-fetchers: eerst ENTSO-E, daarna TenneT."""


import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from sim.config import Settings, load_settings
from sim.prices import (
    Prices,
    _fetch_entsoe_imbalance,
    _fetch_tennet_imbalance,
    _parse_settlement_csv,
    _settlement_to_single_price,
    fetch_or_synthesize,
)
from sim.userconfig import load_user_config


# Twee kwartieren: één tekortrij en één overschotrij.
# Kolommen spiegelen de echte TenneT-publicatie-CSV.
SAMPLE_CSV = (
    "Timeinterval Start Loc,Timeinterval End Loc,Isp,Currency Unit Name,"
    "Price Measurement Unit Name,Incident Reserve Up,Incident Reserve Down,"
    "Price Dispatch Up,Price Dispatch Down,Price Shortage,Price Surplus,"
    "Regulation State,Regulating Condition\n"
    "2025-07-01T00:00:00,2025-07-01T00:15:00,1,EUR,MWh,0,0,180.00,40.00,180.00,40.00,1,Up\n"
    "2025-07-01T00:00:00,2025-07-01T00:15:00,2,EUR,MWh,0,0,150.00,20.00,150.00,20.00,-1,Down\n"
    "2025-07-01T00:00:00,2025-07-01T00:15:00,3,EUR,MWh,0,0,80.00,60.00,80.00,60.00,0,None\n"
)


def test_parse_settlement_csv_builds_local_15min_timestamps() -> None:
    df = _parse_settlement_csv(SAMPLE_CSV)
    assert len(df) == 3
    expected = pd.DatetimeIndex(
        [
            "2025-07-01 00:00:00",
            "2025-07-01 00:15:00",
            "2025-07-01 00:30:00",
        ],
        tz="Europe/Amsterdam",
    )
    assert list(df.index) == list(expected)
    assert "Price Shortage" in df.columns
    assert "Regulation State" in df.columns


def test_settlement_to_single_price_picks_side_by_state() -> None:
    df = _parse_settlement_csv(SAMPLE_CSV)
    series = _settlement_to_single_price(df)
    # Row 1: state=+1 -> shortage 180/MWh -> 0.18 €/kWh
    # Row 2: state=-1 -> surplus 20/MWh -> 0.02 €/kWh
    # Row 3: state= 0 -> mid (80+60)/2 = 70 -> 0.07 €/kWh
    assert series.iloc[0] == pytest.approx(0.180)
    assert series.iloc[1] == pytest.approx(0.020)
    assert series.iloc[2] == pytest.approx(0.070)


def test_settlement_to_single_price_rejects_missing_columns() -> None:
    df = pd.DataFrame({"Price Shortage": [1.0], "Price Surplus": [1.0]})
    with pytest.raises(RuntimeError, match="Regulation State"):
        _settlement_to_single_price(df)


class _FakeResponse:
    def __init__(self, body: str, status: int = 200) -> None:
        self.text = body
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _build_day_csv(date: str) -> str:
    """Een volledige TenneT-dag-CSV: 96 ISP's op 100 €/MWh, toestand 0."""
    rows = ["Timeinterval Start Loc,Timeinterval End Loc,Isp,Currency Unit Name,"
            "Price Measurement Unit Name,Incident Reserve Up,Incident Reserve Down,"
            "Price Dispatch Up,Price Dispatch Down,Price Shortage,Price Surplus,"
            "Regulation State,Regulating Condition"]
    for isp in range(1, 97):
        rows.append(
            f"{date}T00:00:00,{date}T00:15:00,{isp},EUR,MWh,0,0,"
            f"100.00,100.00,100.00,100.00,0,None"
        )
    return "\n".join(rows) + "\n"


def test_fetch_tennet_imbalance_returns_96_quarters_for_one_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eén recente dag moet terugkomen als 96 kwartierbuckets."""
    settings = Settings(cache_dir=tmp_path, output_dir=tmp_path)
    monkeypatch.setenv("TENNET_API_KEY", "test-key-do-not-call-real-api")

    start = datetime(2025, 7, 1, tzinfo=timezone.utc)
    end = datetime(2025, 7, 2, tzinfo=timezone.utc)

    captured: dict[str, object] = {}

    def fake_get(url: str, headers: dict, params: dict, timeout: int) -> _FakeResponse:
        captured["url"] = url
        captured["headers"] = headers
        captured["params"] = params
        return _FakeResponse(_build_day_csv("2025-07-01"))

    with patch("sim.prices.requests.get", side_effect=fake_get):
        series = _fetch_tennet_imbalance(settings, start, end)

    # 96 kwartieren voor de dag in Europe/Amsterdam, daarna omgezet naar UTC.
    assert len(series) == 96
    assert str(series.index.tz) == "UTC"
    # Alle rijen zijn 100 €/MWh = 0.10 €/kWh.
    assert series.tolist() == pytest.approx([0.10] * 96)
    # Header bevat de API-key.
    assert captured["headers"]["apikey"] == "test-key-do-not-call-real-api"
    # Cachebestand is geschreven en herbruikbaar.
    cache_file = tmp_path / "20250701_20250702_imbalance.parquet"
    assert cache_file.exists()


def test_fetch_tennet_imbalance_raises_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(cache_dir=tmp_path, output_dir=tmp_path)
    monkeypatch.delenv("TENNET_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TENNET_API_KEY"):
        _fetch_tennet_imbalance(
            settings,
            datetime(2025, 7, 1, tzinfo=timezone.utc),
            datetime(2025, 7, 2, tzinfo=timezone.utc),
        )


def test_fetch_or_synthesize_falls_back_to_synthetic_imbalance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Als TenneT faalt maar day-ahead werkt, wordt onbalans gesynthetiseerd."""
    monkeypatch.delenv("TENNET_API_KEY", raising=False)
    # Zet entsoe_api_key expliciet op None zodat de ENTSO-E fallback uit staat.
    settings = Settings(cache_dir=tmp_path, output_dir=tmp_path, entsoe_api_key=None)

    start = datetime(2025, 7, 1, tzinfo=timezone.utc)
    end = datetime(2025, 7, 2, tzinfo=timezone.utc)

    # Stub day-ahead zodat de test niet afhangt van EnergyZero-netwerkverkeer.
    fake_da = pd.Series(
        0.10,
        index=pd.date_range(start, end, freq="15min", tz="UTC", inclusive="left"),
        name="day_ahead",
    )
    with patch("sim.prices._fetch_day_ahead", return_value=(fake_da, "energyzero")):
        prices: Prices = fetch_or_synthesize(settings, start, end)

    assert prices.source == "energyzero"
    assert prices.imbalance_source == "synthetic"
    assert len(prices.imbalance) == 96


def test_load_settings_uses_toml_simulation_values(tmp_path: Path) -> None:
    config = tmp_path / "user.toml"
    config.write_text(
        """
[simulation]
questdb_url = "http://questdb.example:9000"
entsoe_api_key = "toml-key"
""".strip()
    )

    user = load_user_config(config)
    settings = load_settings(user)

    assert settings.questdb_url == "http://questdb.example:9000"
    assert settings.entsoe_api_key == "toml-key"


def test_fetch_entsoe_imbalance_collapses_long_short_to_mid_kwh(
    tmp_path: Path,
) -> None:
    """ENTSO-E geeft Long/Short in EUR/MWh; wij nemen mid in EUR/kWh."""
    settings = Settings(cache_dir=tmp_path, output_dir=tmp_path, entsoe_api_key="x")
    start = datetime(2025, 7, 1, tzinfo=timezone.utc)
    end = datetime(2025, 7, 2, tzinfo=timezone.utc)

    # 96 kwartieren in Europe/Brussels. Long=120, Short=80, mid=100 EUR/MWh.
    idx = pd.date_range(
        "2025-07-01", periods=96, freq="15min", tz="Europe/Brussels"
    )
    fake_df = pd.DataFrame({"Long": [120.0] * 96, "Short": [80.0] * 96}, index=idx)

    fake_client = MagicMock()
    fake_client.query_imbalance_prices.return_value = fake_df

    with patch("entsoe.EntsoePandasClient", return_value=fake_client):
        series = _fetch_entsoe_imbalance(settings, start, end)

    assert len(series) == 96
    assert str(series.index.tz) == "UTC"
    assert series.tolist() == pytest.approx([0.10] * 96)
    # Cache file written for reuse.
    assert (tmp_path / "20250701_20250702_imbalance_entsoe.parquet").exists()
    fake_client.query_imbalance_prices.assert_called()


@pytest.mark.skipif(
    not os.environ.get("TENNET_API_KEY"),
    reason="Live TenneT API call requires TENNET_API_KEY",
)
def test_live_tennet_one_day_round_trip(tmp_path: Path) -> None:
    """Live integration test: fetch a single recent day, expect 96 quarters."""
    settings = Settings(cache_dir=tmp_path, output_dir=tmp_path)
    start = datetime(2025, 7, 1, tzinfo=timezone.utc)
    end = datetime(2025, 7, 2, tzinfo=timezone.utc)
    series = _fetch_tennet_imbalance(settings, start, end)
    assert len(series) == 96
    assert series.notna().all()


def test_live_entsoe_one_day_round_trip(tmp_path: Path) -> None:
    """Live integration test: real ENTSO-E imbalance, single recent day."""
    user = load_user_config()
    if not user.simulation.entsoe_api_key:
        pytest.skip("Live ENTSO-E imbalance fetch requires entsoe_api_key in TOML")
    settings = load_settings(user)
    settings.cache_dir = tmp_path
    settings.output_dir = tmp_path
    start = datetime(2025, 7, 1, tzinfo=timezone.utc)
    end = datetime(2025, 7, 2, tzinfo=timezone.utc)
    series = _fetch_entsoe_imbalance(settings, start, end)
    # ENTSO-E returns 96 quarters per day for NL (15-min settlement).
    assert 90 <= len(series) <= 100
    assert series.notna().all()
