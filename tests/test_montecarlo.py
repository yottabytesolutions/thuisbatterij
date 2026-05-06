from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sim.montecarlo import _build_entsoe_imbalance_for_span


def test_build_entsoe_imbalance_for_span_uses_mei_mei_cache(tmp_path: Path) -> None:
    target_index = pd.date_range(
        "2025-05-01", periods=4, freq="15min", tz="UTC"
    )
    hist_index = pd.date_range("2019-05-01", periods=4, freq="15min", tz="UTC")
    expected = [0.11, 0.12, 0.13, 0.14]
    pd.Series(expected, index=hist_index, name="imbalance").to_frame().to_parquet(
        tmp_path / "20190501_20200501_imbalance_entsoe.parquet"
    )

    series = _build_entsoe_imbalance_for_span(tmp_path, 2019, target_index)

    assert series is not None
    assert series.index.equals(target_index)
    assert series.name == "imbalance"
    assert series.tolist() == pytest.approx(expected)


def test_build_entsoe_imbalance_for_span_returns_none_without_cache(
    tmp_path: Path,
) -> None:
    target_index = pd.date_range(
        "2025-05-01", periods=4, freq="15min", tz="UTC"
    )

    assert _build_entsoe_imbalance_for_span(tmp_path, 2019, target_index) is None