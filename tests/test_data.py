"""Tests voor gap-fill gedrag."""

import numpy as np
import pandas as pd

from sim.data import gap_fill_scaled_shape


def test_gap_fill_preserves_daily_total() -> None:
    """Een geschaalde shape-fill moet optellen tot het echte dagtotaal."""
    # 4 dagen: 1-4 juni, met 4 juni als slechte dag.
    idx = pd.date_range("2025-06-01", "2025-06-05", freq="15min", tz="UTC", inclusive="left")
    pattern = [0.05] * 32 + [0.20] * 32 + [0.05] * 32  # telt op tot 9.6 per dag
    # 3 goede dagen plus 96 nullen voor de slechte dag.
    series = pd.Series(pattern * 3 + [0.0] * 96, index=idx, name="v").astype(float)

    bad = pd.DatetimeIndex([pd.Timestamp("2025-06-04", tz="UTC")])
    daily_targets = pd.Series(
        {
            pd.Timestamp("2025-06-01", tz="UTC"): 9.6,
            pd.Timestamp("2025-06-02", tz="UTC"): 9.6,
            pd.Timestamp("2025-06-03", tz="UTC"): 9.6,
            pd.Timestamp("2025-06-04", tz="UTC"): 12.0,  # echte import van deze dag
        }
    )
    filled, gap_idx = gap_fill_scaled_shape(series, bad, daily_targets)

    day4 = filled.loc["2025-06-04"]
    assert abs(day4.sum() - 12.0) < 1e-6
    assert len(gap_idx) == 96


def test_gap_fill_uses_uniform_shape_for_zero_good_days_without_numpy_warning() -> None:
    """Een nulprofiel mag niet leunen op globale NumPy-waarschuwingsonderdrukking."""
    idx = pd.date_range("2025-06-01", "2025-06-03", freq="15min", tz="UTC", inclusive="left")
    series = pd.Series([0.0] * len(idx), index=idx, name="v")
    bad = pd.DatetimeIndex([pd.Timestamp("2025-06-02", tz="UTC")])
    daily_targets = pd.Series({pd.Timestamp("2025-06-02", tz="UTC"): 9.6})

    old_err = np.seterr(invalid="raise", divide="raise")
    try:
        filled, gap_idx = gap_fill_scaled_shape(series, bad, daily_targets)
    finally:
        np.seterr(**old_err)

    day2 = filled.loc["2025-06-02"]
    assert abs(day2.sum() - 9.6) < 1e-6
    assert day2.nunique() == 1
    assert len(gap_idx) == 96
