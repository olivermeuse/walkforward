"""Tests for ``compute_returns_series``.

Written before the implementation (test-first): ``compute_returns_series`` does
not exist yet, so importing it makes the suite go red. These tests pin the
``t -> t + 1`` alignment (shift positions forward one period), that flat
positions earn nothing, and the cross-sectional sum across assets.
"""

import pandas as pd
import pytest

from walkforward.engine import compute_returns_series


def test_single_asset_alignment():
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    returns = pd.DataFrame({"A": [float("nan"), 0.10, 0.10]}, index=dates)
    positions = pd.DataFrame({"A": [1.0, 1.0, 0.0]}, index=dates)

    series = compute_returns_series(positions, returns)

    # Day 0: no prior position -> 0.0. Day 1: day-0 position (1.0) earns the
    # day-1 return (0.10). Day 2: day-1 position (1.0) earns the day-2 return.
    assert series.iloc[0] == pytest.approx(0.0)
    assert series.iloc[1] == pytest.approx(0.10)
    assert series.iloc[2] == pytest.approx(0.10)


def test_flat_position_earns_nothing():
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    returns = pd.DataFrame({"A": [float("nan"), 0.10, -0.20]}, index=dates)
    positions = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=dates)

    series = compute_returns_series(positions, returns)

    assert series.iloc[0] == pytest.approx(0.0)
    assert series.iloc[1] == pytest.approx(0.0)
    assert series.iloc[2] == pytest.approx(0.0)


def test_two_asset_sum():
    dates = pd.date_range("2026-01-01", periods=2, freq="D")
    returns = pd.DataFrame(
        {"A": [float("nan"), 0.10], "B": [float("nan"), 0.05]}, index=dates
    )
    positions = pd.DataFrame({"A": [1.0, 1.0], "B": [1.0, 1.0]}, index=dates)

    series = compute_returns_series(positions, returns)

    # Day 1: both day-0 positions (1.0 each) earn their day-1 returns; the
    # portfolio return is the cross-sectional sum.
    assert series.iloc[0] == pytest.approx(0.0)
    assert series.iloc[1] == pytest.approx(0.10 + 0.05)
