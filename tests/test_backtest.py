"""End-to-end tests for ``backtest``.

Written before the implementation (test-first): ``backtest`` still raises
``NotImplementedError``, so these tests go red. They pin the result interface
(``.returns`` Series, ``.metrics`` dict with exactly four keys) and prove that
``backtest`` is a faithful composition of the already-tested pieces -- it must
introduce no new logic of its own.
"""

import pandas as pd
import pytest

from walkforward import backtest
from walkforward.engine import (
    compute_returns_series,
    max_drawdown,
    run_walk_forward,
    sharpe,
    total_return,
    turnover,
)


def always_long(data_up_to_today):
    return pd.Series(1.0, index=data_up_to_today.columns)


def test_backtest_returns_result_object():
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=dates)

    result = backtest(prices, always_long)

    assert isinstance(result.returns, pd.Series)
    assert isinstance(result.metrics, dict)
    assert set(result.metrics.keys()) == {
        "total_return",
        "sharpe",
        "max_drawdown",
        "turnover",
    }


def test_backtest_end_to_end_matches_components():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    prices = pd.DataFrame(
        {"A": [100.0, 105.0, 103.0, 108.0, 110.0]}, index=dates
    )

    result = backtest(prices, always_long)

    # Recompute the pieces independently.
    positions, returns = run_walk_forward(prices, always_long)
    expected_series = compute_returns_series(positions, returns)

    pd.testing.assert_series_equal(result.returns, expected_series)

    assert result.metrics["total_return"] == pytest.approx(
        total_return(expected_series)
    )
    assert result.metrics["sharpe"] == pytest.approx(sharpe(expected_series))
    assert result.metrics["max_drawdown"] == pytest.approx(
        max_drawdown(expected_series)
    )
    assert result.metrics["turnover"] == pytest.approx(turnover(positions))


def test_backtest_always_long_total_return():
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=dates)

    result = backtest(prices, always_long)

    # Day-0 position earns day-1's 10%, day-1 position earns day-2's 10%;
    # compounded: 1.1 * 1.1 - 1 = 0.21.
    assert result.metrics["total_return"] == pytest.approx(0.21)
