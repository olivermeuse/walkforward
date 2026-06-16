"""Tests for ``run_walk_forward``.

Written before the implementation (test-first): ``run_walk_forward`` does not
exist yet, so importing it makes the suite go red. These tests pin the two
properties the engine must never break: the strategy only ever sees past data,
and a position taken at day ``t`` earns the ``t -> t + 1`` return (no lookahead).
"""

import pandas as pd
import pytest

from walkforward.engine import run_walk_forward


def test_strategy_only_sees_past():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=dates)

    seen_max = []

    def spy(data_up_to_today):
        seen_max.append(data_up_to_today.index.max())
        return pd.Series(0.0, index=data_up_to_today.columns)

    run_walk_forward(prices, spy)

    # On each call the strategy was handed rows ending exactly at the current
    # day -- never a later date. Calls happen in order, one per row.
    assert seen_max == list(prices.index)


def test_called_once_per_day():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=dates)

    calls = {"n": 0}

    def counting(data_up_to_today):
        calls["n"] += 1
        return pd.Series(0.0, index=data_up_to_today.columns)

    run_walk_forward(prices, counting)

    assert calls["n"] == len(prices)


def test_position_earns_next_period_return():
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0, 100.0]}, index=dates)

    D = dates[1]

    def strategy(data_up_to_today):
        # Long (1.0) only once we have seen date D; flat (0.0) before.
        val = 1.0 if data_up_to_today.index.max() >= D else 0.0
        return pd.Series(val, index=data_up_to_today.columns)

    positions, returns = run_walk_forward(prices, strategy)

    # Shift positions forward by one period: the position decided on day t lines
    # up with the t -> t+1 return. The return earned by the position taken at D
    # lands on D+1, never on D itself.
    pnl = positions.shift(1) * returns

    assert pnl["A"].loc[dates[2]] == pytest.approx(returns["A"].loc[dates[2]])
    assert pnl["A"].loc[dates[1]] == pytest.approx(0.0)
