"""Tests for ``compute_returns``.

These are written before the implementation (test-first): ``compute_returns``
does not exist yet, so importing it makes the suite go red until the engine
defines what simple returns should look like.
"""

import numpy as np
import pandas as pd

from walkforward.engine import compute_returns


def test_compute_returns_simple_arithmetic():
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"A": [100.0, 200.0, 100.0]}, index=dates)

    returns = compute_returns(prices)

    assert np.isnan(returns["A"].iloc[0])  # no prior price
    assert returns["A"].iloc[1] == 1.0  # 200/100 - 1
    assert returns["A"].iloc[2] == -0.5  # 100/200 - 1


def test_compute_returns_preserves_index_and_columns():
    dates = pd.date_range("2026-01-01", periods=2, freq="D")
    prices = pd.DataFrame({"A": [10.0, 11.0], "B": [50.0, 55.0]}, index=dates)

    returns = compute_returns(prices)

    assert list(returns.columns) == ["A", "B"]
    assert returns.index.equals(prices.index)
    assert returns["A"].iloc[1] == 0.1
    assert returns["B"].iloc[1] == 0.1
