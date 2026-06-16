"""Tests for the output metrics.

Written before the implementation (test-first): ``total_return``, ``sharpe``,
``max_drawdown`` and ``turnover`` do not exist yet, so importing them makes the
suite go red. These tests pin geometric compounding, parametric Sharpe
annualization, drawdown measured from the running peak, and turnover as the mean
per-period absolute position change.
"""

import numpy as np
import pandas as pd
import pytest

from walkforward.engine import (
    max_drawdown,
    sharpe,
    total_return,
    turnover,
)


def test_total_return_compounds():
    s = pd.Series([0.10, 0.10])
    # 1.1 * 1.1 - 1 = 0.21 (geometric), not 0.20 (summation).
    assert total_return(s) == pytest.approx(0.21)


def test_total_return_with_loss():
    s = pd.Series([0.50, -0.50])
    # 1.5 * 0.5 - 1 = -0.25.
    assert total_return(s) == pytest.approx(-0.25)


def test_sharpe_annualization_is_parametric():
    s = pd.Series([0.01, 0.02, -0.01, 0.03])
    base = s.mean() / s.std(ddof=1)

    assert sharpe(s, periods_per_year=252) == pytest.approx(base * np.sqrt(252))
    # Changing periods_per_year rescales by sqrt of the new factor: the
    # annualization factor is a parameter, not hardcoded.
    assert sharpe(s, periods_per_year=12) == pytest.approx(base * np.sqrt(12))


def test_sharpe_zero_volatility():
    s = pd.Series([0.01, 0.01, 0.01])
    # Zero std must yield 0.0, not NaN or a divide-by-zero error.
    assert sharpe(s) == pytest.approx(0.0)


def test_max_drawdown_measures_from_peak():
    s = pd.Series([0.50, -0.50])
    # Equity curve 1.0 -> 1.5 -> 0.75. Drawdown is measured from the running
    # peak (1.5), so 0.75/1.5 - 1 = -0.5, reported negative.
    assert max_drawdown(s) == pytest.approx(-0.5)


def test_max_drawdown_never_underwater():
    s = pd.Series([0.10, 0.20, 0.30])
    # Monotonically increasing equity -> never below its peak.
    assert max_drawdown(s) == pytest.approx(0.0)


def test_turnover_counts_position_changes():
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    positions = pd.DataFrame({"A": [0.0, 1.0, 1.0, 0.0]}, index=dates)
    # Per-period absolute changes: first row is NaN (no prior position) and is
    # dropped before averaging, leaving [1.0, 0.0, 1.0]; mean = 2/3.
    assert turnover(positions) == pytest.approx(2.0 / 3.0)
