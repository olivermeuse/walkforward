"""The walk-forward backtest engine.

Contract
--------
``backtest(prices, strategy) -> result``

``strategy(data_up_to_today)`` receives every row up to and including day ``t``
and returns the positions to hold over ``t -> t + 1``.

Timing rule (must never break)
------------------------------
Positions derived from data through day ``t`` apply to the ``t -> t + 1``
return. The decision strictly precedes the return it earns, so no future data
can leak backward into a past decision.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    """The result of a backtest.

    Attributes
    ----------
    returns:
        A pandas Series of per-period portfolio returns.
    metrics:
        A dict with keys ``total_return``, ``sharpe``, ``max_drawdown`` and
        ``turnover``.
    """

    returns: pd.Series
    metrics: dict


def compute_returns(prices):
    """Compute simple (arithmetic) per-period returns from a price frame.

    Each value is ``current / prior - 1``; the first row is ``NaN`` (no prior
    price). The index and column order are preserved. This is computed once, up
    front, in a single vectorized pass — never per row inside the backtest loop.

    ``returns.loc[d]`` is the return realized from the prior row to row ``d``.
    """
    return prices.pct_change()


def run_walk_forward(prices, strategy):
    """Walk forward through ``prices``, collecting each day's positions.

    For every date ``t`` in ``prices.index`` (in order, exactly once) the
    strategy is called with ``prices.loc[:t]`` -- the slice up to and including
    ``t``. The slice ends at the current date, so the strategy can never receive
    a future row: the no-lookahead guarantee is structural.

    The loop does only one thing: call the strategy and store its output. No
    P&L, no shift, no multiplication by returns happens here. Positions are
    stored *unshifted*, keyed by the date whose data produced them; the
    ``t -> t + 1`` alignment is applied downstream, vectorized.

    Returns ``(positions, returns)``: ``positions`` is a DataFrame indexed by
    ``prices.index`` with one column per asset (column order matching
    ``prices``), and ``returns`` is ``compute_returns(prices)``.
    """
    records = {}
    for date in prices.index:
        records[date] = strategy(prices.loc[:date])

    positions = pd.DataFrame(records).T.reindex(
        index=prices.index, columns=prices.columns
    )
    returns = compute_returns(prices)
    return positions, returns


def compute_returns_series(positions, returns):
    """Combine unshifted positions with returns into one portfolio return Series.

    ``positions.shift(1)`` applies the ``t -> t + 1`` alignment once over the
    whole matrix: a position held into day ``d`` was decided at ``d - 1`` and
    earns day ``d``'s return. The cross-sectional ``sum(axis=1)`` across assets
    is valid because returns are simple / arithmetic, so per-asset contributions
    add. The first row is ``0.0`` because no position is held into day one.
    """
    return (positions.shift(1) * returns).sum(axis=1).fillna(0.0)


def total_return(returns_series):
    """Geometric total return over the whole series.

    Compounds the per-period returns -- it does not sum them:
    ``(1 + r).prod() - 1``.
    """
    return (1 + returns_series).prod() - 1


def sharpe(returns_series, periods_per_year=252):
    """Annualized Sharpe ratio of a per-period return series.

    ``mean / std * sqrt(periods_per_year)`` with sample std (``ddof=1``). The
    risk-free rate is assumed 0 in v1 -- a stated simplification.
    ``periods_per_year`` is a parameter so non-daily data annualizes correctly;
    never hardcode 252. If the std is 0 or NaN, returns 0.0 rather than dividing
    by zero.
    """
    std = returns_series.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return returns_series.mean() / std * np.sqrt(periods_per_year)


def max_drawdown(returns_series):
    """Maximum drawdown of the equity curve, measured from the running peak.

    Builds the equity curve ``(1 + r).cumprod()``, compares it to its running
    max (``cummax``), and returns the most negative ``equity / peak - 1``.
    Measured from the running peak, not the start, and reported as a negative
    number; 0.0 if the curve is never underwater.
    """
    equity = (1 + returns_series).cumprod()
    drawdown = equity / equity.cummax() - 1
    return min(drawdown.min(), 0.0)


def turnover(positions):
    """Average per-period gross position change.

    ``positions.diff().abs().sum(axis=1)`` is the per-period gross change across
    assets; the first row's diff is undefined and is dropped positionally before
    averaging. A proxy for trading activity / transaction-cost drag, which v1
    does not model.
    """
    return positions.diff().abs().sum(axis=1).iloc[1:].mean()


def backtest(prices, strategy):
    """Run a walk-forward backtest.

    Parameters
    ----------
    prices:
        A DataFrame indexed by date, one column per asset. Should be adjusted /
        total-return prices so that splits and dividends do not book phantom
        P&L.
    strategy:
        A callable ``strategy(data_up_to_today) -> positions``. It is handed
        every row up to and including day ``t`` and returns the positions to
        hold over ``t -> t + 1``.

    Returns
    -------
    BacktestResult
        ``result.returns`` is the per-period portfolio return Series;
        ``result.metrics`` is a dict with ``total_return``, ``sharpe``,
        ``max_drawdown`` and ``turnover``.

    This is a pure composition of the tested components -- it introduces no new
    computation of its own.
    """
    positions, returns = run_walk_forward(prices, strategy)
    r = compute_returns_series(positions, returns)
    metrics = {
        "total_return": total_return(r),
        "sharpe": sharpe(r),
        "max_drawdown": max_drawdown(r),
        "turnover": turnover(positions),
    }
    return BacktestResult(returns=r, metrics=metrics)
