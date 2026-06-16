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

import pandas as pd


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
    result
        The backtest result (return series plus core metrics).
    """
    raise NotImplementedError
