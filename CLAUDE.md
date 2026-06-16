# CLAUDE.md

Guidance for working in this repository.

## Contract

`backtest(prices, strategy) -> result`, where `strategy(data_up_to_today)`
receives every row up to and including day `t` and returns the positions to
hold over `t -> t + 1`.

**Timing rule that must never break:** positions from data through day `t` apply
to the `t -> t + 1` return, so the decision strictly precedes the return it
earns and no future data can leak backward.

## v1 scope is LOCKED

**DOES:**
- correct walk-forward backtest
- fit/retrain cached, not per-day
- P&L vectorized on adjusted / total-return prices
- outputs return series + total return, Sharpe, max drawdown, turnover

**DOES NOT:**
- data fetching
- GUI
- built-in strategies
- optimization
- multi-asset portfolios
- statistical auditor

## Do not pull v2 features into v1

v2 features — deflated Sharpe, permutation / shuffle controls, out-of-sample /
regime-split testing — stay out of v1. If a change heads that way, **stop and
flag it.**

## Fast by construction

Nothing precomputable runs inside the per-day loop:
- build the returns matrix once;
- in the loop, only call the strategy and store its output;
- compute all P&L vectorized after the loop.

Never rebuild an index or refit inside the loop.

## Test-first

Write the failing test before implementing. Keep `backtest` raising
`NotImplementedError` until a test defines what it should do.
