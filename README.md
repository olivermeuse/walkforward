![CI](https://github.com/olivermeuse/walkforward/actions/workflows/ci.yml/badge.svg)

# walkforward

**A backtesting engine that is fast by construction and correct by construction.** Walk-forward backtests run in minutes on a laptop where naive versions take hours, and the interface makes lookahead bias structurally hard to introduce.

> Status: pre-`v0.1.0`, built in public. v1 is the engine only — see the roadmap for what comes after.

## The problem

Most retail backtests are slow *and* quietly wrong, usually for the same fixable reasons:

- **Slow** because work that should happen once (fitting, retraining, index construction) gets recomputed inside the per-day loop — an O(n²) trap that turns a minutes-long job into an hours-long one.
- **Wrong** because future information leaks into past decisions (lookahead bias), or because P&L is booked on raw prices while signals use adjusted prices, so splits and dividends show up as phantom returns.

`walkforward` is structured so those specific mistakes are hard or impossible to make: fit/retrain work is cached, P&L is vectorized on total-return prices, and your strategy is only ever handed past data.

## Install

```bash
pip install walkforward
```

(Until `v0.1.0` is published, install from source: `pip install git+https://github.com/<you>/walkforward`.)

## Usage

The entire interface is one function signature: `strategy(data_up_to_today) -> positions`. Because the function only ever receives past rows, it cannot see the future.

```python
import pandas as pd
from walkforward import backtest

# Price data: a DataFrame indexed by date, one column per asset.
# Use adjusted / total-return prices so dividends and splits don't book phantom P&L.
prices = pd.read_csv("prices.csv", index_col="date", parse_dates=True)

# Your strategy receives only the rows up to and including today, and returns
# the positions to HOLD over the next day. It can never be handed a future row.
def strategy(data_up_to_today):
    fast = data_up_to_today.tail(10).mean()
    slow = data_up_to_today.tail(50).mean()
    return (fast > slow).astype(float)        # 1.0 = long, 0.0 = flat, per asset

result = backtest(prices, strategy)

print(result.metrics)    # {'total_return': ..., 'sharpe': ..., 'max_drawdown': ..., 'turnover': ...}
result.returns.plot()    # the per-period return series
```

**Timing convention (the thing that makes it correct):** `strategy` sees every row through day *t* and returns the positions held from *t* to *t + 1*. The engine applies those positions to the *t → t + 1* total return. The decision strictly precedes the return it earns, so there is no overlap for future data to leak through.

## What v1 is

**Does:** a correct walk-forward backtest; fit/retrain work done once and cached rather than per-day; P&L vectorized on adjusted / total-return prices; outputs the return series plus core metrics (total return, Sharpe, max drawdown, turnover).

**Does not (on purpose, so v1 ships):** no data fetching, no GUI, no built-in strategies, no parameter optimization, no multi-asset portfolio construction, and no statistical auditor yet. Those are roadmap, not v1.

## Roadmap

**v2 — the rigor layer (the eventual differentiator).** Most backtesters are fast but let you fool yourself. v2 makes correct statistical methodology the default:
- **Deflated Sharpe ratio** — corrects for how many strategies you tried before this one (multiple-testing).
- **Permutation / shuffle controls** — prove an edge isn't a mechanical artifact (random predictions should produce no edge).
- **Mandatory out-of-sample / regime-split testing** — no single-period result stands alone.

Positioning: *the backtester that won't let you fool yourself* — every check is here because skipping it produces a confident, wrong answer.

**v3+ (optional, later).** Data helpers, multi-asset portfolio support, parameter optimization.

## License

MIT
