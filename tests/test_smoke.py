"""Smoke test: the package imports and ``backtest`` is callable.

This is the failing test that comes before the implementation. ``backtest`` is
still a stub raising ``NotImplementedError`` (see CLAUDE.md: test-first), so we
only assert that the symbol exists and is callable here. A test that defines the
actual behaviour must be written before the engine is implemented.
"""

import pytest


def test_package_imports():
    import walkforward

    assert hasattr(walkforward, "backtest")


def test_backtest_is_callable():
    from walkforward import backtest

    assert callable(backtest)


def test_backtest_is_still_a_stub():
    from walkforward import backtest

    with pytest.raises(NotImplementedError):
        backtest(None, None)
