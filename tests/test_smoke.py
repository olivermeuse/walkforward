"""Smoke test: the package imports and ``backtest`` is callable."""


def test_package_imports():
    import walkforward

    assert hasattr(walkforward, "backtest")


def test_backtest_is_callable():
    from walkforward import backtest

    assert callable(backtest)
