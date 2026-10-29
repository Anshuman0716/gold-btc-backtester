import numpy as np
import pandas as pd
import pytest

from analytics.tearsheet import compute_tearsheet, get_annualization_factor
from risk.controls import RiskParams, Trade, apply_risk_controls
from strategy.ema_crossover import EMACrossover


def _trade(pnl, pct):
    t = pd.Timestamp("2024-01-01")
    return Trade(t, t, "LONG", 100, 100, 1, 90, 110, pnl=pnl, pnl_pct=pct)


def test_max_drawdown_known_value():
    eq = pd.Series([100.0, 120.0, 90.0, 110.0], index=pd.date_range("2024-01-01", periods=4))
    ts = compute_tearsheet([], eq, initial_capital=100.0)
    assert ts.max_drawdown == pytest.approx(-0.25)
    assert ts.total_return == pytest.approx(0.10)


def test_win_rate_and_profit_factor():
    eq = pd.Series(np.linspace(100, 110, 20), index=pd.date_range("2024-01-01", periods=20))
    trades = [_trade(30, 0.3), _trade(-10, -0.1), _trade(20, 0.2), _trade(-10, -0.1)]
    ts = compute_tearsheet(trades, eq, initial_capital=100.0)
    assert ts.win_rate == pytest.approx(0.5)
    assert ts.profit_factor == pytest.approx(50 / 20)
    assert ts.num_trades == 4


def test_short_equity_returns_empty_tearsheet():
    eq = pd.Series([100.0], index=pd.date_range("2024-01-01", periods=1))
    assert compute_tearsheet([], eq).num_trades == 0


def test_annualization_crypto_vs_gold():
    assert get_annualization_factor("BTC-USD", "1d") > get_annualization_factor("GC=F", "1d")


def test_benchmark_pipeline_is_deterministic(btc):
    def run():
        sig = EMACrossover(9, 21).generate_signals(btc)
        trades, eq = apply_risk_controls(sig, btc, RiskParams())
        return compute_tearsheet(trades, eq).total_return
    assert run() == run()
