"""Execution-engine behaviour on hand-built bars where the right answer is known."""

import pandas as pd
import pytest

from risk.controls import RiskParams, apply_risk_controls
from tests.conftest import make_ohlc


def _signals(n, **at):
    s = pd.Series("HOLD", index=pd.date_range("2024-01-01", periods=n, freq="D"))
    for i, v in at.items():
        s.iloc[int(i[1:])] = v
    return s


def _flat(n, price=100.0, **kw):
    """Flat market with a small range so ATR is positive but SL/TP are never hit."""
    return make_ohlc(
        [price] * n,
        highs=[price + 1] * n,
        lows=[price - 1] * n,
        closes=[price] * n,
    )


PARAMS = dict(initial_capital=100_000, atr_period=3, max_drawdown_pct=0.99)


def test_fills_at_next_open_not_signal_close():
    df = _flat(30)
    df.loc[df.index[20], "open"] = 105.0   # next-bar open differs from signal close
    df.loc[df.index[20], "high"] = 106.0
    trades, _ = apply_risk_controls(_signals(30, b19="BUY"), df, RiskParams(**PARAMS))
    assert trades[0].entry_time == df.index[20]
    assert trades[0].entry_price == pytest.approx(105.0)


def test_costs_reduce_pnl_and_are_recorded():
    df = _flat(30)
    sig = _signals(30, b10="BUY", b20="SELL")
    free, _ = apply_risk_controls(sig, df, RiskParams(**PARAMS))
    paid, _ = apply_risk_controls(sig, df, RiskParams(**PARAMS, commission_bps=10, slippage_bps=10))
    assert free[0].fee == 0
    assert paid[0].fee > 0
    assert paid[0].pnl < free[0].pnl


def test_stop_loss_hit():
    closes = [100.0] * 30
    df = make_ohlc(closes, highs=[101.0] * 30, lows=[99.0] * 30, closes=closes)
    df.loc[df.index[15], "low"] = 50.0
    trades, _ = apply_risk_controls(_signals(30, b10="BUY"), df, RiskParams(**PARAMS))
    assert trades[0].exit_reason == "stop_loss"
    assert trades[0].pnl < 0


def test_pessimistic_same_bar_assumes_stop():
    closes = [100.0] * 30
    df = make_ohlc(closes, highs=[101.0] * 30, lows=[99.0] * 30, closes=closes)
    df.loc[df.index[15], ["high", "low"]] = [500.0, 5.0]   # touches both SL and TP
    sig = _signals(30, b10="BUY")
    pess, _ = apply_risk_controls(sig, df, RiskParams(**PARAMS, pessimistic_fills=True))
    opt, _ = apply_risk_controls(sig, df, RiskParams(**PARAMS, pessimistic_fills=False))
    assert pess[0].exit_reason == "stop_loss"
    assert opt[0].exit_reason == "take_profit"


def test_leverage_cap_limits_notional():
    df = _flat(30)
    trades, _ = apply_risk_controls(
        _signals(30, b10="BUY"), df,
        RiskParams(**PARAMS, risk_per_trade=0.5, max_leverage=1.0),
    )
    t = trades[0]
    assert t.entry_price * t.position_size <= 100_000 * 1.0 + 1e-6


def test_open_position_closed_at_end_of_data():
    df = _flat(30)
    trades, eq = apply_risk_controls(_signals(30, b10="BUY"), df, RiskParams(**PARAMS))
    assert trades[-1].exit_reason == "end_of_data"
    assert trades[-1].exit_time == df.index[-1]


def test_equity_curve_aligned_and_starts_at_capital():
    df = _flat(30)
    _, eq = apply_risk_controls(_signals(30), df, RiskParams(**PARAMS))
    assert eq.index.equals(df.index)
    assert eq.iloc[0] == 100_000
    assert eq.notna().all()


def _crash_frame():
    """Long, then a large gap down that trips a 5% kill-switch."""
    n = 60
    closes = [100.0] * n
    for i in range(25, n):
        closes[i] = 80.0
    opens = closes[:]
    highs = [c + 1 for c in closes]
    lows = [c - 1 for c in closes]
    # keep the gap bar's low above the stop so the kill-switch (not SL) fires
    return make_ohlc(opens, highs, lows, closes)


def test_kill_switch_latch_stops_new_trades():
    df = _crash_frame()
    sig = _signals(60, b10="BUY", b40="BUY", b50="SELL")
    p = RiskParams(initial_capital=100_000, atr_period=3, risk_per_trade=0.05,
                   stop_loss_atr=100.0, take_profit_atr=100.0,
                   max_drawdown_pct=0.05, kill_switch_mode="latch", max_leverage=1.0)
    trades, _ = apply_risk_controls(sig, df, p)
    assert any(t.exit_reason == "kill_switch" for t in trades)
    assert all(t.entry_time <= df.index[30] for t in trades)


def test_kill_switch_reset_resumes_trading():
    df = _crash_frame()
    sig = _signals(60, b10="BUY", b40="BUY", b50="SELL")
    p = RiskParams(initial_capital=100_000, atr_period=3, risk_per_trade=0.05,
                   stop_loss_atr=100.0, take_profit_atr=100.0,
                   max_drawdown_pct=0.05, kill_switch_mode="reset",
                   kill_switch_cooldown_bars=5, max_leverage=1.0)
    trades, _ = apply_risk_controls(sig, df, p)
    assert any(t.entry_time > df.index[30] for t in trades)
