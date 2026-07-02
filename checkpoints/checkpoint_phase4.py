"""
Phase 4 Checkpoint -- Analytics Tearsheet Validation

Validates:
    1. Full tearsheet computed from Phase 3 backtest
    2. All metric values are sane (finite, correct ranges)
    3. Drawdown is negative (or zero)
    4. Win rate matches manual count
    5. Profit factor is consistent with trade P&Ls
    6. to_dict() produces formatted output

Run:
    python -m checkpoints.checkpoint_phase4
"""

import sys
from datetime import datetime, timedelta

sys.path.insert(0, ".")

from data.yfinance_source import YFinanceSource
from strategy.ema_crossover import EMACrossover
from risk.controls import apply_risk_controls, RiskParams
from analytics.tearsheet import compute_tearsheet, get_annualization_factor
import numpy as np


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main():
    print("\n" + "== " * 20)
    print("  PHASE 4 CHECKPOINT -- ANALYTICS TEARSHEET")
    print("== " * 20)

    # --- Run a full backtest (reuse Phase 3 pipeline) ---
    separator("Running BTC-USD 1d Backtest")
    source = YFinanceSource()
    end = datetime.now()
    start = end - timedelta(days=365)
    df = source.fetch_ohlcv("BTC-USD", "1d", start, end)

    strategy = EMACrossover(fast_period=9, slow_period=21)
    signals = strategy.generate_signals(df)

    params = RiskParams(
        initial_capital=100_000,
        risk_per_trade=0.01,
        atr_period=14,
        stop_loss_atr=2.0,
        take_profit_atr=3.0,
        max_drawdown_pct=0.20,
    )

    trades, equity_curve = apply_risk_controls(signals, df, params)
    print(f"  Trades: {len(trades)}, Equity points: {len(equity_curve)}")

    # --- Test 1: Compute tearsheet ---
    separator("Test 1: Compute Tearsheet")
    result = compute_tearsheet(
        trades=trades,
        equity_curve=equity_curve,
        initial_capital=params.initial_capital,
        asset="BTC-USD",
        interval="1d",
    )

    print(f"  Total Return:       {result.total_return * 100:.2f}%")
    print(f"  Sharpe Ratio:       {result.sharpe_ratio:.4f}")
    print(f"  Sortino Ratio:      {result.sortino_ratio:.4f}")
    print(f"  Calmar Ratio:       {result.calmar_ratio:.4f}")
    print(f"  Max Drawdown:       {result.max_drawdown * 100:.2f}%")
    print(f"  Max DD Duration:    {result.max_drawdown_duration} bars")
    print(f"  Win Rate:           {result.win_rate * 100:.1f}%")
    print(f"  Profit Factor:      {result.profit_factor:.4f}")
    print(f"  Avg Trade Return:   {result.avg_trade_return * 100:.4f}%")
    print(f"  Num Trades:         {result.num_trades}")

    assert result.num_trades == len(trades), "Trade count mismatch!"
    print("\n  [PASSED]")

    # --- Test 2: Metric sanity checks ---
    separator("Test 2: Metric Sanity Checks")

    # Sharpe should be finite
    assert np.isfinite(result.sharpe_ratio), f"Sharpe is not finite: {result.sharpe_ratio}"
    print(f"  Sharpe is finite: OK")

    # Max drawdown should be <= 0
    assert result.max_drawdown <= 0, f"Max DD should be <= 0, got {result.max_drawdown}"
    print(f"  Max drawdown <= 0: OK ({result.max_drawdown * 100:.2f}%)")

    # Max DD duration should be >= 0
    assert result.max_drawdown_duration >= 0, f"DD duration < 0: {result.max_drawdown_duration}"
    print(f"  Max DD duration >= 0: OK ({result.max_drawdown_duration} bars)")

    # Win rate should be 0-1
    assert 0 <= result.win_rate <= 1, f"Win rate out of range: {result.win_rate}"
    print(f"  Win rate in [0, 1]: OK ({result.win_rate:.2f})")

    # Profit factor should be >= 0
    assert result.profit_factor >= 0, f"Profit factor < 0: {result.profit_factor}"
    print(f"  Profit factor >= 0: OK ({result.profit_factor:.4f})")

    print("\n  [PASSED]")

    # --- Test 3: Win rate manual verification ---
    separator("Test 3: Win Rate Verification")
    manual_wins = sum(1 for t in trades if t.pnl > 0)
    manual_win_rate = manual_wins / len(trades) if trades else 0
    print(f"  Manual wins: {manual_wins}/{len(trades)} = {manual_win_rate:.4f}")
    print(f"  Tearsheet:   {result.win_rate:.4f}")
    assert abs(result.win_rate - manual_win_rate) < 0.001, "Win rate mismatch!"
    print("\n  [PASSED]")

    # --- Test 4: Profit factor verification ---
    separator("Test 4: Profit Factor Verification")
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    manual_pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    print(f"  Gross profit: ${gross_profit:,.2f}")
    print(f"  Gross loss:   ${gross_loss:,.2f}")
    print(f"  Manual PF:    {manual_pf:.4f}")
    print(f"  Tearsheet PF: {result.profit_factor:.4f}")
    assert abs(result.profit_factor - manual_pf) < 0.001, "Profit factor mismatch!"
    print("\n  [PASSED]")

    # --- Test 5: Annualization factor ---
    separator("Test 5: Annualization Factor")
    btc_1d = get_annualization_factor("BTC-USD", "1d")
    btc_1h = get_annualization_factor("BTC-USD", "1h")
    gold_1d = get_annualization_factor("GC=F", "1d")
    print(f"  BTC-USD 1d: sqrt(365) = {btc_1d:.4f}")
    print(f"  BTC-USD 1h: sqrt(8760) = {btc_1h:.4f}")
    print(f"  GC=F   1d: sqrt(252) = {gold_1d:.4f}")
    assert abs(btc_1d - np.sqrt(365)) < 0.01, "BTC 1d annualization wrong!"
    assert abs(gold_1d - np.sqrt(252)) < 0.01, "Gold 1d annualization wrong!"
    print("\n  [PASSED]")

    # --- Test 6: to_dict() ---
    separator("Test 6: Formatted Output (to_dict)")
    formatted = result.to_dict()
    print(f"  Tearsheet Summary:")
    for key, value in formatted.items():
        print(f"    {key:<20} {value}")
    assert isinstance(formatted, dict), "to_dict should return dict!"
    assert len(formatted) == 10, f"Expected 10 metrics, got {len(formatted)}"
    print("\n  [PASSED]")

    # --- Summary ---
    separator("ALL PHASE 4 TESTS PASSED")
    print("  Analytics tearsheet validated.")
    print("  Ready for Phase 5 (Audit Logging).\n")


if __name__ == "__main__":
    main()
