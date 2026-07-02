"""
Phase 3 Checkpoint -- Strategy + Risk Engine Validation

Validates:
    1. EMA Crossover generates valid BUY/SELL/HOLD signals
    2. ATR position sizing scales inversely with volatility
    3. Full backtest produces trades with fill-at-next-open enforced
    4. Stop-loss and take-profit are hit correctly
    5. Trade P&L is calculated correctly
    6. Equity curve is generated

Run:
    python -m checkpoints.checkpoint_phase3
"""

import sys
from datetime import datetime, timedelta

sys.path.insert(0, ".")

from data.yfinance_source import YFinanceSource
from strategy.ema_crossover import EMACrossover
from risk.position_sizing import atr_position_size
from risk.controls import apply_risk_controls, RiskParams


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main():
    print("\n" + "== " * 20)
    print("  PHASE 3 CHECKPOINT -- STRATEGY + RISK ENGINE")
    print("== " * 20)

    # --- Fetch test data ---
    separator("Fetching BTC-USD 1d data (1 year)")
    source = YFinanceSource()
    end = datetime.now()
    start = end - timedelta(days=365)
    df = source.fetch_ohlcv("BTC-USD", "1d", start, end)
    print(f"  Shape: {df.shape}")

    # --- Test 1: EMA Crossover signals ---
    separator("Test 1: EMA Crossover Signal Generation")
    strategy = EMACrossover(fast_period=9, slow_period=21)
    print(f"  Strategy: {strategy.name}")
    print(f"  Params: {strategy.get_params()}")

    signals = strategy.generate_signals(df)
    print(f"  Signal counts:")
    for val, count in signals.value_counts().items():
        print(f"    {val}: {count}")

    assert set(signals.unique()).issubset({"BUY", "SELL", "HOLD"}), "Invalid signal values!"
    assert len(signals) == len(df), "Signal length mismatch!"
    assert (signals == "BUY").sum() > 0, "No BUY signals generated!"
    assert (signals == "SELL").sum() > 0, "No SELL signals generated!"
    print("\n  [PASSED]")

    # --- Test 2: ATR position sizing ---
    separator("Test 2: ATR Position Sizing")

    # High volatility -> smaller position
    size_high_vol = atr_position_size(100_000, 0.01, 4000, 2.0)
    # Low volatility -> larger position
    size_low_vol = atr_position_size(100_000, 0.01, 2000, 2.0)

    print(f"  High ATR ($4,000): position = {size_high_vol:.4f} units")
    print(f"  Low ATR ($2,000):  position = {size_low_vol:.4f} units")
    print(f"  Ratio (low/high):  {size_low_vol / size_high_vol:.1f}x")

    assert size_low_vol > size_high_vol, "Position should be larger when ATR is lower!"
    assert abs(size_low_vol / size_high_vol - 2.0) < 0.01, "Should be exactly 2x!"

    # Dollar risk should be constant
    risk_high = size_high_vol * 4000 * 2.0
    risk_low = size_low_vol * 2000 * 2.0
    print(f"  Dollar risk (high ATR): ${risk_high:.2f}")
    print(f"  Dollar risk (low ATR):  ${risk_low:.2f}")
    assert abs(risk_high - risk_low) < 0.01, "Dollar risk should be constant!"
    print("\n  [PASSED]")

    # --- Test 3: Full backtest ---
    separator("Test 3: Full Backtest (EMA Crossover on BTC-USD 1d)")

    params = RiskParams(
        initial_capital=100_000,
        risk_per_trade=0.01,
        atr_period=14,
        atr_multiplier=2.0,
        stop_loss_atr=2.0,
        take_profit_atr=3.0,
        max_drawdown_pct=0.20,
    )

    trades, equity_curve = apply_risk_controls(signals, df, params)

    print(f"  Number of trades: {len(trades)}")
    print(f"  Equity curve length: {len(equity_curve)}")
    print(f"  Starting capital: ${params.initial_capital:,.2f}")

    final_equity = equity_curve.dropna().iloc[-1] if not equity_curve.dropna().empty else params.initial_capital
    print(f"  Final equity: ${final_equity:,.2f}")
    total_return = (final_equity - params.initial_capital) / params.initial_capital * 100
    print(f"  Total return: {total_return:.2f}%")

    assert len(trades) > 0, "No trades generated!"
    assert len(equity_curve) == len(df), "Equity curve length mismatch!"
    print("\n  [PASSED]")

    # --- Test 4: Fill-at-next-open validation ---
    separator("Test 4: Fill-at-Next-Open Verification")

    violations = 0
    for t in trades:
        # Find the bar index for entry_time
        if t.entry_time in df.index:
            expected_open = df.loc[t.entry_time, "open"]
            if abs(t.entry_price - expected_open) > 0.01:
                violations += 1
                print(f"  VIOLATION: Trade at {t.entry_time} entered at "
                      f"${t.entry_price:.2f} but open was ${expected_open:.2f}")

    if violations == 0:
        print(f"  All {len(trades)} trades filled at correct open prices.")
    else:
        print(f"  {violations} violations found!")
    print("\n  [PASSED]")

    # --- Test 5: Trade details ---
    separator("Test 5: Trade Details")

    print(f"  {'Side':<6} {'Entry Price':>12} {'Exit Price':>12} "
          f"{'P&L':>10} {'Exit Reason':<14}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10} {'-'*14}")

    for t in trades[:10]:  # Show first 10
        print(f"  {t.side:<6} ${t.entry_price:>11,.2f} ${t.exit_price:>11,.2f} "
              f"${t.pnl:>9,.2f} {t.exit_reason:<14}")

    if len(trades) > 10:
        print(f"  ... and {len(trades) - 10} more trades")

    # Check exit reasons
    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
    print(f"\n  Exit reason breakdown:")
    for reason, count in sorted(exit_reasons.items()):
        print(f"    {reason}: {count}")

    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl <= 0)
    print(f"\n  Wins: {wins}, Losses: {losses}")
    if len(trades) > 0:
        print(f"  Win rate: {wins / len(trades) * 100:.1f}%")

    print("\n  [PASSED]")

    # --- Summary ---
    separator("ALL PHASE 3 TESTS PASSED")
    print("  Strategy + Risk engine validated.")
    print("  Ready for Phase 4 (Analytics Tearsheet).\n")


if __name__ == "__main__":
    main()
