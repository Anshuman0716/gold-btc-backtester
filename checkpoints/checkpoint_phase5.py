"""
Phase 5 Checkpoint -- Audit Logging Validation

Validates:
    1. Database initializes and table exists
    2. Three backtests with different params are recorded
    3. All 3 runs queryable from SQLite
    4. Run IDs are unique UUIDs
    5. Strategy params stored as valid JSON
    6. get_all_runs() returns correct DataFrame

Run:
    python -m checkpoints.checkpoint_phase5
"""

import sys
import os
import json
from datetime import datetime, timedelta

sys.path.insert(0, ".")

from data.yfinance_source import YFinanceSource
from strategy.ema_crossover import EMACrossover
from risk.controls import apply_risk_controls, RiskParams
from analytics.tearsheet import compute_tearsheet
from logging_.db import init_db
from logging_.recorder import record_run, get_all_runs


TEST_DB = "test_backtest_log.db"


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def run_backtest(df, fast, slow):
    """Helper: run a single backtest with given EMA params."""
    strategy = EMACrossover(fast_period=fast, slow_period=slow)
    signals = strategy.generate_signals(df)
    params = RiskParams(initial_capital=100_000)
    trades, equity = apply_risk_controls(signals, df, params)
    tearsheet = compute_tearsheet(trades, equity, 100_000, "BTC-USD", "1d")
    return strategy, tearsheet


def main():
    print("\n" + "== " * 20)
    print("  PHASE 5 CHECKPOINT -- AUDIT LOGGING")
    print("== " * 20)

    # Clean up test DB from previous runs
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

    # --- Fetch data once ---
    separator("Fetching BTC-USD 1d data")
    source = YFinanceSource()
    end = datetime.now()
    start = end - timedelta(days=365)
    df = source.fetch_ohlcv("BTC-USD", "1d", start, end)
    print(f"  Shape: {df.shape}")

    # --- Test 1: Database initialization ---
    separator("Test 1: Database Initialization")
    conn = init_db(TEST_DB)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='backtest_runs'"
    )
    table = cursor.fetchone()
    conn.close()
    assert table is not None, "backtest_runs table not created!"
    print(f"  Table 'backtest_runs' exists: OK")
    print("\n  [PASSED]")

    # --- Test 2: Record 3 backtests with different params ---
    separator("Test 2: Record 3 Backtests")

    configs = [
        (5, 15),   # Fast EMA crossover
        (9, 21),   # Default
        (12, 50),  # Slow EMA crossover
    ]

    run_ids = []
    for fast, slow in configs:
        strategy, tearsheet = run_backtest(df, fast, slow)
        run_id = record_run(
            asset="BTC-USD",
            interval="1d",
            start_date=start,
            end_date=end,
            strategy_name=strategy.name,
            strategy_params=strategy.get_params(),
            tearsheet=tearsheet,
            db_path=TEST_DB,
        )
        run_ids.append(run_id)
        print(f"  EMA({fast}/{slow}): run_id={run_id[:8]}... "
              f"return={tearsheet.total_return*100:.2f}% "
              f"sharpe={tearsheet.sharpe_ratio:.2f} "
              f"trades={tearsheet.num_trades}")

    assert len(run_ids) == 3, f"Expected 3 runs, got {len(run_ids)}"
    print("\n  [PASSED]")

    # --- Test 3: Query all runs ---
    separator("Test 3: Query All Runs")
    runs_df = get_all_runs(db_path=TEST_DB)
    print(f"  Rows returned: {len(runs_df)}")
    print(f"  Columns: {list(runs_df.columns)}")
    assert len(runs_df) == 3, f"Expected 3 rows, got {len(runs_df)}"
    print("\n  [PASSED]")

    # --- Test 4: Unique UUIDs ---
    separator("Test 4: Unique Run IDs")
    unique_ids = set(runs_df["run_id"])
    print(f"  Unique IDs: {len(unique_ids)}")
    assert len(unique_ids) == 3, "Run IDs are not unique!"

    # Validate UUID format
    import uuid
    for rid in unique_ids:
        try:
            uuid.UUID(rid)
            print(f"    {rid[:8]}... valid UUID")
        except ValueError:
            print(f"    {rid} -- INVALID UUID!")
            sys.exit(1)
    print("\n  [PASSED]")

    # --- Test 5: JSON params ---
    separator("Test 5: Strategy Params as JSON")
    for _, row in runs_df.iterrows():
        params = json.loads(row["strategy_params"])
        assert isinstance(params, dict), "Params should be a dict!"
        assert "fast_period" in params, "Missing fast_period!"
        assert "slow_period" in params, "Missing slow_period!"
        print(f"    {row['run_id'][:8]}... params={params}")
    print("\n  [PASSED]")

    # --- Test 6: Data integrity ---
    separator("Test 6: Data Integrity")
    for _, row in runs_df.iterrows():
        assert row["asset"] == "BTC-USD", f"Wrong asset: {row['asset']}"
        assert row["interval"] == "1d", f"Wrong interval: {row['interval']}"
        assert -1 <= row["total_return"] <= 10, f"Return out of range: {row['total_return']}"
        assert row["num_trades"] >= 0, f"Negative trades: {row['num_trades']}"

    print(f"  All 3 runs have valid asset, interval, return, and trade count.")
    print("\n  [PASSED]")

    # --- Cleanup ---
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
        print(f"  Cleaned up {TEST_DB}")

    # --- Summary ---
    separator("ALL PHASE 5 TESTS PASSED")
    print("  Audit logging validated. Every run is reproducible.")
    print("  Ready for Phase 6 (Dashboard -- pending design brief).\n")


if __name__ == "__main__":
    main()
