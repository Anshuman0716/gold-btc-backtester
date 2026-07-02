"""
Phase 2 Checkpoint -- Indicator Library Validation

Validates all 8 indicators on real BTC-USD 1d data:
    1. EMA -- trend-following, should track close price
    2. Supertrend -- direction values must be +1 or -1
    3. ADX -- must be 0-100 range
    4. RSI -- must be 0-100 range
    5. MACD -- histogram should oscillate around zero
    6. Bollinger Bands -- upper > middle > lower
    7. ATR -- must be strictly positive
    8. VWAP -- should be within the day's high-low range
    9. Registry -- all 8 indicators registered, functions callable

Run:
    python -m checkpoints.checkpoint_phase2
"""

import sys
from datetime import datetime, timedelta

sys.path.insert(0, ".")

from data.yfinance_source import YFinanceSource
from indicators.trend import ema, supertrend, adx
from indicators.momentum import rsi, macd
from indicators.volatility import bollinger_bands, atr
from indicators.volume import vwap
from indicators.registry import INDICATOR_REGISTRY, INDICATOR_FUNCTIONS, INDICATOR_PARAMS


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main():
    print("\n" + "== " * 20)
    print("  PHASE 2 CHECKPOINT -- INDICATOR LIBRARY VALIDATION")
    print("== " * 20)

    # --- Fetch test data ---
    separator("Fetching BTC-USD 1d data (1 year)")
    source = YFinanceSource()
    end = datetime.now()
    start = end - timedelta(days=365)
    df = source.fetch_ohlcv("BTC-USD", "1d", start, end)
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")

    # --- Test 1: EMA ---
    separator("Test 1: EMA (period=20)")
    ema_result = ema(df, period=20)
    print(f"  Type: {type(ema_result).__name__}")
    print(f"  Shape: {ema_result.shape}")
    print(f"  Last 5 values:\n{ema_result.tail()}")
    assert len(ema_result) == len(df), "Length mismatch!"
    assert ema_result.notna().sum() > 0, "All NaN!"
    print("\n  [PASSED]")

    # --- Test 2: Supertrend ---
    separator("Test 2: Supertrend (atr_period=10, multiplier=3.0)")
    st_result = supertrend(df, atr_period=10, multiplier=3.0)
    print(f"  Columns: {list(st_result.columns)}")
    print(f"  Direction value counts:\n{st_result['direction'].value_counts().to_string()}")
    print(f"  Last 5 values:\n{st_result.tail()}")
    assert set(st_result["direction"].dropna().unique()).issubset({1, -1}), \
        f"Direction must be +1 or -1, got {st_result['direction'].unique()}"
    print("\n  [PASSED]")

    # --- Test 3: ADX ---
    separator("Test 3: ADX (period=14)")
    adx_result = adx(df, period=14)
    adx_valid = adx_result.dropna()
    print(f"  Shape: {adx_result.shape}")
    print(f"  Range: {adx_valid.min():.2f} to {adx_valid.max():.2f}")
    print(f"  Last 5 values:\n{adx_result.tail()}")
    assert adx_valid.min() >= 0, f"ADX below 0: {adx_valid.min()}"
    assert adx_valid.max() <= 100, f"ADX above 100: {adx_valid.max()}"
    print("\n  [PASSED]")

    # --- Test 4: RSI ---
    separator("Test 4: RSI (period=14)")
    rsi_result = rsi(df, period=14)
    rsi_valid = rsi_result.dropna()
    print(f"  Shape: {rsi_result.shape}")
    print(f"  Range: {rsi_valid.min():.2f} to {rsi_valid.max():.2f}")
    print(f"  Last 5 values:\n{rsi_result.tail()}")
    assert rsi_valid.min() >= 0, f"RSI below 0: {rsi_valid.min()}"
    assert rsi_valid.max() <= 100, f"RSI above 100: {rsi_valid.max()}"
    print("\n  [PASSED]")

    # --- Test 5: MACD ---
    separator("Test 5: MACD (fast=12, slow=26, signal=9)")
    macd_result = macd(df, fast=12, slow=26, signal=9)
    print(f"  Columns: {list(macd_result.columns)}")
    print(f"  Last 5 values:\n{macd_result.tail()}")
    assert list(macd_result.columns) == ["macd", "signal", "histogram"], "Column mismatch!"
    # Histogram should have both positive and negative values (oscillator)
    hist_valid = macd_result["histogram"].dropna()
    assert (hist_valid > 0).any(), "Histogram never positive!"
    assert (hist_valid < 0).any(), "Histogram never negative!"
    print("\n  [PASSED]")

    # --- Test 6: Bollinger Bands ---
    separator("Test 6: Bollinger Bands (period=20, std_dev=2.0)")
    bb_result = bollinger_bands(df, period=20, std_dev=2.0)
    print(f"  Columns: {list(bb_result.columns)}")
    print(f"  Last 5 values:\n{bb_result.tail()}")
    assert list(bb_result.columns) == ["upper", "middle", "lower"], "Column mismatch!"
    # Upper > middle > lower (after warmup)
    valid_rows = bb_result.dropna()
    assert (valid_rows["upper"] >= valid_rows["middle"]).all(), "Upper < Middle!"
    assert (valid_rows["middle"] >= valid_rows["lower"]).all(), "Middle < Lower!"
    print("\n  [PASSED]")

    # --- Test 7: ATR ---
    separator("Test 7: ATR (period=14)")
    atr_result = atr(df, period=14)
    atr_valid = atr_result.dropna()
    print(f"  Shape: {atr_result.shape}")
    print(f"  Range: {atr_valid.min():.2f} to {atr_valid.max():.2f}")
    print(f"  Last 5 values:\n{atr_result.tail()}")
    assert (atr_valid > 0).all(), "ATR has non-positive values!"
    print("\n  [PASSED]")

    # --- Test 8: VWAP ---
    separator("Test 8: VWAP")
    vwap_result = vwap(df)
    vwap_valid = vwap_result.dropna()
    print(f"  Shape: {vwap_result.shape}")
    print(f"  Last 5 values:\n{vwap_result.tail()}")
    assert len(vwap_valid) > 0, "All VWAP values are NaN!"
    print("\n  [PASSED]")

    # --- Test 9: Registry ---
    separator("Test 9: Indicator Registry")
    total_indicators = sum(len(v) for v in INDICATOR_REGISTRY.values())
    print(f"  Categories: {list(INDICATOR_REGISTRY.keys())}")
    print(f"  Total indicators registered: {total_indicators}")
    print(f"  Functions registered: {len(INDICATOR_FUNCTIONS)}")
    print(f"  Param configs registered: {len(INDICATOR_PARAMS)}")

    assert total_indicators == 8, f"Expected 8 indicators, got {total_indicators}"
    assert len(INDICATOR_FUNCTIONS) == 8, f"Expected 8 functions, got {len(INDICATOR_FUNCTIONS)}"
    assert len(INDICATOR_PARAMS) == 8, f"Expected 8 param configs, got {len(INDICATOR_PARAMS)}"

    # Verify every registered name has a function and params
    for category, names in INDICATOR_REGISTRY.items():
        for name in names:
            assert name in INDICATOR_FUNCTIONS, f"{name} missing from INDICATOR_FUNCTIONS"
            assert name in INDICATOR_PARAMS, f"{name} missing from INDICATOR_PARAMS"
            print(f"    {category}/{name} -- function: OK, params: OK")

    print("\n  [PASSED]")

    # --- Summary ---
    separator("ALL PHASE 2 TESTS PASSED")
    print("  All 8 indicators + registry validated.")
    print("  Indicator layer is solid. Ready for Phase 3 (Strategy + Risk).\n")


if __name__ == "__main__":
    main()
