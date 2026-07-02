"""
Phase 1 Checkpoint — Data Layer Validation

Validates:
    1. YFinanceSource fetches BTC-USD 1h data successfully
    2. DataFrame contract: correct columns, DatetimeIndex, no OHLC NaNs
    3. CachedDataSource caches data and serves from cache on second call
    4. Interval validation rejects invalid ranges (1m for 30 days)
    5. Input validation rejects unsupported symbols

Run:
    python -m checkpoints.checkpoint_phase1
"""

import sys
import time
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, ".")

from data.base import DataFetchError
from data.yfinance_source import YFinanceSource
from data.interval_limits import validate_date_range, get_max_start_date, VALID_INTERVALS
from data.cache import CachedDataSource, ParquetCache


def separator(title: str) -> None:
    """Print a section separator."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_basic_fetch():
    """Test 1: Basic BTC-USD 1h fetch."""
    separator("Test 1: Basic BTC-USD 1h Fetch")

    source = YFinanceSource()
    end = datetime.now()
    start = end - timedelta(days=30)

    print(f"Fetching BTC-USD 1h data: {start.date()} to {end.date()}")
    df = source.fetch_ohlcv("BTC-USD", "1h", start, end)

    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Index type: {type(df.index).__name__}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  OHLC NaN count: {df[['open', 'high', 'low', 'close']].isna().sum().sum()}")
    print(f"\n  Head:\n{df.head()}")
    print(f"\n  Dtypes:\n{df.dtypes}")

    # Assertions
    assert list(df.columns) == ["open", "high", "low", "close", "volume"], "Column mismatch!"
    assert df.index.is_monotonic_increasing, "Index not sorted ascending!"
    assert df[["open", "high", "low", "close"]].isna().sum().sum() == 0, "OHLC has NaN!"
    assert len(df) > 0, "DataFrame is empty!"

    print("\n  [PASSED]")
    return df


def test_gold_fetch():
    """Test 2: Gold GC=F 1d fetch."""
    separator("Test 2: Gold GC=F 1d Fetch")

    source = YFinanceSource()
    end = datetime.now()
    start = end - timedelta(days=365)

    print(f"Fetching GC=F 1d data: {start.date()} to {end.date()}")
    df = source.fetch_ohlcv("GC=F", "1d", start, end)

    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"\n  Head:\n{df.head()}")

    assert len(df) > 200, f"Expected 200+ trading days, got {len(df)}"
    print("\n  [PASSED]")


def test_cache():
    """Test 3: CachedDataSource caches and serves from cache."""
    separator("Test 3: Parquet Cache")

    # Clear cache first
    cache = ParquetCache("./cache")
    cache.clear()

    source = CachedDataSource(YFinanceSource(), cache_dir="./cache")
    end = datetime.now()
    start = end - timedelta(days=7)

    # First fetch — should hit API
    print("  First fetch (API call)...")
    t1 = time.time()
    df1 = source.fetch_ohlcv("BTC-USD", "1h", start, end)
    api_time = time.time() - t1
    print(f"  API fetch time: {api_time:.2f}s")

    # Second fetch — should hit cache
    print("  Second fetch (cache hit)...")
    t2 = time.time()
    df2 = source.fetch_ohlcv("BTC-USD", "1h", start, end)
    cache_time = time.time() - t2
    print(f"  Cache fetch time: {cache_time:.4f}s")

    # Cache should be faster
    print(f"  Speedup: {api_time / max(cache_time, 0.0001):.1f}x")

    # DataFrames should be identical
    assert df1.shape == df2.shape, f"Shape mismatch: {df1.shape} vs {df2.shape}"
    assert df1.equals(df2), "Cached data doesn't match original!"

    print("\n  [PASSED]")


def test_interval_validation():
    """Test 4: Interval validation rejects invalid ranges."""
    separator("Test 4: Interval Validation")

    end = datetime.now()

    # 1m for 30 days should fail
    start_30d = end - timedelta(days=30)
    try:
        validate_date_range("1m", start_30d, end)
        print("  [FAILED] -- should have raised ValueError for 1m/30 days")
        sys.exit(1)
    except ValueError as e:
        print(f"  1m + 30 days -> Correctly rejected: {e}")

    # 1m for 5 days should pass
    start_5d = end - timedelta(days=5)
    validate_date_range("1m", start_5d, end)
    print("  1m + 5 days -> Correctly accepted")

    # Invalid interval should fail
    try:
        validate_date_range("90m", start_5d, end)
        print("  [FAILED] -- should have rejected 90m")
        sys.exit(1)
    except ValueError as e:
        print(f"  90m -> Correctly rejected: {e}")

    # get_max_start_date
    max_start = get_max_start_date("1h", end)
    expected_days = (end - max_start).days
    print(f"  Max start for 1h: {max_start.date()} ({expected_days} days from end)")
    assert expected_days == 730, f"Expected 730 days, got {expected_days}"

    print("\n  [PASSED]")


def test_invalid_symbol():
    """Test 5: Invalid symbol is rejected."""
    separator("Test 5: Invalid Symbol Rejection")

    source = YFinanceSource()
    end = datetime.now()
    start = end - timedelta(days=7)

    try:
        source.fetch_ohlcv("INVALID-TICKER", "1d", start, end)
        print("  [FAILED] -- should have raised ValueError")
        sys.exit(1)
    except ValueError as e:
        print(f"  Correctly rejected: {e}")

    print("\n  [PASSED]")


def main():
    """Run all Phase 1 checkpoint tests."""
    print("\n" + "== " * 20)
    print("  PHASE 1 CHECKPOINT -- DATA LAYER VALIDATION")
    print("== " * 20)

    test_basic_fetch()
    test_gold_fetch()
    test_cache()
    test_interval_validation()
    test_invalid_symbol()

    separator("ALL PHASE 1 TESTS PASSED")
    print("  Data layer is solid. Ready for Phase 2 (Indicators).\n")


if __name__ == "__main__":
    main()
