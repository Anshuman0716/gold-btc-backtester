"""
Parquet-based caching layer.

Two components:
    1. ParquetCache — low-level get/put for Parquet files.
    2. CachedDataSource — decorator that wraps any DataSource with caching.

Design: Decorator pattern (composition over inheritance). CachedDataSource
takes any DataSource and adds transparent caching. The engine/strategy layers
don't know whether they're using a cached or uncached source — same interface.

Cache key format: {symbol}_{interval}_{start}_{end}.parquet
Cache directory: ./cache/ (gitignored)
"""

import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from data.base import DataSource


class ParquetCache:
    """Low-level Parquet file cache.

    Stores and retrieves DataFrames as Parquet files in a local directory.
    Parquet is used over CSV because it:
        - Preserves dtypes (no re-parsing needed)
        - Is column-oriented and compressed (smaller files)
        - Preserves DatetimeIndex natively

    Usage:
        cache = ParquetCache("./cache")
        cache.put("btc_1h_2024", df)
        df = cache.get("btc_1h_2024")  # Returns DataFrame or None
    """

    def __init__(self, cache_dir: str = "./cache") -> None:
        """Initialize the cache directory.

        Args:
            cache_dir: Path to the cache directory. Created if it doesn't exist.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        """Convert a cache key to a filesystem path.

        Uses MD5 hash of the key to avoid filesystem issues with special
        characters in ticker symbols (e.g. "GC=F").
        """
        safe_name = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_name}.parquet"

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Retrieve a DataFrame from the cache.

        Args:
            key: Cache key string.

        Returns:
            The cached DataFrame, or None if not found.
        """
        path = self._key_to_path(key)
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                # Corrupt cache file — delete and return None
                path.unlink(missing_ok=True)
                return None
        return None

    def put(self, key: str, df: pd.DataFrame) -> None:
        """Store a DataFrame in the cache.

        Args:
            key: Cache key string.
            df: DataFrame to cache.
        """
        path = self._key_to_path(key)
        df.to_parquet(path)

    def clear(self) -> None:
        """Delete all cached files."""
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()


class CachedDataSource(DataSource):
    """Decorator that adds Parquet caching to any DataSource.

    Wraps an existing DataSource (e.g. YFinanceSource). On fetch_ohlcv():
        1. Check cache for existing data → return if found.
        2. If not cached, delegate to the wrapped source.
        3. Cache the result before returning.

    The engine/strategy layers use this identically to any other DataSource.

    Usage:
        raw_source = YFinanceSource()
        source = CachedDataSource(raw_source)
        df = source.fetch_ohlcv(...)  # First call hits API, second hits cache
    """

    def __init__(
        self,
        source: DataSource,
        cache_dir: str = "./cache",
    ) -> None:
        """Initialize with a wrapped DataSource and cache directory.

        Args:
            source: The underlying DataSource to wrap.
            cache_dir: Path to the cache directory.
        """
        self._source = source
        self._cache = ParquetCache(cache_dir)

    def _build_cache_key(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> str:
        """Build a unique cache key from fetch parameters."""
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        return f"{symbol}_{interval}_{start_str}_{end_str}"

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV data, using cache if available.

        Args:
            symbol: Ticker symbol.
            interval: Bar interval.
            start: Start of the date range.
            end: End of the date range.

        Returns:
            OHLCV DataFrame (from cache or fresh from the source).

        Raises:
            ValueError: If inputs are invalid.
            DataFetchError: If the underlying source fails.
        """
        key = self._build_cache_key(symbol, interval, start, end)

        # Try cache first
        cached_df = self._cache.get(key)
        if cached_df is not None:
            return cached_df

        # Cache miss — fetch from source
        df = self._source.fetch_ohlcv(symbol, interval, start, end)

        # Cache the result
        self._cache.put(key, df)

        return df
