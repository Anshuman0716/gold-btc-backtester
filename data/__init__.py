"""
Data layer — abstract DataSource interface, yfinance implementation,
interval validation, and Parquet-based caching.
"""

from data.base import DataSource, DataFetchError
from data.yfinance_source import YFinanceSource
from data.interval_limits import INTERVAL_LIMITS, VALID_INTERVALS, validate_date_range, get_max_start_date
from data.cache import ParquetCache, CachedDataSource

__all__ = [
    "DataSource",
    "DataFetchError",
    "YFinanceSource",
    "INTERVAL_LIMITS",
    "VALID_INTERVALS",
    "validate_date_range",
    "get_max_start_date",
    "ParquetCache",
    "CachedDataSource",
]
