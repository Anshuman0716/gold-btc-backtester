"""
Abstract DataSource interface.

Every data source (yfinance, Zerodha Kite, CSV files, etc.) must implement
this interface. The engine, strategy, and dashboard layers depend ONLY on
this contract — never on a concrete source.

Contract:
    fetch_ohlcv(symbol, interval, start, end) → pd.DataFrame
        - Columns: [open, high, low, close, volume] (lowercase)
        - Index: DatetimeIndex (timezone-naive, UTC-normalized)
        - No NaN rows in OHLC columns
        - Sorted ascending by timestamp
"""

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class DataFetchError(Exception):
    """Raised when the underlying data source fails to return data.

    This is distinct from ValueError (invalid inputs) — DataFetchError means
    the inputs were valid but the source returned empty/corrupt data or the
    API call itself failed.
    """
    pass


class DataSource(ABC):
    """Abstract base class for all OHLCV data sources.

    Subclasses must implement fetch_ohlcv(). The returned DataFrame must
    satisfy the column/index contract documented above. The base class
    provides _validate_output() to enforce this.
    """

    REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a given symbol and time range.

        Args:
            symbol: Ticker symbol (e.g. "BTC-USD", "GC=F").
            interval: Bar interval (e.g. "1m", "1h", "1d").
            start: Start of the date range (inclusive).
            end: End of the date range (inclusive).

        Returns:
            pd.DataFrame with columns [open, high, low, close, volume]
            and a DatetimeIndex sorted ascending.

        Raises:
            ValueError: If inputs are invalid (bad symbol, interval, dates).
            DataFetchError: If the source returns empty/corrupt data.
        """
        pass

    def _validate_output(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and normalize the output DataFrame.

        Enforces the contract: correct columns, DatetimeIndex, no OHLC NaNs,
        sorted ascending. Called by concrete implementations after fetching.

        Args:
            df: Raw DataFrame from the data source.
            symbol: Ticker symbol (for error messages).

        Returns:
            Validated and cleaned DataFrame.

        Raises:
            DataFetchError: If DataFrame is empty or missing required columns.
        """
        if df is None or df.empty:
            raise DataFetchError(
                f"No data returned for {symbol}. "
                "Check that the symbol is valid and the date range has trading data."
            )

        # Normalize column names to lowercase
        df.columns = [col.lower() for col in df.columns]

        # Check required columns exist
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise DataFetchError(
                f"Missing required columns for {symbol}: {missing}. "
                f"Got columns: {list(df.columns)}"
            )

        # Keep only required columns (drop extras like 'adj close')
        df = df[self.REQUIRED_COLUMNS].copy()

        # Drop rows where any OHLC value is NaN (volume NaN is acceptable)
        ohlc_cols = ["open", "high", "low", "close"]
        df = df.dropna(subset=ohlc_cols)

        if df.empty:
            raise DataFetchError(
                f"All rows for {symbol} had NaN in OHLC columns after cleaning."
            )

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataFetchError(
                f"Expected DatetimeIndex for {symbol}, got {type(df.index).__name__}."
            )

        # Sort ascending by timestamp
        df = df.sort_index()

        return df
