"""
Concrete yfinance DataSource implementation.

Fetches OHLCV data from Yahoo Finance via the yfinance library.
Validates interval/date range before making API calls to fail fast
with clear error messages.
"""

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from data.base import DataSource, DataFetchError
from data.interval_limits import VALID_INTERVALS, validate_date_range


# Symbols supported in v1
SUPPORTED_SYMBOLS: frozenset[str] = frozenset({"GC=F", "BTC-USD"})


class YFinanceSource(DataSource):
    """Fetches OHLCV data from Yahoo Finance.

    Usage:
        source = YFinanceSource()
        df = source.fetch_ohlcv("BTC-USD", "1h", start, end)

    The returned DataFrame has:
        - Columns: [open, high, low, close, volume]
        - Index: DatetimeIndex, sorted ascending
        - No NaN in OHLC columns
    """

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from yfinance.

        Validates inputs (symbol, interval, date range) before making the
        API call. Normalizes and cleans the output via _validate_output().

        Args:
            symbol: Ticker symbol ("BTC-USD" or "GC=F").
            interval: Bar interval (must be in VALID_INTERVALS).
            start: Start of the date range (inclusive).
            end: End of the date range (inclusive).

        Returns:
            Clean OHLCV DataFrame.

        Raises:
            ValueError: If symbol, interval, or date range is invalid.
            DataFetchError: If yfinance returns empty/corrupt data.
        """
        # --- Input validation (fail fast before API call) ---
        if symbol not in SUPPORTED_SYMBOLS:
            raise ValueError(
                f"Unsupported symbol '{symbol}'. "
                f"Supported: {sorted(SUPPORTED_SYMBOLS)}"
            )

        if interval not in VALID_INTERVALS:
            raise ValueError(
                f"Invalid interval '{interval}'. "
                f"Must be one of: {sorted(VALID_INTERVALS)}"
            )

        validate_date_range(interval, start, end)

        # --- Fetch data ---
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                interval=interval,
                start=start.strftime("%Y-%m-%d"),
                # yfinance's `end` is exclusive; add a day so the docstring's
                # "end (inclusive)" holds and the last requested bar is returned.
                end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=True,
                actions=False,
            )
        except Exception as e:
            raise DataFetchError(
                f"yfinance API call failed for {symbol} "
                f"({interval}, {start.date()} to {end.date()}): {e}"
            ) from e

        # --- Validate and normalize output ---
        df = self._validate_output(df, symbol)

        return df
