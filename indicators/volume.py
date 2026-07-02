"""
Volume indicators — VWAP.

VWAP is primarily an intraday indicator. For daily bars, it represents
the volume-weighted average price for that single bar (less useful).
For intraday bars (1m, 5m, 15m, 30m, 1h), it accumulates within each
trading session.

Convention:
    - Input: pd.DataFrame with columns [open, high, low, close, volume]
    - Output: pd.Series
"""

import numpy as np
import pandas as pd


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price (VWAP).

    VWAP is the ratio of cumulative (price * volume) to cumulative volume,
    reset at each trading day boundary.

    Interpretation:
        - Price above VWAP: buyers are in control (bullish)
        - Price below VWAP: sellers are in control (bearish)
        - Institutional traders use VWAP as a benchmark for execution quality

    Calculation:
        1. Typical Price = (high + low + close) / 3
        2. For each trading day:
            VWAP = cumsum(TP * volume) / cumsum(volume)

    Note: For daily bars, VWAP equals the typical price of each bar
    since there's no intraday accumulation.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.

    Returns:
        pd.Series with VWAP values.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_volume = typical_price * df["volume"]

    # Group by date for intraday reset
    # For daily data, each bar is its own group (VWAP = typical price)
    if hasattr(df.index, "date"):
        date_groups = df.index.date
    else:
        # Fallback: no grouping (cumulative VWAP)
        date_groups = np.zeros(len(df))

    cum_tp_volume = tp_volume.groupby(date_groups).cumsum()
    cum_volume = df["volume"].groupby(date_groups).cumsum()

    # Avoid division by zero (bars with zero volume)
    vwap_series = cum_tp_volume / cum_volume.replace(0, float("nan"))

    return vwap_series
