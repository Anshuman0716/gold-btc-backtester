"""
Volatility indicators — Bollinger Bands, ATR.

ATR is also used by the risk layer (position sizing and stop-loss).
Having it in the indicators layer keeps it reusable across the codebase.

Convention:
    - Input: pd.DataFrame with columns [open, high, low, close, volume]
    - Output: pd.Series or pd.DataFrame (documented per function)
"""

import pandas as pd


def bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands.

    Bollinger Bands consist of a middle band (SMA) with upper and lower
    bands at a configurable number of standard deviations.

    Interpretation:
        - Price near upper band: potentially overbought
        - Price near lower band: potentially oversold
        - Band squeeze (narrow bands): low volatility, breakout likely
        - Band expansion: high volatility

    Calculation:
        1. Middle = SMA(close, period)
        2. Std = rolling standard deviation of close over period
        3. Upper = Middle + std_dev * Std
        4. Lower = Middle - std_dev * Std

    Args:
        df: OHLCV DataFrame.
        period: SMA lookback period (default 20).
        std_dev: Number of standard deviations (default 2.0).

    Returns:
        pd.DataFrame with columns:
            - 'upper': Upper Bollinger Band
            - 'middle': Middle band (SMA)
            - 'lower': Lower Bollinger Band
    """
    if period < 1:
        raise ValueError(f"Bollinger period must be >= 1, got {period}")
    if std_dev <= 0:
        raise ValueError(f"std_dev must be > 0, got {std_dev}")

    close = df["close"]

    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()

    upper = middle + std_dev * std
    lower = middle - std_dev * std

    return pd.DataFrame({
        "upper": upper,
        "middle": middle,
        "lower": lower,
    }, index=df.index)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (ATR).

    ATR measures market volatility. Higher ATR = more volatile market.
    Used by:
        - Risk layer: ATR-based position sizing and stop-loss
        - Supertrend indicator: band width calculation

    Calculation (Wilder's method):
        1. True Range = max of:
            a. high - low (current bar's range)
            b. |high - prev_close| (gap up)
            c. |low - prev_close| (gap down)
        2. ATR = Wilder's EMA of True Range (alpha = 1/period)

    Args:
        df: OHLCV DataFrame.
        period: Smoothing period (default 14).

    Returns:
        pd.Series with ATR values.
    """
    if period < 1:
        raise ValueError(f"ATR period must be >= 1, got {period}")

    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range: max of three components
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    # Wilder's smoothing
    atr_series = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    return atr_series
