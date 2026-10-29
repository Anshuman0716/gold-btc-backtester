"""
Momentum indicators — RSI, MACD.

Hand-rolled using pandas. RSI uses Wilder's smoothing method (exponential
moving average with alpha = 1/period), which is the industry standard.

Convention:
    - Input: pd.DataFrame with columns [open, high, low, close, volume]
    - Output: pd.Series or pd.DataFrame (documented per function)
"""

import pandas as pd


def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI).

    RSI measures the speed and magnitude of price changes on a 0-100 scale.
    Interpretation:
        - RSI > 70: overbought (potential reversal down)
        - RSI < 30: oversold (potential reversal up)
        - RSI = 50: neutral

    Calculation (Wilder's method):
        1. delta = close - close.shift(1)
        2. gain = delta where delta > 0, else 0
        3. loss = -delta where delta < 0, else 0
        4. avg_gain = Wilder's EMA of gains (alpha = 1/period)
        5. avg_loss = Wilder's EMA of losses (alpha = 1/period)
        6. RS = avg_gain / avg_loss
        7. RSI = 100 - (100 / (1 + RS))

    Args:
        df: OHLCV DataFrame.
        period: Lookback period (default 14).

    Returns:
        pd.Series with RSI values (0-100 range).
    """
    if period < 1:
        raise ValueError(f"RSI period must be >= 1, got {period}")

    delta = df["close"].diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder's smoothing: EMA with alpha = 1/period
    alpha = 1 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi_series = 100 - (100 / (1 + rs))

    # When avg_loss is 0 (all gains), RSI = 100. Only that case is mapped;
    # warmup NaNs stay NaN (NaN != 0 is True, so they are left alone).
    rsi_series = rsi_series.where(avg_loss != 0, 100.0)

    return rsi_series


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence (MACD).

    MACD shows the relationship between two EMAs of price.
    Interpretation:
        - MACD line crosses above signal line: bullish
        - MACD line crosses below signal line: bearish
        - Histogram > 0: bullish momentum
        - Histogram < 0: bearish momentum

    Calculation:
        1. MACD line = EMA(fast) - EMA(slow)
        2. Signal line = EMA(MACD line, signal period)
        3. Histogram = MACD line - Signal line

    Args:
        df: OHLCV DataFrame.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal: Signal line EMA period (default 9).

    Returns:
        pd.DataFrame with columns:
            - 'macd': MACD line values
            - 'signal': Signal line values
            - 'histogram': MACD - Signal
    """
    if fast >= slow:
        raise ValueError(
            f"Fast period ({fast}) must be less than slow period ({slow})."
        )

    close = df["close"]

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }, index=df.index)
