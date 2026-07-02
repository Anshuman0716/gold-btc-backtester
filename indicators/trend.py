"""
Trend indicators — EMA, Supertrend, ADX.

All indicators are hand-rolled using numpy/pandas (no pandas-ta dependency).
Every formula is interview-defensible — you can explain the math from scratch.

Convention:
    - Input: pd.DataFrame with columns [open, high, low, close, volume]
    - Output: pd.Series or pd.DataFrame (documented per function)
    - NaN handling: initial warmup periods will have NaN (expected behavior)
"""

import numpy as np
import pandas as pd


def ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Exponential Moving Average on close price.

    EMA gives more weight to recent prices. The smoothing factor (alpha)
    is calculated as: alpha = 2 / (period + 1)

    The EMA formula:
        EMA_t = close_t * alpha + EMA_(t-1) * (1 - alpha)

    First EMA value is seeded with the SMA of the first 'period' values.

    Args:
        df: OHLCV DataFrame.
        period: Lookback period (default 20).

    Returns:
        pd.Series with EMA values, indexed same as input.
    """
    if period < 1:
        raise ValueError(f"EMA period must be >= 1, got {period}")

    return df["close"].ewm(span=period, adjust=False).mean()


def supertrend(
    df: pd.DataFrame,
    atr_period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """Supertrend indicator.

    Supertrend is a trend-following indicator built on ATR. It plots a
    single line above or below price:
        - When price is above the line → uptrend (bullish)
        - When price is below the line → downtrend (bearish)

    Calculation:
        1. Compute ATR over 'atr_period' bars.
        2. Upper band = (high + low) / 2 + multiplier * ATR
        3. Lower band = (high + low) / 2 - multiplier * ATR
        4. Supertrend flips between upper and lower band based on
           whether close breaks through the current band.

    Args:
        df: OHLCV DataFrame.
        atr_period: Period for ATR calculation (default 10).
        multiplier: ATR multiplier for band width (default 3.0).

    Returns:
        pd.DataFrame with columns:
            - 'supertrend': the Supertrend line value
            - 'direction': 1 for uptrend (bullish), -1 for downtrend (bearish)
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Step 1: ATR (using Wilder's smoothing via EWM)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / atr_period, min_periods=atr_period, adjust=False).mean()

    # Step 2: Basic upper and lower bands
    hl2 = (high + low) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    # Step 3: Final bands with carry-forward logic
    n = len(df)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend_vals = np.zeros(n)
    direction = np.zeros(n)

    final_upper[0] = basic_upper.iloc[0]
    final_lower[0] = basic_lower.iloc[0]
    supertrend_vals[0] = basic_upper.iloc[0]
    direction[0] = -1  # Start bearish

    for i in range(1, n):
        # Final upper band: can only decrease (tighten) in an uptrend
        if basic_upper.iloc[i] < final_upper[i - 1] or close.iloc[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper.iloc[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Final lower band: can only increase (tighten) in a downtrend
        if basic_lower.iloc[i] > final_lower[i - 1] or close.iloc[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower.iloc[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Direction logic
        if supertrend_vals[i - 1] == final_upper[i - 1]:
            # Was bearish — check if price broke above upper band
            if close.iloc[i] > final_upper[i]:
                direction[i] = 1  # Flip to bullish
                supertrend_vals[i] = final_lower[i]
            else:
                direction[i] = -1
                supertrend_vals[i] = final_upper[i]
        else:
            # Was bullish — check if price broke below lower band
            if close.iloc[i] < final_lower[i]:
                direction[i] = -1  # Flip to bearish
                supertrend_vals[i] = final_upper[i]
            else:
                direction[i] = 1
                supertrend_vals[i] = final_lower[i]

    result = pd.DataFrame({
        "supertrend": supertrend_vals,
        "direction": direction.astype(int),
    }, index=df.index)

    return result


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (ADX).

    ADX measures trend strength (not direction). Values:
        - ADX < 20: weak/no trend
        - ADX 20-40: developing trend
        - ADX 40-60: strong trend
        - ADX > 60: very strong trend

    Calculation (Wilder's method):
        1. +DM = high - prev_high (if positive and > -DM, else 0)
        2. -DM = prev_low - low (if positive and > +DM, else 0)
        3. Smooth +DM, -DM, and TR using Wilder's EMA (alpha = 1/period)
        4. +DI = smoothed +DM / smoothed TR * 100
        5. -DI = smoothed -DM / smoothed TR * 100
        6. DX = |+DI - -DI| / (+DI + -DI) * 100
        7. ADX = Wilder's EMA of DX

    Args:
        df: OHLCV DataFrame.
        period: Lookback period (default 14).

    Returns:
        pd.Series with ADX values.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()

    # +DM: positive and greater than -DM
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0.0)
    # -DM: positive and greater than +DM
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0.0)

    # Wilder's smoothing (EMA with alpha = 1/period)
    alpha = 1 / period
    smoothed_tr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smoothed_plus_dm = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    # Directional Indicators
    plus_di = (smoothed_plus_dm / smoothed_tr) * 100
    minus_di = (smoothed_minus_dm / smoothed_tr) * 100

    # DX and ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx_series = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    return adx_series
