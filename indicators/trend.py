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
        The first (atr_period - 1) warmup bars are NaN in both columns.
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

    # Step 3: Final bands with carry-forward logic.
    # ATR is NaN during warmup, so the recursion must start at the first bar
    # where ATR is valid. Seeding at bar 0 would carry NaN forward forever
    # (every comparison against NaN is False).
    n = len(df)
    bu = basic_upper.to_numpy(dtype=float)
    bl = basic_lower.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    mid = hl2.to_numpy(dtype=float)

    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    supertrend_vals = np.full(n, np.nan)
    direction = np.full(n, np.nan)  # NaN during warmup, then +1 / -1

    valid = np.flatnonzero(~np.isnan(bu) & ~np.isnan(bl))
    if valid.size == 0:
        return pd.DataFrame(
            {"supertrend": supertrend_vals, "direction": direction},
            index=df.index,
        )

    start = valid[0]
    final_upper[start] = bu[start]
    final_lower[start] = bl[start]
    # Initial direction: bullish if close is above the bar's midpoint.
    if c[start] >= mid[start]:
        direction[start] = 1
        supertrend_vals[start] = final_lower[start]
    else:
        direction[start] = -1
        supertrend_vals[start] = final_upper[start]

    for i in range(start + 1, n):
        # Final upper band: only moves down, unless price closed above it
        if bu[i] < final_upper[i - 1] or c[i - 1] > final_upper[i - 1]:
            final_upper[i] = bu[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Final lower band: only moves up, unless price closed below it
        if bl[i] > final_lower[i - 1] or c[i - 1] < final_lower[i - 1]:
            final_lower[i] = bl[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Direction: track state explicitly (no float-equality checks)
        if direction[i - 1] == -1:
            # Bearish — flip if close breaks above the upper band
            direction[i] = 1 if c[i] > final_upper[i] else -1
        else:
            # Bullish — flip if close breaks below the lower band
            direction[i] = -1 if c[i] < final_lower[i] else 1

        supertrend_vals[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    # direction stays float so warmup bars can be NaN (+1.0 / -1.0 afterwards)
    result = pd.DataFrame({
        "supertrend": supertrend_vals,
        "direction": direction,
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

    # Directional Indicators (guard divide-by-zero on flat / zero-range markets)
    safe_tr = smoothed_tr.replace(0, np.nan)
    plus_di = (smoothed_plus_dm / safe_tr) * 100
    minus_di = (smoothed_minus_dm / safe_tr) * 100

    # DX and ADX; 0/0 (no directional movement) -> undefined, resolved to 0 below
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = (abs(plus_di - minus_di) / di_sum) * 100
    adx_series = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    # Flat series / warm-up -> finite 0 trend strength rather than NaN
    return adx_series.fillna(0.0)
