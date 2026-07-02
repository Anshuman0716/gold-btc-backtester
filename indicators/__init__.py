"""
Indicators library — categorized technical indicators for backtesting.

Categories:
    - Trend: EMA, Supertrend, ADX
    - Momentum: RSI, MACD
    - Volatility: Bollinger Bands, ATR
    - Volume: VWAP

All indicators are hand-rolled using numpy/pandas (no external TA library).
"""

from indicators.trend import ema, supertrend, adx
from indicators.momentum import rsi, macd
from indicators.volatility import bollinger_bands, atr
from indicators.volume import vwap
from indicators.registry import INDICATOR_REGISTRY, INDICATOR_FUNCTIONS, INDICATOR_PARAMS

__all__ = [
    # Trend
    "ema",
    "supertrend",
    "adx",
    # Momentum
    "rsi",
    "macd",
    # Volatility
    "bollinger_bands",
    "atr",
    # Volume
    "vwap",
    # Registry
    "INDICATOR_REGISTRY",
    "INDICATOR_FUNCTIONS",
    "INDICATOR_PARAMS",
]
