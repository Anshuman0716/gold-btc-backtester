"""
Indicator registry — maps category names to indicator functions and params.

Used by the dashboard to:
    1. Organize indicators into collapsible category groups
    2. Auto-generate parameter sliders/inputs from INDICATOR_PARAMS
    3. Look up and call the correct function from INDICATOR_FUNCTIONS

Adding a new indicator requires:
    1. Write the function in the appropriate category file
    2. Add entries to INDICATOR_REGISTRY, INDICATOR_FUNCTIONS, and INDICATOR_PARAMS
    Zero changes to existing code — open/closed principle.
"""

from indicators.trend import ema, supertrend, adx
from indicators.momentum import rsi, macd
from indicators.volatility import bollinger_bands, atr
from indicators.volume import vwap


# Category -> list of indicator names (for dashboard grouping)
INDICATOR_REGISTRY: dict[str, list[str]] = {
    "Trend": ["EMA", "Supertrend", "ADX"],
    "Momentum": ["RSI", "MACD"],
    "Volatility": ["Bollinger Bands", "ATR"],
    "Volume": ["VWAP"],
}

# Indicator name -> callable function
INDICATOR_FUNCTIONS: dict[str, callable] = {
    "EMA": ema,
    "Supertrend": supertrend,
    "ADX": adx,
    "RSI": rsi,
    "MACD": macd,
    "Bollinger Bands": bollinger_bands,
    "ATR": atr,
    "VWAP": vwap,
}

# Indicator name -> default parameters (for dashboard auto-generation)
# Each param: {"label": str, "default": value, "min": value, "max": value, "step": value}
INDICATOR_PARAMS: dict[str, dict] = {
    "EMA": {
        "period": {"label": "Period", "default": 20, "min": 2, "max": 200, "step": 1},
    },
    "Supertrend": {
        "atr_period": {"label": "ATR Period", "default": 10, "min": 1, "max": 50, "step": 1},
        "multiplier": {"label": "Multiplier", "default": 3.0, "min": 0.5, "max": 10.0, "step": 0.5},
    },
    "ADX": {
        "period": {"label": "Period", "default": 14, "min": 2, "max": 50, "step": 1},
    },
    "RSI": {
        "period": {"label": "Period", "default": 14, "min": 2, "max": 50, "step": 1},
    },
    "MACD": {
        "fast": {"label": "Fast Period", "default": 12, "min": 2, "max": 50, "step": 1},
        "slow": {"label": "Slow Period", "default": 26, "min": 10, "max": 100, "step": 1},
        "signal": {"label": "Signal Period", "default": 9, "min": 2, "max": 50, "step": 1},
    },
    "Bollinger Bands": {
        "period": {"label": "Period", "default": 20, "min": 2, "max": 100, "step": 1},
        "std_dev": {"label": "Std Dev", "default": 2.0, "min": 0.5, "max": 5.0, "step": 0.5},
    },
    "ATR": {
        "period": {"label": "Period", "default": 14, "min": 1, "max": 50, "step": 1},
    },
    "VWAP": {},  # No configurable params
}
