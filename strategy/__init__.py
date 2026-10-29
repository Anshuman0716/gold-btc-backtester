"""Strategy layer — abstract interface, concrete strategies, and a registry."""

from strategy.base import Strategy
from strategy.ema_crossover import EMACrossover
from strategy.donchian import DonchianBreakout
from strategy.momentum import TimeSeriesMomentum
from strategy.mean_reversion import BollingerMeanReversion

# Registry drives the dashboard: display name -> class + param spec.
# Param spec: {label, default, min, max, step, type ("int" | "float")}.
STRATEGY_REGISTRY: dict[str, dict] = {
    "EMA Crossover": {
        "class": EMACrossover,
        "params": {
            "fast_period": {"label": "Fast EMA", "default": 9, "min": 2, "max": 50, "step": 1, "type": "int"},
            "slow_period": {"label": "Slow EMA", "default": 21, "min": 5, "max": 200, "step": 1, "type": "int"},
        },
    },
    "Donchian Breakout": {
        "class": DonchianBreakout,
        "params": {
            "channel_period": {"label": "Channel Period", "default": 20, "min": 5, "max": 100, "step": 1, "type": "int"},
        },
    },
    "Time-Series Momentum": {
        "class": TimeSeriesMomentum,
        "params": {
            "lookback": {"label": "Lookback (bars)", "default": 50, "min": 5, "max": 200, "step": 1, "type": "int"},
        },
    },
    "Bollinger Mean-Reversion": {
        "class": BollingerMeanReversion,
        "params": {
            "bb_period": {"label": "BB Period", "default": 20, "min": 5, "max": 100, "step": 1, "type": "int"},
            "bb_std": {"label": "BB Std Dev", "default": 2.0, "min": 1.0, "max": 3.0, "step": 0.5, "type": "float"},
            "rsi_period": {"label": "RSI Period", "default": 14, "min": 2, "max": 50, "step": 1, "type": "int"},
            "oversold": {"label": "Oversold <", "default": 30.0, "min": 5.0, "max": 45.0, "step": 5.0, "type": "float"},
            "overbought": {"label": "Overbought >", "default": 70.0, "min": 55.0, "max": 95.0, "step": 5.0, "type": "float"},
        },
    },
}

__all__ = [
    "Strategy",
    "EMACrossover",
    "DonchianBreakout",
    "TimeSeriesMomentum",
    "BollingerMeanReversion",
    "STRATEGY_REGISTRY",
]
