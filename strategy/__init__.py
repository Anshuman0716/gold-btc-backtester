"""
Strategy layer — abstract Strategy interface and concrete implementations.
"""

from strategy.base import Strategy
from strategy.ema_crossover import EMACrossover

__all__ = [
    "Strategy",
    "EMACrossover",
]
