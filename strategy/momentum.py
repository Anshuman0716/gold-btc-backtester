"""Time-series momentum strategy (classic TSMOM)."""

import numpy as np
import pandas as pd

from strategy.base import Strategy


class TimeSeriesMomentum(Strategy):
    """Long when the k-bar return is positive, short when negative.

    Regime = sign(close - close.shift(lookback)). A simple, robust trend filter
    documented to work on crypto and commodities.
    """

    def __init__(self, lookback: int = 50) -> None:
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")
        self._lookback = lookback

    @property
    def name(self) -> str:
        return "Time-Series Momentum"

    def get_params(self) -> dict:
        return {"lookback": self._lookback}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        mom = df["close"] - df["close"].shift(self._lookback)
        state = pd.Series(np.nan, index=df.index)
        state[mom > 0] = 1
        state[mom < 0] = -1
        return self._edge_signals(state)
