"""Donchian Channel breakout strategy (Turtle-style)."""

import numpy as np
import pandas as pd

from strategy.base import Strategy


class DonchianBreakout(Strategy):
    """Go long on an N-bar high breakout, short on an N-bar low breakout.

    The channel uses the PRIOR bar's rolling extremes (shift(1)) so the current
    bar's own high/low can't leak into its signal (no look-ahead).
    """

    def __init__(self, channel_period: int = 20) -> None:
        if channel_period < 2:
            raise ValueError(f"channel_period must be >= 2, got {channel_period}")
        self._channel_period = channel_period

    @property
    def name(self) -> str:
        return "Donchian Breakout"

    def get_params(self) -> dict:
        return {"channel_period": self._channel_period}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        n = self._channel_period
        upper = df["high"].rolling(n).max().shift(1)
        lower = df["low"].rolling(n).min().shift(1)
        state = pd.Series(np.nan, index=df.index)
        state[df["close"] > upper] = 1
        state[df["close"] < lower] = -1
        return self._edge_signals(state)
