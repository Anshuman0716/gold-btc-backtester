"""Bollinger-Band mean-reversion, RSI-filtered."""

import numpy as np
import pandas as pd

from strategy.base import Strategy
from indicators.volatility import bollinger_bands
from indicators.momentum import rsi


class BollingerMeanReversion(Strategy):
    """Fade extremes: go long at the lower band when oversold, short at the
    upper band when overbought. In this engine positions exit by reversing at
    the opposite extreme or via stop-loss/take-profit (there is no flat state).
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> None:
        if bb_period < 2:
            raise ValueError(f"bb_period must be >= 2, got {bb_period}")
        if bb_std <= 0:
            raise ValueError(f"bb_std must be > 0, got {bb_std}")
        if rsi_period < 2:
            raise ValueError(f"rsi_period must be >= 2, got {rsi_period}")
        if not (0 < oversold < overbought < 100):
            raise ValueError("require 0 < oversold < overbought < 100")
        self._bb_period = bb_period
        self._bb_std = bb_std
        self._rsi_period = rsi_period
        self._oversold = oversold
        self._overbought = overbought

    @property
    def name(self) -> str:
        return "Bollinger Mean-Reversion"

    def get_params(self) -> dict:
        return {
            "bb_period": self._bb_period,
            "bb_std": self._bb_std,
            "rsi_period": self._rsi_period,
            "oversold": self._oversold,
            "overbought": self._overbought,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        bb = bollinger_bands(df, self._bb_period, self._bb_std)
        r = rsi(df, self._rsi_period)
        long_sig = (df["close"] < bb["lower"]) & (r < self._oversold)
        short_sig = (df["close"] > bb["upper"]) & (r > self._overbought)
        state = pd.Series(np.nan, index=df.index)
        state[long_sig] = 1
        state[short_sig] = -1
        return self._edge_signals(state)
