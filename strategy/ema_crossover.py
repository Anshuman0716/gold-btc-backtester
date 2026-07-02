"""
EMA Crossover Strategy.

The simplest trend-following strategy: when the fast EMA crosses above
the slow EMA, go long (BUY). When fast crosses below slow, go short (SELL).
Otherwise, HOLD.

This is the v1 strategy. Future strategies (regime-aware, mean reversion)
will implement the same Strategy interface with zero changes to this file.
"""

import pandas as pd

from strategy.base import Strategy
from indicators.trend import ema


class EMACrossover(Strategy):
    """EMA Crossover trading strategy.

    Signals:
        - BUY:  fast EMA crosses ABOVE slow EMA (bullish crossover)
        - SELL: fast EMA crosses BELOW slow EMA (bearish crossover)
        - HOLD: no crossover on this bar

    Crossover detection:
        A crossover occurs when the relative position of fast vs slow EMA
        changes from one bar to the next:
            - prev: fast <= slow, curr: fast > slow  -> BUY
            - prev: fast >= slow, curr: fast < slow  -> SELL

    Args:
        fast_period: Period for the fast (short-term) EMA. Default 9.
        slow_period: Period for the slow (long-term) EMA. Default 21.
    """

    def __init__(self, fast_period: int = 9, slow_period: int = 21) -> None:
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be less than "
                f"slow_period ({slow_period})."
            )
        self._fast_period = fast_period
        self._slow_period = slow_period

    @property
    def name(self) -> str:
        return "EMA Crossover"

    def get_params(self) -> dict:
        return {
            "fast_period": self._fast_period,
            "slow_period": self._slow_period,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate BUY/SELL/HOLD signals based on EMA crossovers.

        Args:
            df: OHLCV DataFrame.

        Returns:
            pd.Series of signal strings aligned to df.index.
        """
        fast_ema = ema(df, period=self._fast_period)
        slow_ema = ema(df, period=self._slow_period)

        # Relative position: fast > slow
        fast_above = fast_ema > slow_ema
        # Previous bar's relative position (fill NaN from shift with False)
        prev_fast_above = fast_above.shift(1).fillna(False).astype(bool)

        # Initialize all signals as HOLD
        signals = pd.Series("HOLD", index=df.index)

        # Bullish crossover: was below (or equal), now above
        bullish_cross = (~prev_fast_above) & fast_above
        signals[bullish_cross] = "BUY"

        # Bearish crossover: was above (or equal), now below
        bearish_cross = prev_fast_above & (~fast_above)
        signals[bearish_cross] = "SELL"

        # First bar is always HOLD (no previous bar to compare)
        signals.iloc[0] = "HOLD"

        # Validate signal values
        signals = self._validate_signals(signals)

        return signals
