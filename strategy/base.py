"""
Abstract Strategy interface.

Every strategy must implement generate_signals() which takes an OHLCV
DataFrame and returns a pandas Series of signal values: BUY, SELL, or HOLD.

Signal contract:
    - Output Series is aligned to the input DataFrame's index.
    - Signal on bar N means: "at the CLOSE of bar N, I want to act."
    - The engine enforces fill-at-next-open: the order executes at bar N+1's open.
    - This separation ensures ZERO lookahead bias — the strategy never sees
      future data when generating signals.

Adding a new strategy:
    1. Create a new file in strategy/ (e.g. mean_reversion.py)
    2. Subclass Strategy, implement generate_signals() and get_params()
    3. Zero changes to existing code — open/closed principle.
"""

from abc import ABC, abstractmethod

import pandas as pd


# Valid signal values
VALID_SIGNALS = frozenset({"BUY", "SELL", "HOLD"})


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses must implement:
        - generate_signals(df) -> pd.Series of BUY/SELL/HOLD
        - get_params() -> dict of current parameter values
        - name (property) -> human-readable strategy name
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name (e.g. 'EMA Crossover')."""
        pass

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals from OHLCV data.

        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume]
                and DatetimeIndex.

        Returns:
            pd.Series with values from {"BUY", "SELL", "HOLD"}, indexed
            identically to the input DataFrame.
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """Return current strategy parameters as a dict.

        Used for:
            - Audit logging (recording exact params per backtest run)
            - Dashboard display
            - Reproducibility
        """
        pass

    def _edge_signals(self, state: pd.Series) -> pd.Series:
        """Convert a +1/-1 regime series into edge-triggered BUY/SELL/HOLD.

        The engine opens/flips on BUY/SELL and ignores repeats, so a strategy
        emits a signal only when the regime CHANGES. NaN entries carry the prior
        regime forward (ffill); the first bar is always HOLD.
        """
        state = state.ffill()
        prev = state.shift(1)
        signals = pd.Series("HOLD", index=state.index)
        signals[(state == 1) & (prev != 1)] = "BUY"
        signals[(state == -1) & (prev != -1)] = "SELL"
        signals.iloc[0] = "HOLD"
        return self._validate_signals(signals)

    def _validate_signals(self, signals: pd.Series) -> pd.Series:
        """Validate that all signal values are in {BUY, SELL, HOLD}.

        Called by concrete implementations after generating signals.

        Args:
            signals: Raw signal Series.

        Returns:
            Validated signal Series.

        Raises:
            ValueError: If any signal value is invalid.
        """
        invalid = set(signals.dropna().unique()) - VALID_SIGNALS
        if invalid:
            raise ValueError(
                f"Invalid signal values: {invalid}. "
                f"Must be one of: {VALID_SIGNALS}"
            )
        return signals
