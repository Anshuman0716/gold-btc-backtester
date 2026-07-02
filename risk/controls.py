"""
Risk controls -- stop-loss, take-profit, max drawdown kill-switch,
and fill-at-next-open order execution.

This module is the bridge between raw strategy signals and executable orders.
It enforces:
    1. Fill-at-next-open: signals on bar N execute at bar N+1's open price.
       This is the #1 defense against lookahead bias.
    2. Stop-loss: exit if price moves against the position by a threshold.
    3. Take-profit: exit if price moves in favor by a threshold.
    4. Max drawdown kill-switch: block all new trades if portfolio drawdown
       exceeds a configurable limit (e.g. 20%).

The output is a list of Trade objects representing completed round-trip trades
(entry + exit), which feeds directly into the analytics tearsheet.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from risk.position_sizing import atr_position_size
from indicators.volatility import atr as compute_atr


@dataclass
class RiskParams:
    """Configurable risk management parameters.

    All thresholds are expressed as ATR multiples by default.
    Set use_pct=True to interpret stop_loss and take_profit as
    fixed percentages instead.
    """
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.01          # 1% of account per trade
    atr_period: int = 14
    atr_multiplier: float = 2.0           # For position sizing stop distance
    stop_loss_atr: float = 2.0            # Stop-loss at 2x ATR from entry
    take_profit_atr: float = 3.0          # Take-profit at 3x ATR from entry
    max_drawdown_pct: float = 0.20        # Kill-switch at 20% drawdown
    use_pct: bool = False                 # If True, SL/TP are fixed %
    stop_loss_pct: float = 0.02           # 2% stop-loss (when use_pct=True)
    take_profit_pct: float = 0.03         # 3% take-profit (when use_pct=True)


@dataclass
class Order:
    """Represents a pending order to be filled at the next bar's open."""
    signal_time: datetime                 # When the signal was generated
    side: str                             # "BUY" or "SELL"
    signal_price: float                   # Close price when signal was generated


@dataclass
class Trade:
    """A completed round-trip trade (entry + exit).

    This is the atomic unit that feeds into the analytics tearsheet.
    """
    entry_time: datetime
    exit_time: datetime
    side: str                             # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    position_size: float
    stop_loss: float
    take_profit: float
    pnl: float = 0.0                      # Dollar P&L
    pnl_pct: float = 0.0                  # Percentage return
    exit_reason: str = ""                 # "signal", "stop_loss", "take_profit", "kill_switch"


def apply_risk_controls(
    signals: pd.Series,
    df: pd.DataFrame,
    params: RiskParams,
) -> tuple[list[Trade], pd.Series]:
    """Process raw signals through risk controls and produce trades.

    This is the main engine function. It walks through every bar sequentially,
    managing position state, checking stop-loss/take-profit, and enforcing
    the max drawdown kill-switch.

    Fill-at-next-open rule:
        A signal on bar N creates a pending order. The order fills at
        bar N+1's OPEN price. This ensures the strategy never trades
        on information it wouldn't have in live trading.

    Args:
        signals: pd.Series of BUY/SELL/HOLD from a Strategy.
        df: OHLCV DataFrame aligned with signals.
        params: RiskParams with all configurable thresholds.

    Returns:
        Tuple of:
            - list[Trade]: completed round-trip trades
            - pd.Series: equity curve (portfolio value at each bar)
    """
    # Compute ATR for position sizing and SL/TP
    atr_series = compute_atr(df, period=params.atr_period)

    trades: list[Trade] = []
    equity_curve = pd.Series(np.nan, index=df.index, dtype=float)

    # Portfolio state
    capital = params.initial_capital
    position: Optional[dict] = None      # Active position (None = flat)
    pending_order: Optional[Order] = None # Order waiting to fill
    peak_equity = capital                 # For drawdown calculation
    kill_switch_active = False            # True when max DD exceeded

    for i in range(len(df)):
        bar_time = df.index[i]
        bar_open = df["open"].iloc[i]
        bar_high = df["high"].iloc[i]
        bar_low = df["low"].iloc[i]
        bar_close = df["close"].iloc[i]
        current_atr = atr_series.iloc[i] if not np.isnan(atr_series.iloc[i]) else None

        # --- Step 1: Fill pending order at this bar's open ---
        if pending_order is not None:
            fill_price = bar_open

            if current_atr is not None and current_atr > 0:
                pos_size = atr_position_size(
                    capital, params.risk_per_trade, current_atr, params.atr_multiplier
                )
            else:
                # Fallback: risk 1% of capital at current price
                pos_size = (capital * params.risk_per_trade) / fill_price if fill_price > 0 else 0

            # Calculate SL/TP levels
            if params.use_pct:
                if pending_order.side == "BUY":
                    sl = fill_price * (1 - params.stop_loss_pct)
                    tp = fill_price * (1 + params.take_profit_pct)
                else:
                    sl = fill_price * (1 + params.stop_loss_pct)
                    tp = fill_price * (1 - params.take_profit_pct)
            else:
                atr_val = current_atr if current_atr else fill_price * 0.02
                if pending_order.side == "BUY":
                    sl = fill_price - params.stop_loss_atr * atr_val
                    tp = fill_price + params.take_profit_atr * atr_val
                else:
                    sl = fill_price + params.stop_loss_atr * atr_val
                    tp = fill_price - params.take_profit_atr * atr_val

            position = {
                "side": "LONG" if pending_order.side == "BUY" else "SHORT",
                "entry_time": bar_time,
                "entry_price": fill_price,
                "size": pos_size,
                "stop_loss": sl,
                "take_profit": tp,
            }
            pending_order = None

        # --- Step 2: Check SL/TP on current bar (if in position) ---
        if position is not None:
            exit_price = None
            exit_reason = ""

            if position["side"] == "LONG":
                # Stop-loss hit (price went below SL)
                if bar_low <= position["stop_loss"]:
                    exit_price = position["stop_loss"]
                    exit_reason = "stop_loss"
                # Take-profit hit (price went above TP)
                elif bar_high >= position["take_profit"]:
                    exit_price = position["take_profit"]
                    exit_reason = "take_profit"
            else:  # SHORT
                # Stop-loss hit (price went above SL)
                if bar_high >= position["stop_loss"]:
                    exit_price = position["stop_loss"]
                    exit_reason = "stop_loss"
                # Take-profit hit (price went below TP)
                elif bar_low <= position["take_profit"]:
                    exit_price = position["take_profit"]
                    exit_reason = "take_profit"

            if exit_price is not None:
                trade = _close_position(position, bar_time, exit_price, exit_reason)
                trades.append(trade)
                capital += trade.pnl
                position = None

        # --- Step 3: Check max drawdown kill-switch ---
        current_equity = capital
        if position is not None:
            # Mark-to-market: unrealized P&L
            if position["side"] == "LONG":
                unrealized = (bar_close - position["entry_price"]) * position["size"]
            else:
                unrealized = (position["entry_price"] - bar_close) * position["size"]
            current_equity = capital + unrealized

        peak_equity = max(peak_equity, current_equity)
        if peak_equity > 0:
            current_drawdown = (peak_equity - current_equity) / peak_equity
            if current_drawdown >= params.max_drawdown_pct:
                kill_switch_active = True

                # Force-close any open position
                if position is not None:
                    trade = _close_position(position, bar_time, bar_close, "kill_switch")
                    trades.append(trade)
                    capital += trade.pnl
                    position = None

        equity_curve.iloc[i] = current_equity

        # --- Step 4: Process new signal (generate pending order) ---
        signal = signals.iloc[i]

        if signal in ("BUY", "SELL") and not kill_switch_active:
            # Close existing position if signal is opposite
            if position is not None:
                opposite = (
                    (position["side"] == "LONG" and signal == "SELL") or
                    (position["side"] == "SHORT" and signal == "BUY")
                )
                if opposite:
                    trade = _close_position(position, bar_time, bar_close, "signal")
                    trades.append(trade)
                    capital += trade.pnl
                    position = None
                else:
                    # Same direction signal while in position -- ignore
                    continue

            # Create pending order (fills at NEXT bar's open)
            pending_order = Order(
                signal_time=bar_time,
                side=signal,
                signal_price=bar_close,
            )

    # --- End of data: close any remaining position at last bar's close ---
    if position is not None:
        last_bar = df.index[-1]
        last_close = df["close"].iloc[-1]
        trade = _close_position(position, last_bar, last_close, "end_of_data")
        trades.append(trade)
        capital += trade.pnl
        equity_curve.iloc[-1] = capital

    return trades, equity_curve


def _close_position(
    position: dict,
    exit_time: datetime,
    exit_price: float,
    exit_reason: str,
) -> Trade:
    """Close an open position and create a Trade record.

    Args:
        position: Active position dict.
        exit_time: Timestamp of the exit bar.
        exit_price: Price at which the position is closed.
        exit_reason: Why the position was closed.

    Returns:
        Completed Trade object with P&L calculated.
    """
    entry_price = position["entry_price"]
    size = position["size"]

    if position["side"] == "LONG":
        pnl = (exit_price - entry_price) * size
    else:  # SHORT
        pnl = (entry_price - exit_price) * size

    pnl_pct = (exit_price - entry_price) / entry_price if entry_price != 0 else 0
    if position["side"] == "SHORT":
        pnl_pct = -pnl_pct

    return Trade(
        entry_time=position["entry_time"],
        exit_time=exit_time,
        side=position["side"],
        entry_price=entry_price,
        exit_price=exit_price,
        position_size=size,
        stop_loss=position["stop_loss"],
        take_profit=position["take_profit"],
        pnl=pnl,
        pnl_pct=pnl_pct,
        exit_reason=exit_reason,
    )
