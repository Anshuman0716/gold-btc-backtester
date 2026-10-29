"""
Risk controls -- stop-loss, take-profit, max drawdown kill-switch,
transaction costs, leverage cap, and fill-at-next-open order execution.

This module is the bridge between raw strategy signals and executable orders.
It enforces:
    1. Fill-at-next-open: signals on bar N execute at bar N+1's open price.
    2. Transaction costs: commission (bps) + slippage (bps) applied on both
       entry and exit fills. PnL is reported NET of costs.
    3. Stop-loss / take-profit, with a pessimistic same-bar assumption.
    4. Leverage cap: position notional is capped at max_leverage * capital.
    5. Max drawdown kill-switch (latch or cooldown-reset).

The output is a list of Trade objects (net of costs) plus the equity curve.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from risk.position_sizing import atr_position_size
from indicators.volatility import atr as compute_atr


@dataclass
class RiskParams:
    """Configurable risk management parameters."""
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.01
    atr_period: int = 14
    atr_multiplier: float = 2.0
    stop_loss_atr: float = 2.0
    take_profit_atr: float = 3.0
    max_drawdown_pct: float = 0.20
    use_pct: bool = False
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.03
    # --- Phase 1 additions ---
    commission_bps: float = 0.0          # per-side commission, basis points of notional
    slippage_bps: float = 0.0            # per-side slippage, basis points of fill price
    max_leverage: float = 1.0            # cap: notional <= max_leverage * capital
    pessimistic_fills: bool = True       # if both SL & TP touched same bar, assume SL
    kill_switch_mode: str = "latch"      # "latch" (permanent) or "reset" (cooldown)
    kill_switch_cooldown_bars: int = 20  # bars to wait before resuming (reset mode)


@dataclass
class Order:
    """A pending order to be filled at the next bar's open."""
    signal_time: datetime
    side: str                             # "BUY" or "SELL"
    signal_price: float


@dataclass
class Trade:
    """A completed round-trip trade (entry + exit), net of costs."""
    entry_time: datetime
    exit_time: datetime
    side: str                             # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    position_size: float
    stop_loss: float
    take_profit: float
    pnl: float = 0.0                      # Dollar P&L, NET of fees
    pnl_pct: float = 0.0                  # Net return on notional
    fee: float = 0.0                      # Total commission+slippage cost in dollars
    exit_reason: str = ""


def _apply_slippage(price: float, action: str, slip: float) -> float:
    """Adjust a fill price against the trader by `slip` (fraction).

    action 'buy'  -> pay up (price * (1+slip))
    action 'sell' -> receive less (price * (1-slip))
    """
    if action == "buy":
        return price * (1.0 + slip)
    return price * (1.0 - slip)


def apply_risk_controls(
    signals: pd.Series,
    df: pd.DataFrame,
    params: RiskParams,
) -> tuple[list[Trade], pd.Series]:
    """Process raw signals through risk controls and produce net-of-cost trades."""
    atr_series = compute_atr(df, period=params.atr_period)

    comm = params.commission_bps / 10_000.0
    slip = params.slippage_bps / 10_000.0

    trades: list[Trade] = []
    equity_curve = pd.Series(np.nan, index=df.index, dtype=float)

    capital = params.initial_capital
    position: Optional[dict] = None
    pending_order: Optional[Order] = None
    peak_equity = capital
    kill_switch_active = False
    cooldown_remaining = 0

    for i in range(len(df)):
        bar_time = df.index[i]
        bar_open = df["open"].iloc[i]
        bar_high = df["high"].iloc[i]
        bar_low = df["low"].iloc[i]
        bar_close = df["close"].iloc[i]
        current_atr = atr_series.iloc[i] if not np.isnan(atr_series.iloc[i]) else None

        # --- Step 1: Fill pending order at this bar's open (with slippage) ---
        if pending_order is not None:
            raw_fill = bar_open
            action = "buy" if pending_order.side == "BUY" else "sell"
            fill_price = _apply_slippage(raw_fill, action, slip)

            if current_atr is not None and current_atr > 0:
                pos_size = atr_position_size(
                    capital, params.risk_per_trade, current_atr, params.atr_multiplier
                )
            else:
                pos_size = (capital * params.risk_per_trade) / fill_price if fill_price > 0 else 0

            # Leverage cap: notional <= max_leverage * capital
            if fill_price > 0 and params.max_leverage > 0:
                max_size = (params.max_leverage * capital) / fill_price
                pos_size = min(pos_size, max_size)

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

            entry_fee = pos_size * fill_price * comm

            position = {
                "side": "LONG" if pending_order.side == "BUY" else "SHORT",
                "entry_time": bar_time,
                "entry_price": fill_price,
                "size": pos_size,
                "stop_loss": sl,
                "take_profit": tp,
                "entry_fee": entry_fee,
            }
            pending_order = None

        # --- Step 2: Check SL/TP on current bar (if in position) ---
        if position is not None:
            exit_price = None
            exit_reason = ""
            hit_sl = hit_tp = False

            if position["side"] == "LONG":
                hit_sl = bar_low <= position["stop_loss"]
                hit_tp = bar_high >= position["take_profit"]
            else:
                hit_sl = bar_high >= position["stop_loss"]
                hit_tp = bar_low <= position["take_profit"]

            if hit_sl and hit_tp:
                # Ambiguous intrabar order; pessimistic = assume stop-loss first
                if params.pessimistic_fills:
                    exit_price, exit_reason = position["stop_loss"], "stop_loss"
                else:
                    exit_price, exit_reason = position["take_profit"], "take_profit"
            elif hit_sl:
                exit_price, exit_reason = position["stop_loss"], "stop_loss"
            elif hit_tp:
                exit_price, exit_reason = position["take_profit"], "take_profit"

            if exit_price is not None:
                trade = _close_position(position, bar_time, exit_price, exit_reason, comm, slip)
                trades.append(trade)
                capital += trade.pnl
                position = None

        # --- Step 3: Max drawdown kill-switch ---
        current_equity = capital
        if position is not None:
            if position["side"] == "LONG":
                unrealized = (bar_close - position["entry_price"]) * position["size"]
            else:
                unrealized = (position["entry_price"] - bar_close) * position["size"]
            current_equity = capital + unrealized

        peak_equity = max(peak_equity, current_equity)
        if peak_equity > 0:
            current_drawdown = (peak_equity - current_equity) / peak_equity
            if not kill_switch_active and current_drawdown >= params.max_drawdown_pct:
                kill_switch_active = True
                if position is not None:
                    trade = _close_position(position, bar_time, bar_close, "kill_switch", comm, slip)
                    trades.append(trade)
                    capital += trade.pnl
                    position = None
                    current_equity = capital
                if params.kill_switch_mode == "reset":
                    cooldown_remaining = params.kill_switch_cooldown_bars

        # Cooldown-based reset: resume trading after cooldown, restart DD clock
        if kill_switch_active and params.kill_switch_mode == "reset":
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
            if cooldown_remaining <= 0:
                kill_switch_active = False
                peak_equity = current_equity

        equity_curve.iloc[i] = current_equity

        # --- Step 4: Process new signal (pending order fills next bar) ---
        signal = signals.iloc[i]
        if signal in ("BUY", "SELL") and not kill_switch_active:
            if position is not None:
                opposite = (
                    (position["side"] == "LONG" and signal == "SELL") or
                    (position["side"] == "SHORT" and signal == "BUY")
                )
                if opposite:
                    trade = _close_position(position, bar_time, bar_close, "signal", comm, slip)
                    trades.append(trade)
                    capital += trade.pnl
                    position = None
                    # equity at this bar reflects the realized exit (fix stale-equity timing)
                    equity_curve.iloc[i] = capital
                else:
                    continue

            pending_order = Order(
                signal_time=bar_time,
                side=signal,
                signal_price=bar_close,
            )

    # --- End of data: close any remaining position at last bar's close ---
    if position is not None:
        last_bar = df.index[-1]
        last_close = df["close"].iloc[-1]
        trade = _close_position(position, last_bar, last_close, "end_of_data", comm, slip)
        trades.append(trade)
        capital += trade.pnl
        equity_curve.iloc[-1] = capital

    return trades, equity_curve


def _close_position(
    position: dict,
    exit_time: datetime,
    exit_price: float,
    exit_reason: str,
    comm: float = 0.0,
    slip: float = 0.0,
) -> Trade:
    """Close a position, applying exit slippage + commission. PnL is net of costs."""
    entry_price = position["entry_price"]
    size = position["size"]
    entry_fee = position.get("entry_fee", 0.0)

    # Exit slippage works against us
    if position["side"] == "LONG":
        exit_fill = _apply_slippage(exit_price, "sell", slip)
        gross_pnl = (exit_fill - entry_price) * size
    else:
        exit_fill = _apply_slippage(exit_price, "buy", slip)
        gross_pnl = (entry_price - exit_fill) * size

    exit_fee = size * exit_fill * comm
    total_fee = entry_fee + exit_fee

    notional = entry_price * size
    net_pnl_full = gross_pnl - total_fee
    pnl_pct = (net_pnl_full / notional) if notional != 0 else 0.0

    return Trade(
        entry_time=position["entry_time"],
        exit_time=exit_time,
        side=position["side"],
        entry_price=entry_price,
        exit_price=exit_fill,
        position_size=size,
        stop_loss=position["stop_loss"],
        take_profit=position["take_profit"],
        pnl=net_pnl_full,
        pnl_pct=pnl_pct,
        fee=total_fee,
        exit_reason=exit_reason,
    )
