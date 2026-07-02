"""
Performance analytics tearsheet.

Computes institutional-grade metrics from a list of trades and an equity curve.
All ratios use BAR-COUNT-BASED annualization so that metrics are comparable
across different intervals (1m, 1h, 1d).

Annualization:
    annualization_factor = sqrt(bars_per_year)

    Bars per year depends on the asset and interval:
        BTC (24/7):   1d -> 365, 1h -> 8760, 15m -> 35040
        Gold (M-F):   1d -> 252, 1h -> 1638, 15m -> 6552

    This is passed as a parameter so the caller decides based on
    the asset and interval being backtested.

Metrics produced:
    - Total Return (%)
    - Sharpe Ratio (annualized)
    - Sortino Ratio (annualized, downside deviation only)
    - Calmar Ratio (annualized return / max drawdown)
    - Maximum Drawdown (%)
    - Maximum Drawdown Duration (bars)
    - Win Rate (%)
    - Profit Factor (gross profit / gross loss)
    - Average Trade Return (%)
    - Number of Trades
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from risk.controls import Trade


# Bars per year lookup for annualization
# Format: {asset: {interval: bars_per_year}}
BARS_PER_YEAR: dict[str, dict[str, int]] = {
    "BTC-USD": {
        "1m": 365 * 24 * 60,      # 525,600
        "5m": 365 * 24 * 12,      # 105,120
        "15m": 365 * 24 * 4,      # 35,040
        "30m": 365 * 24 * 2,      # 17,520
        "1h": 365 * 24,           # 8,760
        "1d": 365,                # 365
    },
    "GC=F": {
        "1m": 252 * 6.5 * 60,     # 98,280
        "5m": 252 * 6.5 * 12,     # 19,656
        "15m": 252 * 6.5 * 4,     # 6,552
        "30m": 252 * 6.5 * 2,     # 3,276
        "1h": 252 * 6.5,          # 1,638
        "1d": 252,                # 252
    },
}


def get_annualization_factor(asset: str, interval: str) -> float:
    """Get the annualization factor (sqrt of bars per year).

    Args:
        asset: Ticker symbol ("BTC-USD" or "GC=F").
        interval: Bar interval ("1m", "5m", "15m", "30m", "1h", "1d").

    Returns:
        sqrt(bars_per_year) for the given asset/interval combination.
        Defaults to sqrt(252) if asset/interval not found.
    """
    bars = BARS_PER_YEAR.get(asset, {}).get(interval, 252)
    return np.sqrt(bars)


@dataclass
class TearsheetResult:
    """Complete performance analytics for a backtest run.

    All percentage values are stored as decimals (0.05 = 5%).
    The equity_curve is the full time series for charting.
    """
    equity_curve: pd.Series
    total_return: float             # e.g. 0.05 = 5%
    sharpe_ratio: float             # Annualized
    sortino_ratio: float            # Annualized (downside only)
    calmar_ratio: float             # Annualized return / max drawdown
    max_drawdown: float             # e.g. -0.15 = -15%
    max_drawdown_duration: int      # In bars (not calendar days)
    win_rate: float                 # e.g. 0.60 = 60%
    profit_factor: float            # Gross profit / gross loss
    avg_trade_return: float         # Mean P&L % per trade
    num_trades: int

    def to_dict(self) -> dict:
        """Convert to a flat dict for display / logging.

        Formats percentages as human-readable strings.
        """
        return {
            "Total Return": f"{self.total_return * 100:.2f}%",
            "Sharpe Ratio": f"{self.sharpe_ratio:.2f}",
            "Sortino Ratio": f"{self.sortino_ratio:.2f}",
            "Calmar Ratio": f"{self.calmar_ratio:.2f}",
            "Max Drawdown": f"{self.max_drawdown * 100:.2f}%",
            "Max DD Duration": f"{self.max_drawdown_duration} bars",
            "Win Rate": f"{self.win_rate * 100:.1f}%",
            "Profit Factor": f"{self.profit_factor:.2f}",
            "Avg Trade Return": f"{self.avg_trade_return * 100:.2f}%",
            "Num Trades": self.num_trades,
        }


def compute_tearsheet(
    trades: list[Trade],
    equity_curve: pd.Series,
    initial_capital: float = 100_000.0,
    asset: str = "BTC-USD",
    interval: str = "1d",
    risk_free_rate: float = 0.0,
) -> TearsheetResult:
    """Compute full performance tearsheet from trades and equity curve.

    Args:
        trades: List of completed Trade objects from the risk engine.
        equity_curve: Portfolio value at each bar (from apply_risk_controls).
        initial_capital: Starting portfolio value.
        asset: Ticker symbol (for annualization factor lookup).
        interval: Bar interval (for annualization factor lookup).
        risk_free_rate: Annualized risk-free rate (default 0).

    Returns:
        TearsheetResult with all computed metrics.
    """
    ann_factor = get_annualization_factor(asset, interval)

    # --- Clean equity curve ---
    eq = equity_curve.dropna()
    if len(eq) < 2:
        return _empty_tearsheet(equity_curve, len(trades))

    # --- Total Return ---
    total_return = (eq.iloc[-1] - initial_capital) / initial_capital

    # --- Returns series (bar-over-bar) ---
    returns = eq.pct_change().dropna()
    if len(returns) < 2:
        return _empty_tearsheet(equity_curve, len(trades))

    # --- Sharpe Ratio (annualized) ---
    # Sharpe = (mean_return - rf_per_bar) / std_return * sqrt(bars_per_year)
    bars_per_year = ann_factor ** 2  # Undo sqrt to get raw bars
    rf_per_bar = risk_free_rate / bars_per_year if bars_per_year > 0 else 0

    mean_return = returns.mean()
    std_return = returns.std()

    if std_return > 0:
        sharpe = (mean_return - rf_per_bar) / std_return * ann_factor
    else:
        sharpe = 0.0

    # --- Sortino Ratio (annualized, downside deviation only) ---
    downside_returns = returns[returns < rf_per_bar]
    if len(downside_returns) > 0:
        downside_std = np.sqrt(((downside_returns - rf_per_bar) ** 2).mean())
        if downside_std > 0:
            sortino = (mean_return - rf_per_bar) / downside_std * ann_factor
        else:
            sortino = 0.0
    else:
        # No losing bars -- infinite Sortino, cap at 0
        sortino = 0.0 if mean_return <= rf_per_bar else float("inf")

    # --- Maximum Drawdown ---
    peak = eq.cummax()
    drawdown = (eq - peak) / peak
    max_dd = drawdown.min()  # Most negative value

    # --- Maximum Drawdown Duration (in bars) ---
    max_dd_duration = _compute_max_dd_duration(eq)

    # --- Calmar Ratio (annualized return / |max drawdown|) ---
    annualized_return = (1 + total_return) ** (bars_per_year / len(eq)) - 1 if len(eq) > 0 else 0
    if max_dd != 0:
        calmar = annualized_return / abs(max_dd)
    else:
        calmar = 0.0

    # --- Trade-level metrics ---
    num_trades = len(trades)

    if num_trades > 0:
        # Win Rate
        wins = sum(1 for t in trades if t.pnl > 0)
        win_rate = wins / num_trades

        # Profit Factor (gross profit / gross loss)
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Average Trade Return (%)
        trade_returns = [t.pnl_pct for t in trades]
        avg_trade_return = np.mean(trade_returns)
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_return = 0.0

    return TearsheetResult(
        equity_curve=equity_curve,
        total_return=total_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_return=avg_trade_return,
        num_trades=num_trades,
    )


def _compute_max_dd_duration(equity: pd.Series) -> int:
    """Compute the longest drawdown duration in bars.

    A drawdown period starts when equity drops below its running peak
    and ends when equity makes a new high.

    Args:
        equity: Clean equity curve (no NaN).

    Returns:
        Duration of the longest drawdown in bars.
    """
    peak = equity.cummax()
    in_drawdown = equity < peak

    if not in_drawdown.any():
        return 0

    # Find consecutive drawdown stretches
    max_duration = 0
    current_duration = 0

    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return max_duration


def _empty_tearsheet(equity_curve: pd.Series, num_trades: int) -> TearsheetResult:
    """Return a zeroed-out tearsheet for edge cases (no data, no trades)."""
    return TearsheetResult(
        equity_curve=equity_curve,
        total_return=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
        max_drawdown=0.0,
        max_drawdown_duration=0,
        win_rate=0.0,
        profit_factor=0.0,
        avg_trade_return=0.0,
        num_trades=num_trades,
    )
