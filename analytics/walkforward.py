"""
Walk-forward / out-of-sample evaluation.

Rolling anchored walk-forward:
    1. On each in-sample (IS) train window, grid-search the strategy's parameters
       and keep the combo that maximizes the objective (Sharpe by default).
    2. Apply *those* parameters to the following out-of-sample (OOS) test window,
       with capital carried forward across windows.
    3. Stitch the OOS test segments into one continuous equity curve and report
       aggregate OOS metrics alongside the mean IS metric, exposing the overfit
       gap (IS - OOS). OOS results never see data used to pick their parameters.

This is the single most important guard against curve-fitting: a strategy that
looks great in-sample but collapses out-of-sample is overfit, and this makes
that visible.
"""

from dataclasses import dataclass, replace
from itertools import product
from typing import Optional

import pandas as pd

from analytics.tearsheet import TearsheetResult, compute_tearsheet
from risk.controls import RiskParams, Trade, apply_risk_controls


@dataclass
class WalkForwardResult:
    asset: str
    strategy: str
    folds: list                 # per-fold dicts (windows, chosen params, IS/OOS metrics)
    oos: TearsheetResult        # aggregate out-of-sample tearsheet (stitched equity)
    is_mean_sharpe: float       # mean in-sample Sharpe of the chosen params
    oos_sharpe: float
    oos_return: float

    @property
    def overfit_gap(self) -> float:
        """IS Sharpe minus OOS Sharpe. Large positive = overfit."""
        return self.is_mean_sharpe - self.oos_sharpe


def _param_combos(strategy_cls, grid: dict):
    """Yield valid kwarg dicts from a {param: [values]} grid (skips invalid combos)."""
    keys = list(grid)
    for values in product(*[grid[k] for k in keys]):
        kw = dict(zip(keys, values))
        try:
            strategy_cls(**kw)       # validate (e.g. EMA fast < slow)
        except ValueError:
            continue
        yield kw


def walk_forward(
    df: pd.DataFrame,
    strategy_cls,
    grid: dict,
    risk_params: RiskParams,
    asset: str = "BTC-USD",
    interval: str = "1d",
    train_bars: Optional[int] = None,
    test_bars: Optional[int] = None,
    step_bars: Optional[int] = None,
    objective: str = "sharpe",
) -> WalkForwardResult:
    """Run rolling walk-forward. See module docstring.

    Window sizing defaults to train=50% / test=15% of the series, stepping by the
    test size (non-overlapping OOS segments).
    """
    n = len(df)
    if train_bars is None:
        train_bars = int(n * 0.5)
    if test_bars is None:
        test_bars = max(1, int(n * 0.15))
    if step_bars is None:
        step_bars = test_bars

    name = strategy_cls(**next(_param_combos(strategy_cls, grid))).name
    first_capital = risk_params.initial_capital
    capital = first_capital

    folds = []
    oos_equity_parts: list[pd.Series] = []
    oos_trades: list[Trade] = []

    start = 0
    while start + train_bars + test_bars <= n:
        train = df.iloc[start:start + train_bars]
        test = df.iloc[start + train_bars:start + train_bars + test_bars]

        # --- IS grid search ---
        best_kw, best_obj, best_is = None, float("-inf"), None
        for kw in _param_combos(strategy_cls, grid):
            strat = strategy_cls(**kw)
            sig = strat.generate_signals(train)
            tr, eq = apply_risk_controls(sig, train, risk_params)
            ts = compute_tearsheet(tr, eq, risk_params.initial_capital, asset, interval)
            obj = ts.sharpe_ratio if objective == "sharpe" else ts.total_return
            if obj > best_obj:
                best_kw, best_obj, best_is = kw, obj, ts

        # --- OOS test with chosen params, capital carried forward ---
        rp = replace(risk_params, initial_capital=capital)
        strat = strategy_cls(**best_kw)
        sig = strat.generate_signals(test)
        tr, eq = apply_risk_controls(sig, test, rp)
        ts_oos = compute_tearsheet(tr, eq, capital, asset, interval)
        eq_clean = eq.dropna()
        if len(eq_clean):
            capital = float(eq_clean.iloc[-1])
        oos_equity_parts.append(eq)
        oos_trades.extend(tr)

        folds.append({
            "train_start": train.index[0].date(), "train_end": train.index[-1].date(),
            "test_start": test.index[0].date(), "test_end": test.index[-1].date(),
            "params": best_kw,
            "is_sharpe": best_is.sharpe_ratio, "is_return": best_is.total_return,
            "oos_sharpe": ts_oos.sharpe_ratio, "oos_return": ts_oos.total_return,
            "oos_trades": ts_oos.num_trades,
        })
        start += step_bars

    if not folds:
        empty = compute_tearsheet([], pd.Series(dtype=float), first_capital, asset, interval)
        return WalkForwardResult(asset, name, [], empty, 0.0, 0.0, 0.0)

    combined_eq = pd.concat(oos_equity_parts)
    combined_eq = combined_eq[~combined_eq.index.duplicated(keep="first")]
    oos_ts = compute_tearsheet(oos_trades, combined_eq, first_capital, asset, interval)
    is_mean_sharpe = sum(f["is_sharpe"] for f in folds) / len(folds)

    return WalkForwardResult(
        asset=asset,
        strategy=name,
        folds=folds,
        oos=oos_ts,
        is_mean_sharpe=is_mean_sharpe,
        oos_sharpe=oos_ts.sharpe_ratio,
        oos_return=oos_ts.total_return,
    )
