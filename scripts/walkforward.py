"""
Walk-forward / out-of-sample report — reproducible, offline from committed fixtures.

For each asset x strategy, parameters are grid-searched on each in-sample window and
applied to the following out-of-sample window (capital carried forward). The report
shows mean in-sample Sharpe vs aggregate out-of-sample Sharpe and the overfit gap.

Usage:
    python scripts/walkforward.py                 # write results/walkforward.md
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from analytics.walkforward import walk_forward
from risk.controls import RiskParams
from strategy import STRATEGY_REGISTRY

START, END, INTERVAL = "2022-01-01", "2024-12-31", "1d"
SYMBOLS = ["BTC-USD", "GC=F"]
TRAIN_BARS, TEST_BARS = 378, 126   # ~1.5y train / ~6m test, stepping by test size

RISK = RiskParams(
    initial_capital=100_000, risk_per_trade=0.01, atr_period=14,
    stop_loss_atr=2.0, take_profit_atr=3.0, max_drawdown_pct=0.20,
    commission_bps=10, slippage_bps=5,   # costs ON for walk-forward
)

# Discrete search grids (kept small; bounded by registry param ranges).
GRIDS = {
    "EMA Crossover": {"fast_period": [5, 9, 20], "slow_period": [21, 50, 100]},
    "Donchian Breakout": {"channel_period": [10, 20, 55]},
    "Time-Series Momentum": {"lookback": [20, 50, 100]},
    "Bollinger Mean-Reversion": {
        "bb_period": [20], "bb_std": [2.0], "rsi_period": [14],
        "oversold": [25, 30], "overbought": [70, 75],
    },
}

FIXTURE_DIR = ROOT / "tests" / "fixtures"
RESULTS_PATH = ROOT / "results" / "walkforward.md"


def fixture_path(symbol: str) -> Path:
    safe = symbol.replace("=", "_").replace("-", "_")
    return FIXTURE_DIR / f"{safe}_{INTERVAL}_{START}_{END}.parquet"


def load_fixture(symbol: str) -> pd.DataFrame:
    path = fixture_path(symbol)
    if not path.exists():
        sys.exit(f"Missing fixture {path.relative_to(ROOT)}. Run scripts/benchmark.py --refresh-fixtures once.")
    return pd.read_parquet(path)


def run() -> str:
    rows = []
    for symbol in SYMBOLS:
        df = load_fixture(symbol)
        for name, spec in STRATEGY_REGISTRY.items():
            r = walk_forward(df, spec["class"], GRIDS[name], RISK,
                             asset=symbol, interval=INTERVAL,
                             train_bars=TRAIN_BARS, test_bars=TEST_BARS)
            rows.append(
                f"| {symbol} | {name} | {len(r.folds)} | {r.is_mean_sharpe:+.2f} "
                f"| {r.oos_sharpe:+.2f} | {r.overfit_gap:+.2f} | {r.oos_return * 100:+.2f}% "
                f"| {r.oos.max_drawdown * 100:.2f}% | {r.oos.num_trades} |"
            )
    lines = [
        f"Walk-forward, fixed window **{START} to {END}**, {INTERVAL} bars, committed fixtures. "
        f"Train {TRAIN_BARS} bars / test {TEST_BARS} bars, stepping by test size. "
        "Parameters are chosen in-sample (max Sharpe) and applied out-of-sample; capital carries forward. "
        "Costs: 10 bps commission + 5 bps slippage per side.",
        "",
        "A large positive **Overfit gap** (mean IS Sharpe minus OOS Sharpe) means the in-sample "
        "parameter choice did not generalize.",
        "",
        "| Asset | Strategy | Folds | IS Sharpe (mean) | OOS Sharpe | Overfit gap | OOS Return | OOS Max DD | OOS Trades |",
        "|---|---|---|---|---|---|---|---|---|",
        *rows,
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    table = run()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("# Walk-forward results\n\n" + table, encoding="utf-8", newline="\n")
    print(table)
    print(f"  wrote {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
