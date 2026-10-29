# Gold & Bitcoin Quantitative Backtesting Platform

A multi-timeframe quantitative backtesting platform for **Gold (GC=F)** and **Bitcoin (BTC-USD)**, featuring a categorized indicator library, a risk-managed strategy engine, performance analytics, audit logging, and a Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Prototype-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> **Project status: working prototype.** The architecture and core engine are solid, but this is
> not yet a production backtester. See **Known limitations** and **Roadmap** below before relying
> on any result. A full audit and production plan live in the project docs
> (`claude/codebase-audit-2026-10-28.md`, `claude/implementation-plan.md`).

## Features

- **Multi-Timeframe Support**: 1m, 5m, 15m, 30m, 1h, 1d with automatic yfinance lookback-limit enforcement
- **8 Hand-Rolled Indicators**: EMA, Supertrend, ADX, RSI, MACD, Bollinger Bands, ATR, VWAP — implemented from first principles (no pandas-ta)
- **Strategy Engine**: EMA Crossover, extensible via an abstract `Strategy` interface
- **Risk Management**: ATR-based position sizing, stop-loss, take-profit, and a max-drawdown kill-switch
- **Fill-at-Next-Open Rule**: signals on bar N execute at bar N+1's open (verified: no look-ahead on entries)
- **Analytics**: Sharpe, Sortino, Calmar, max drawdown + duration, win rate, profit factor, avg trade return
- **Audit Logging**: SQLite records every backtest run
- **Parquet Caching**: transparent data caching via a decorator pattern
- **Dark-Themed Dashboard**: Streamlit UI with candlestick charts, indicator overlays, equity curves, and trade markers

## Known limitations (as of 2026-10-29)

These are tracked and scheduled in `claude/implementation-plan.md`:

- **No transaction costs.** Commissions and slippage are not modeled, so reported returns are optimistic.
- **Sharpe/Sortino** are computed over every bar including flat, out-of-market bars, which distorts the ratios.
- **Max-drawdown kill-switch is permanent** — it does not re-enable after equity recovers.
- **Audit log does not store risk or cost parameters**, so a logged run is not yet fully reproducible from the DB alone.
- **Single strategy** shipped (EMA Crossover). Three additional strategies are planned.
- **Tests require a live network** (yfinance); there is no offline/CI test suite yet.

## Architecture

```
app.py (Streamlit Dashboard)
  |
  +-- data/           DataSource ABC -> YFinanceSource -> CachedDataSource
  +-- indicators/     Trend | Momentum | Volatility | Volume (8 indicators)
  +-- strategy/       Strategy ABC -> EMACrossover
  +-- risk/           ATR Position Sizing + SL/TP/Kill-Switch + Fill-at-Next-Open
  +-- analytics/      TearsheetResult (10 metrics, bar-count annualization)
  +-- logging_/       SQLite audit trail (backtest_runs table)
```

Every layer depends only on the layer below it via public interfaces (dependency inversion). Adding
new data sources, strategies, or indicators is intended to require **new files only**.

## Quick Start

### Prerequisites
- Python 3.11+

### Installation

```bash
git clone https://github.com/Anshuman0716/gold-btc-backtester.git
cd gold-btc-backtester
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### Run the Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The suite is fully offline (uses the committed parquet fixtures).

### Run Checkpoint Scripts

> Note: these are manual validation scripts that require a live internet connection (yfinance).
> A deterministic, offline `pytest` suite is planned (see Roadmap).

```bash
python -m checkpoints.checkpoint_phase1   # Data layer
python -m checkpoints.checkpoint_phase2   # Indicators
python -m checkpoints.checkpoint_phase3   # Strategy + Risk
python -m checkpoints.checkpoint_phase4   # Analytics
python -m checkpoints.checkpoint_phase5   # Logging
```

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.11+ |
| Data Source | yfinance |
| Data Format | pandas DataFrame (OHLCV) |
| Caching | Parquet (via pandas + pyarrow) |
| Indicators | Hand-rolled (numpy/pandas) |
| Charting | Plotly |
| Dashboard | Streamlit |
| Audit Logging | SQLite |

## Roadmap

- **Phase 0 — Credibility fixes:** fix Supertrend seeding, RSI warm-up, add a `LICENSE` file, replace static sample numbers with a reproducible fixed-date backtest _(done)_.
- **Phase 1 — Honest backtester:** transaction-cost model, corrected Sharpe/Sortino methodology, resettable kill-switch, leverage/cash guard, richer audit log (risk + cost params).
- **Phase 2 — Engineering hardening:** offline `pytest` suite (47 tests, no network), GitHub Actions CI (tests + benchmark-reproducibility check), pinned dependencies, inclusive end-date handling _(done)_.
- **Phase 3 — Extensibility:** add Donchian Breakout, Time-Series Momentum, and Bollinger/RSI Mean-Reversion strategies; add walk-forward / out-of-sample evaluation.

## Sample Results

Reproducible fixed-date benchmark. Re-run it with `python scripts/benchmark.py --update-readme`;
the numbers come only from the committed data fixtures, so they are identical on any date.

<!-- BENCHMARK:START -->
Fixed window **2022-01-01 to 2024-12-31**, 1d bars, data from committed fixtures (`tests/fixtures/`). Generated by `python scripts/benchmark.py`.

Risk: capital $100,000, risk/trade 1%, ATR(14) SL 2.0x / TP 3.0x, kill-switch at 20% drawdown. **No commissions or slippage are modeled yet**, so returns are optimistic.

| Asset | Strategy | Return | Buy & Hold | Sharpe | Sortino | Max DD | Win Rate | Profit Factor | Trades |
|---|---|---|---|---|---|---|---|---|---|
| BTC-USD | EMA Crossover (fast_period=9, slow_period=21) | -0.27% | 95.92% | 0.00 | 0.00 | -6.94% | 39.1% | 0.99 | 46 |
| BTC-USD | EMA Crossover (fast_period=20, slow_period=50) | -5.29% | 95.92% | -0.54 | -0.29 | -8.27% | 29.4% | 0.54 | 17 |
| GC=F | EMA Crossover (fast_period=9, slow_period=21) | -0.66% | 46.71% | -0.05 | -0.04 | -4.75% | 31.2% | 0.95 | 32 |
| GC=F | EMA Crossover (fast_period=20, slow_period=50) | 2.32% | 46.71% | 0.44 | 0.22 | -2.83% | 50.0% | 1.62 | 10 |

| Asset | First bar | Last bar | Bars | Fixture | SHA-256 (12) |
|---|---|---|---|---|---|
| BTC-USD | 2022-01-01 | 2024-12-31 | 1096 | `BTC_USD_1d_2022-01-01_2024-12-31.parquet` | `bef9a5f9fd0e` |
| GC=F | 2022-01-03 | 2024-12-31 | 754 | `GC_F_1d_2022-01-01_2024-12-31.parquet` | `9105ab137d45` |
<!-- BENCHMARK:END -->

## License

MIT — see the [`LICENSE`](LICENSE) file.
