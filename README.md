# Gold & Bitcoin Quantitative Backtesting Platform

A multi-timeframe quantitative backtesting platform for **Gold (GC=F)** and **Bitcoin (BTC-USD)**, featuring a categorized indicator library, risk-managed strategy engine, institutional-grade performance analytics, audit logging, and a Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Multi-Timeframe Support**: 1m, 5m, 15m, 30m, 1h, 1d with automatic yfinance lookback limit enforcement
- **8 Hand-Rolled Indicators**: EMA, Supertrend, ADX, RSI, MACD, Bollinger Bands, ATR, VWAP — all implemented from first principles (no pandas-ta)
- **Strategy Engine**: EMA Crossover with configurable parameters, extensible via abstract interface
- **Risk Management**: ATR-based position sizing, stop-loss, take-profit, and max drawdown kill-switch
- **Fill-at-Next-Open Rule**: Zero lookahead bias — signals on bar N execute at bar N+1's open
- **Institutional Analytics**: Sharpe, Sortino, Calmar ratios with bar-count-based annualization, plus win rate, profit factor, max drawdown duration
- **Audit Logging**: SQLite database records every backtest run with full parameters for reproducibility
- **Parquet Caching**: Transparent data caching with decorator pattern for fast repeated backtests
- **Dark-Themed Dashboard**: Streamlit UI with candlestick charts, indicator overlays, equity curves, and trade markers

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

Every layer depends only on the layer below it via public interfaces (dependency inversion). Adding new data sources, strategies, or indicators requires **new files only** — zero changes to existing code.

## Quick Start

### Prerequisites
- Python 3.11+

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/gold-btc-backtester.git
cd gold-btc-backtester
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### Run Checkpoint Tests

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
| Deployment | Streamlit Community Cloud |

## Project Structure

```
gold-btc-backtester/
├── app.py                    # Streamlit dashboard entry point
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── config.toml           # Dark theme configuration
├── data/
│   ├── base.py               # Abstract DataSource interface
│   ├── yfinance_source.py    # Concrete yfinance implementation
│   ├── interval_limits.py    # Lookback config + validation
│   └── cache.py              # Parquet-based caching
├── indicators/
│   ├── trend.py              # EMA, Supertrend, ADX
│   ├── momentum.py           # RSI, MACD
│   ├── volatility.py         # Bollinger Bands, ATR
│   ├── volume.py             # VWAP
│   └── registry.py           # Category -> indicator mapping
├── strategy/
│   ├── base.py               # Strategy interface (signal contract)
│   └── ema_crossover.py      # EMA Crossover strategy
├── risk/
│   ├── position_sizing.py    # ATR-based position sizing
│   └── controls.py           # SL, TP, max drawdown kill-switch
├── analytics/
│   └── tearsheet.py          # Sharpe, Sortino, Calmar, Max DD, etc.
├── logging_/
│   ├── db.py                 # SQLite schema and connection
│   └── recorder.py           # Backtest run recorder
└── checkpoints/
    ├── checkpoint_phase1.py  # Data layer validation
    ├── checkpoint_phase2.py  # Indicator validation
    ├── checkpoint_phase3.py  # Strategy + Risk validation
    ├── checkpoint_phase4.py  # Analytics validation
    └── checkpoint_phase5.py  # Logging validation
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Hand-rolled indicators | Interview-defensible — every formula explained from first principles |
| Parquet caching | Column-oriented, compressed, type-preserving — real data engineering |
| Decorator pattern for caching | `CachedDataSource` wraps any `DataSource` — composition over inheritance |
| Fill-at-next-open | Prevents lookahead bias — the #1 flaw interviewers look for |
| Bar-count annualization | Metrics comparable across intervals (1h Sharpe == 1d Sharpe) |
| SQLite audit log | Institutional audit-trail habits without overengineering |

## Sample Results

**BTC-USD 1d, EMA(9/21), 1 year backtest:**

| Metric | Value |
|---|---|
| Total Return | +3.72% |
| Sharpe Ratio | 0.95 |
| Sortino Ratio | 0.72 |
| Max Drawdown | -4.39% |
| Win Rate | 43.8% |
| Profit Factor | 1.71 |

## License

MIT
