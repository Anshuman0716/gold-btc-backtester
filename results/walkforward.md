# Walk-forward results

Walk-forward, fixed window **2022-01-01 to 2024-12-31**, 1d bars, committed fixtures. Train 378 bars / test 126 bars, stepping by test size. Parameters are chosen in-sample (max Sharpe) and applied out-of-sample; capital carries forward. Costs: 10 bps commission + 5 bps slippage per side.

A large positive **Overfit gap** (mean IS Sharpe minus OOS Sharpe) means the in-sample parameter choice did not generalize.

| Asset | Strategy | Folds | IS Sharpe (mean) | OOS Sharpe | Overfit gap | OOS Return | OOS Max DD | OOS Trades |
|---|---|---|---|---|---|---|---|---|
| BTC-USD | EMA Crossover | 5 | +2.97 | -0.72 | +3.69 | -2.62% | -4.27% | 23 |
| BTC-USD | Donchian Breakout | 5 | +1.42 | -1.17 | +2.59 | -2.93% | -5.00% | 12 |
| BTC-USD | Time-Series Momentum | 5 | +1.00 | -2.22 | +3.22 | -5.77% | -7.65% | 32 |
| BTC-USD | Bollinger Mean-Reversion | 5 | -0.21 | -4.93 | +4.72 | -4.12% | -4.83% | 4 |
| GC=F | EMA Crossover | 2 | +1.71 | 1.15 | +0.56 | +1.32% | -3.33% | 8 |
| GC=F | Donchian Breakout | 2 | +0.94 | -2.33 | +3.27 | -3.83% | -4.82% | 7 |
| GC=F | Time-Series Momentum | 2 | +0.13 | -0.86 | +0.98 | -1.70% | -4.05% | 19 |
| GC=F | Bollinger Mean-Reversion | 2 | +7.70 | 4.19 | +3.51 | +1.66% | -1.11% | 3 |
