from analytics.walkforward import walk_forward
from risk.controls import RiskParams
from strategy.ema_crossover import EMACrossover

GRID = {"fast_period": [5, 9], "slow_period": [21, 50]}
RISK = RiskParams(commission_bps=10, slippage_bps=5)


def test_walk_forward_runs_and_is_out_of_sample(btc):
    r = walk_forward(btc, EMACrossover, GRID, RISK, asset="BTC-USD",
                     train_bars=378, test_bars=126)
    assert len(r.folds) >= 2
    import numpy as np
    assert np.isfinite(r.oos_sharpe) and np.isfinite(r.oos_return)
    for f in r.folds:
        assert f["test_start"] > f["train_end"]          # OOS strictly after IS
        assert f["params"]["fast_period"] in GRID["fast_period"]
        assert f["params"]["slow_period"] in GRID["slow_period"]


def test_walk_forward_is_deterministic(btc):
    a = walk_forward(btc, EMACrossover, GRID, RISK, train_bars=378, test_bars=126)
    b = walk_forward(btc, EMACrossover, GRID, RISK, train_bars=378, test_bars=126)
    assert a.oos_return == b.oos_return
    assert [f["params"] for f in a.folds] == [f["params"] for f in b.folds]
