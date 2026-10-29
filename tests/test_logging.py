import json
from datetime import datetime

import pandas as pd

from analytics.tearsheet import compute_tearsheet
from logging_.recorder import get_all_runs, record_run


def test_record_and_read_back(tmp_path):
    db = str(tmp_path / "log.db")
    eq = pd.Series([100.0, 101.0, 102.0, 103.0], index=pd.date_range("2024-01-01", periods=4))
    ts = compute_tearsheet([], eq, initial_capital=100.0)
    assert len(get_all_runs(db)) == 0

    run_id = record_run("BTC-USD", "1d", datetime(2024, 1, 1), datetime(2024, 1, 4),
                        "EMA Crossover", {"fast_period": 9}, ts, db_path=db)
    runs = get_all_runs(db)
    assert len(runs) == 1
    assert runs.iloc[0]["run_id"] == run_id
    assert json.loads(runs.iloc[0]["strategy_params"]) == {"fast_period": 9}


def test_records_risk_params_and_fingerprint(tmp_path):
    from risk.controls import RiskParams
    db = str(tmp_path / "log2.db")
    eq = pd.Series([100.0, 101.0, 102.0], index=pd.date_range("2024-01-01", periods=3))
    ts = compute_tearsheet([], eq, initial_capital=100.0)
    rp = RiskParams(commission_bps=10, slippage_bps=5, stop_loss_atr=2.0, max_leverage=1.0)
    record_run("BTC-USD", "1d", datetime(2024, 1, 1), datetime(2024, 1, 3),
               "EMA Crossover", {"fast_period": 9}, ts,
               risk_params=rp, data_fingerprint="abc123", db_path=db)
    row = get_all_runs(db).iloc[0]
    assert row["commission_bps"] == 10 and row["slippage_bps"] == 5
    assert row["stop_loss_atr"] == 2.0 and row["data_fingerprint"] == "abc123"
