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
