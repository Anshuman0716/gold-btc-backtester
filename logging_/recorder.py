"""
Backtest run recorder.

Single function called after every backtest: record_run().
Performs an atomic INSERT of all parameters and metrics into SQLite.
Returns the UUID for the recorded run.

Also provides get_all_runs() for the dashboard history view.
"""

import json
import uuid
from datetime import datetime

import pandas as pd

from analytics.tearsheet import TearsheetResult
from logging_.db import init_db, DEFAULT_DB_PATH


INSERT_RUN_SQL = """
INSERT INTO backtest_runs (
    run_id, timestamp, asset, interval, start_date, end_date,
    strategy_name, strategy_params,
    total_return, sharpe_ratio, sortino_ratio, calmar_ratio,
    max_drawdown, win_rate, profit_factor, num_trades,
    initial_capital, risk_per_trade, stop_loss_atr, take_profit_atr,
    max_drawdown_pct, commission_bps, slippage_bps, max_leverage,
    data_fingerprint
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


def record_run(
    asset: str,
    interval: str,
    start_date: datetime,
    end_date: datetime,
    strategy_name: str,
    strategy_params: dict,
    tearsheet: TearsheetResult,
    risk_params=None,
    data_fingerprint: str = None,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Record a completed backtest run to the audit log.

    Performs a single atomic INSERT. If the insert fails, no partial
    data is written (SQLite transaction guarantee).

    Args:
        asset: Ticker symbol (e.g. "BTC-USD", "GC=F").
        interval: Bar interval (e.g. "1d", "1h").
        start_date: Backtest window start.
        end_date: Backtest window end.
        strategy_name: Human-readable strategy name.
        strategy_params: Dict of all configurable parameters.
        tearsheet: TearsheetResult from compute_tearsheet().
        db_path: Path to SQLite database.

    Returns:
        run_id (UUID string) of the recorded run.
    """
    conn = init_db(db_path)
    run_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    def _rp(attr):
        return getattr(risk_params, attr, None) if risk_params is not None else None

    try:
        conn.execute(INSERT_RUN_SQL, (
            run_id,
            now,
            asset,
            interval,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            strategy_name,
            json.dumps(strategy_params),
            tearsheet.total_return,
            tearsheet.sharpe_ratio,
            tearsheet.sortino_ratio,
            tearsheet.calmar_ratio,
            tearsheet.max_drawdown,
            tearsheet.win_rate,
            tearsheet.profit_factor,
            tearsheet.num_trades,
            _rp("initial_capital"),
            _rp("risk_per_trade"),
            _rp("stop_loss_atr"),
            _rp("take_profit_atr"),
            _rp("max_drawdown_pct"),
            _rp("commission_bps"),
            _rp("slippage_bps"),
            _rp("max_leverage"),
            data_fingerprint,
        ))
        conn.commit()
    finally:
        conn.close()

    return run_id


def get_all_runs(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Retrieve all recorded backtest runs as a DataFrame.

    Used by the dashboard to display run history.

    Args:
        db_path: Path to SQLite database.

    Returns:
        pd.DataFrame with all runs, sorted by timestamp descending.
        Empty DataFrame if no runs exist.
    """
    conn = init_db(db_path)

    try:
        df = pd.read_sql_query(
            "SELECT * FROM backtest_runs ORDER BY timestamp DESC",
            conn,
        )
    finally:
        conn.close()

    return df
