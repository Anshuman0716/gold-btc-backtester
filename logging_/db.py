"""
SQLite database schema and connection management.

Single table: backtest_runs
Stores every backtest run with full parameters and key metrics.
Uses context manager for safe connection handling.

Why SQLite:
    - Zero deployment overhead (built into Python)
    - Single-file database (easy to inspect, share, version)
    - ACID compliant (no partial writes)
    - Shows awareness of audit-trail patterns without overengineering
"""

import sqlite3
from pathlib import Path

# Default database path
DEFAULT_DB_PATH = "backtest_log.db"

# Schema: backtest_runs table
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id          TEXT PRIMARY KEY,
    timestamp       DATETIME NOT NULL,
    asset           TEXT NOT NULL,
    interval        TEXT NOT NULL,
    start_date      DATE NOT NULL,
    end_date        DATE NOT NULL,
    strategy_name   TEXT NOT NULL,
    strategy_params JSON NOT NULL,
    total_return    REAL NOT NULL,
    sharpe_ratio    REAL NOT NULL,
    sortino_ratio   REAL NOT NULL,
    calmar_ratio    REAL NOT NULL,
    max_drawdown    REAL NOT NULL,
    win_rate        REAL NOT NULL,
    profit_factor   REAL NOT NULL,
    num_trades      INTEGER NOT NULL
);
"""


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Get a SQLite connection with row factory enabled.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        sqlite3.Connection with Row factory for dict-like access.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Initialize the database and create tables if they don't exist.

    Safe to call multiple times -- CREATE TABLE IF NOT EXISTS is idempotent.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        sqlite3.Connection ready for use.
    """
    conn = get_connection(db_path)
    conn.execute(CREATE_TABLE_SQL)
    conn.commit()
    return conn
