"""
SQLite database schema, migration, and connection management.

Single table: backtest_runs. Stores every backtest run with its metrics AND the
risk/cost parameters + a data fingerprint, so a run can be reproduced from its row.
Older databases are upgraded in place via a lightweight ADD COLUMN migration.
"""

import sqlite3

# Default database path
DEFAULT_DB_PATH = "backtest_log.db"

# Core columns (present since v1).
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

# Reproducibility columns added in P1.6. Nullable so pre-existing rows and
# legacy record_run() calls (no risk params) remain valid.
EXTRA_COLUMNS: list[tuple[str, str]] = [
    ("initial_capital", "REAL"),
    ("risk_per_trade", "REAL"),
    ("stop_loss_atr", "REAL"),
    ("take_profit_atr", "REAL"),
    ("max_drawdown_pct", "REAL"),
    ("commission_bps", "REAL"),
    ("slippage_bps", "REAL"),
    ("max_leverage", "REAL"),
    ("data_fingerprint", "TEXT"),
]


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Get a SQLite connection with Row factory enabled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add any missing reproducibility columns to an existing table."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(backtest_runs)").fetchall()}
    for col, decl in EXTRA_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE backtest_runs ADD COLUMN {col} {decl}")


def init_db(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Initialize the database, creating the table and upgrading old schemas.

    Safe to call repeatedly: CREATE TABLE IF NOT EXISTS is idempotent and the
    migration only adds columns that are missing.
    """
    conn = get_connection(db_path)
    conn.execute(CREATE_TABLE_SQL)
    # Fresh DBs: ensure the extra columns exist too.
    for col, decl in EXTRA_COLUMNS:
        try:
            conn.execute(f"ALTER TABLE backtest_runs ADD COLUMN {col} {decl}")
        except sqlite3.OperationalError:
            pass  # already present
    conn.commit()
    return conn
