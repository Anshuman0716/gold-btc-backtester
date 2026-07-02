"""
Audit logging layer -- SQLite-based backtest run recording.

Demonstrates institutional audit-trail habits: every backtest run
is permanently stored with its parameters and results, making
any run fully reproducible.
"""

from logging_.db import init_db, get_connection
from logging_.recorder import record_run, get_all_runs

__all__ = [
    "init_db",
    "get_connection",
    "record_run",
    "get_all_runs",
]
