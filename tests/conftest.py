"""Shared fixtures. Everything runs offline from committed parquet files."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def btc() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / "BTC_USD_1d_2022-01-01_2024-12-31.parquet")


@pytest.fixture(scope="session")
def gold() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / "GC_F_1d_2022-01-01_2024-12-31.parquet")


def make_ohlc(opens, highs=None, lows=None, closes=None) -> pd.DataFrame:
    """Build a small hand-crafted OHLCV frame on a daily index."""
    n = len(opens)
    closes = closes if closes is not None else opens
    highs = highs if highs is not None else [max(o, c) for o, c in zip(opens, closes)]
    lows = lows if lows is not None else [min(o, c) for o, c in zip(opens, closes)]
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": [1.0] * n},
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )
