from datetime import datetime, timedelta

import pandas as pd
import pytest

from data.base import DataFetchError
from data.cache import CachedDataSource
from data.interval_limits import get_max_start_date, validate_date_range
from data.yfinance_source import YFinanceSource


class _FakeTicker:
    last_kwargs: dict = {}

    def __init__(self, symbol):
        pass

    def history(self, **kwargs):
        _FakeTicker.last_kwargs = kwargs
        idx = pd.date_range("2024-01-01", "2024-01-05", freq="D")
        return pd.DataFrame(
            {"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 10.0, "Extra": 0},
            index=idx,
        )


def test_end_date_is_inclusive(monkeypatch):
    monkeypatch.setattr("data.yfinance_source.yf.Ticker", _FakeTicker)
    YFinanceSource().fetch_ohlcv("BTC-USD", "1d", datetime(2024, 1, 1), datetime(2024, 1, 5))
    # yfinance treats `end` as exclusive, so we must ask for the day after.
    assert _FakeTicker.last_kwargs["end"] == "2024-01-06"
    assert _FakeTicker.last_kwargs["start"] == "2024-01-01"


def test_output_contract(monkeypatch):
    monkeypatch.setattr("data.yfinance_source.yf.Ticker", _FakeTicker)
    df = YFinanceSource().fetch_ohlcv("BTC-USD", "1d", datetime(2024, 1, 1), datetime(2024, 1, 5))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.is_monotonic_increasing


def test_rejects_bad_inputs():
    src = YFinanceSource()
    with pytest.raises(ValueError):
        src.fetch_ohlcv("AAPL", "1d", datetime(2024, 1, 1), datetime(2024, 2, 1))
    with pytest.raises(ValueError):
        src.fetch_ohlcv("BTC-USD", "3mo", datetime(2024, 1, 1), datetime(2024, 2, 1))


def test_empty_frame_raises(monkeypatch):
    class Empty(_FakeTicker):
        def history(self, **kw):
            return pd.DataFrame()

    monkeypatch.setattr("data.yfinance_source.yf.Ticker", Empty)
    with pytest.raises(DataFetchError):
        YFinanceSource().fetch_ohlcv("BTC-USD", "1d", datetime(2024, 1, 1), datetime(2024, 1, 5))


def test_interval_limits():
    validate_date_range("1d", datetime(2010, 1, 1), datetime(2024, 1, 1))
    with pytest.raises(ValueError):
        validate_date_range("1m", datetime(2024, 1, 1), datetime(2024, 3, 1))
    with pytest.raises(ValueError):
        validate_date_range("1d", datetime(2024, 2, 1), datetime(2024, 1, 1))
    assert get_max_start_date("1h", datetime(2024, 12, 31)) == datetime(2024, 12, 31) - timedelta(days=730)


def test_cache_hits_source_once(tmp_path, btc):
    calls = []

    class Counting(YFinanceSource):
        def fetch_ohlcv(self, *a, **k):
            calls.append(1)
            return btc.head(10)

    src = CachedDataSource(Counting(), cache_dir=str(tmp_path))
    a = src.fetch_ohlcv("BTC-USD", "1d", datetime(2024, 1, 1), datetime(2024, 2, 1))
    b = src.fetch_ohlcv("BTC-USD", "1d", datetime(2024, 1, 1), datetime(2024, 2, 1))
    assert len(calls) == 1
    pd.testing.assert_frame_equal(a, b, check_freq=False)
