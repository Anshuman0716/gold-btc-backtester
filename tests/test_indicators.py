import numpy as np
import pandas as pd
import pytest

from indicators.registry import INDICATOR_FUNCTIONS, INDICATOR_PARAMS
from indicators.trend import ema
from indicators.momentum import rsi
from indicators.volatility import atr, bollinger_bands


def test_registry_is_consistent():
    assert set(INDICATOR_FUNCTIONS) == set(INDICATOR_PARAMS)


@pytest.mark.parametrize("name", sorted(INDICATOR_FUNCTIONS))
def test_every_indicator_runs_on_real_data(name, btc):
    out = INDICATOR_FUNCTIONS[name](btc)
    assert len(out) == len(btc)
    assert out.index.equals(btc.index)


def test_ema_tracks_constant_series():
    df = pd.DataFrame({"close": [10.0] * 50})
    assert ema(df, 10).dropna().eq(10.0).all()


def test_rsi_bounds(btc):
    r = rsi(btc, 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_rsi_all_gains_is_high():
    df = pd.DataFrame({"close": np.arange(1.0, 60.0)})
    assert rsi(df, 14).iloc[-1] > 99


def test_atr_positive(btc):
    assert (atr(btc, 14).dropna() > 0).all()


def test_bollinger_ordering(btc):
    bb = bollinger_bands(btc, 20, 2.0).dropna()
    cols = list(bb.columns)
    up = [c for c in cols if "upper" in c.lower()][0]
    mid = [c for c in cols if "mid" in c.lower()][0]
    lo = [c for c in cols if "lower" in c.lower()][0]
    assert (bb[up] >= bb[mid]).all() and (bb[mid] >= bb[lo]).all()


def test_indicators_have_no_lookahead(btc):
    """Value at bar t must not change when future bars are removed."""
    cut = 300
    for name in ("EMA", "RSI", "ATR"):
        f = INDICATOR_FUNCTIONS[name]
        full = f(btc)
        part = f(btc.iloc[:cut])
        np.testing.assert_allclose(full.iloc[:cut].values, part.values, equal_nan=True)
