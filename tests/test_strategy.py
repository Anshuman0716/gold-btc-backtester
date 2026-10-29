import pytest

from strategy.base import VALID_SIGNALS
from strategy.ema_crossover import EMACrossover


def test_rejects_inverted_periods():
    with pytest.raises(ValueError):
        EMACrossover(fast_period=21, slow_period=9)


def test_signals_valid_and_aligned(btc):
    s = EMACrossover(9, 21).generate_signals(btc)
    assert s.index.equals(btc.index)
    assert set(s.unique()) <= VALID_SIGNALS
    assert s.iloc[0] == "HOLD"
    assert (s == "BUY").sum() > 0 and (s == "SELL").sum() > 0


def test_signals_have_no_lookahead(btc):
    strat = EMACrossover(9, 21)
    full = strat.generate_signals(btc)
    part = strat.generate_signals(btc.iloc[:400])
    assert (full.iloc[:400].values == part.values).all()


def test_params_round_trip():
    assert EMACrossover(5, 15).get_params() == {"fast_period": 5, "slow_period": 15}
