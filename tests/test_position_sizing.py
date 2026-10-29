import pytest

from risk.position_sizing import atr_position_size


def test_docstring_example():
    assert atr_position_size(100_000, 0.01, 2_000, 2.0) == pytest.approx(0.25)


def test_scales_inversely_with_atr():
    assert atr_position_size(1e5, 0.01, 100) == pytest.approx(
        2 * atr_position_size(1e5, 0.01, 200)
    )


@pytest.mark.parametrize("kw", [
    dict(account_value=0, risk_per_trade=0.01, atr_value=1),
    dict(account_value=1, risk_per_trade=0, atr_value=1),
    dict(account_value=1, risk_per_trade=1.5, atr_value=1),
    dict(account_value=1, risk_per_trade=0.01, atr_value=0),
    dict(account_value=1, risk_per_trade=0.01, atr_value=1, atr_multiplier=0),
])
def test_invalid_inputs(kw):
    with pytest.raises(ValueError):
        atr_position_size(**kw)
