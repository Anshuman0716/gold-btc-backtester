"""
ATR-based position sizing.

Core institutional concept: position size scales INVERSELY with volatility.
When ATR is high (volatile market), position size shrinks to control risk.
When ATR is low (calm market), position size grows.

This ensures that the dollar risk per trade stays approximately constant
regardless of market conditions.

Formula:
    position_size = (account_value * risk_per_trade) / (atr * atr_multiplier)

Example:
    Account: $100,000, Risk per trade: 1%, ATR: $2,000, Multiplier: 2
    Position size = (100,000 * 0.01) / (2,000 * 2) = 0.25 BTC
    Dollar risk = 0.25 * 2,000 * 2 = $1,000 (exactly 1% of account)
"""


def atr_position_size(
    account_value: float,
    risk_per_trade: float,
    atr_value: float,
    atr_multiplier: float = 2.0,
) -> float:
    """Calculate position size based on ATR.

    Args:
        account_value: Current portfolio value in dollars.
        risk_per_trade: Fraction of account to risk per trade (e.g. 0.01 = 1%).
        atr_value: Current ATR value (same unit as price).
        atr_multiplier: Multiplier for ATR to set stop distance (default 2.0).

    Returns:
        Position size in units of the asset (e.g. number of BTC or
        number of Gold contracts).

    Raises:
        ValueError: If any input is invalid (negative, zero where not allowed).
    """
    if account_value <= 0:
        raise ValueError(f"account_value must be > 0, got {account_value}")
    if not (0 < risk_per_trade <= 1):
        raise ValueError(
            f"risk_per_trade must be in (0, 1], got {risk_per_trade}"
        )
    if atr_value <= 0:
        raise ValueError(f"atr_value must be > 0, got {atr_value}")
    if atr_multiplier <= 0:
        raise ValueError(f"atr_multiplier must be > 0, got {atr_multiplier}")

    dollar_risk = account_value * risk_per_trade
    stop_distance = atr_value * atr_multiplier
    position_size = dollar_risk / stop_distance

    return position_size
