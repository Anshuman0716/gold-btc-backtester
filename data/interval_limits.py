"""
yfinance interval lookback limits and validation.

yfinance imposes maximum lookback windows per interval. This module
centralizes those limits so the dashboard can clamp date pickers and
the data layer can reject invalid requests before making API calls.

Limits (as of yfinance 0.2.x):
    1m  → 7 days
    5m  → 60 days
    15m → 60 days
    30m → 60 days
    1h  → 730 days
    1d  → unlimited

Excluded intervals: 90m, 3mo (not supported in this platform).
"""

from datetime import datetime, timedelta
from typing import Optional


# Maximum lookback in days for each interval.
# None means unlimited (no restriction).
INTERVAL_LIMITS: dict[str, Optional[int]] = {
    "1m": 7,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "1h": 730,
    "1d": None,
}

# Valid intervals for this platform
VALID_INTERVALS: frozenset[str] = frozenset(INTERVAL_LIMITS.keys())

# Human-readable labels for the dashboard
INTERVAL_LABELS: dict[str, str] = {
    "1m": "1 Minute (max 7 days)",
    "5m": "5 Minutes (max 60 days)",
    "15m": "15 Minutes (max 60 days)",
    "30m": "30 Minutes (max 60 days)",
    "1h": "1 Hour (max 730 days)",
    "1d": "1 Day (unlimited)",
}


def validate_date_range(
    interval: str,
    start: datetime,
    end: datetime,
) -> None:
    """Validate that the requested date range is within yfinance limits.

    Args:
        interval: Bar interval (must be in VALID_INTERVALS).
        start: Start of the requested range.
        end: End of the requested range.

    Raises:
        ValueError: If interval is invalid, dates are inverted, or the
                     range exceeds the yfinance lookback limit.
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(
            f"Invalid interval '{interval}'. "
            f"Must be one of: {sorted(VALID_INTERVALS)}"
        )

    if start >= end:
        raise ValueError(
            f"Start date ({start}) must be before end date ({end})."
        )

    max_days = INTERVAL_LIMITS[interval]
    if max_days is not None:
        requested_days = (end - start).days
        if requested_days > max_days:
            raise ValueError(
                f"Interval '{interval}' supports max {max_days} days lookback, "
                f"but {requested_days} days were requested "
                f"({start.date()} to {end.date()}). "
                f"Reduce the date range or use a larger interval."
            )


def get_max_start_date(interval: str, end: datetime) -> datetime:
    """Get the earliest valid start date for a given interval and end date.

    Used by the dashboard to dynamically clamp the date picker.

    Args:
        interval: Bar interval (must be in VALID_INTERVALS).
        end: End of the requested range.

    Returns:
        The earliest valid start datetime. For unlimited intervals (1d),
        returns 2010-01-01 as a practical floor.

    Raises:
        ValueError: If interval is not in VALID_INTERVALS.
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(
            f"Invalid interval '{interval}'. "
            f"Must be one of: {sorted(VALID_INTERVALS)}"
        )

    max_days = INTERVAL_LIMITS[interval]
    if max_days is None:
        # Practical floor for unlimited intervals
        return datetime(2010, 1, 1)

    return end - timedelta(days=max_days)
