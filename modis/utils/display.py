"""
Display utilities for MODIS.

This module provides small helpers for formatting output in the console.
"""


def adjust_time(seconds: float) -> str:
    """Convert a duration in seconds to a human-readable string.

    Automatically selects the most convenient unit — nanoseconds,
    microseconds, milliseconds, seconds, minutes, hours, or days.

    Args:
        seconds (float): Duration in seconds.

    Returns:
        str: Human-readable duration string with the unit appended
        (e.g., ``'2.34 minutes'``, ``'150.00 ms'``).

    Examples:
        >>> adjust_time(0.0000004)
        '400.00 ns'
        >>> adjust_time(3661)
        '1.02 hours'
    """
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} minutes"
    elif seconds < 86400:
        return f"{seconds / 3600:.2f} hours"
    else:
        return f"{seconds / 86400:.2f} days"