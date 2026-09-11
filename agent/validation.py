"""Shared input-validation helpers. All raise plain ValueError with a
message safe to surface directly to the user through the LLM."""
import math


def require_finite(value, name):
    if value is None or not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def require_positive(value, name):
    value = require_finite(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def require_positive_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def require_range(lo, hi, name):
    lo = require_finite(lo, f"{name}_min")
    hi = require_finite(hi, f"{name}_max")
    if lo >= hi:
        raise ValueError(f"{name} range is invalid: min ({lo}) must be < max ({hi})")
    return lo, hi


def require_nonempty(seq, name):
    if seq is None or len(seq) == 0:
        raise ValueError(f"{name} must not be empty")
    return seq


def require_columns(rows, required_cols, name):
    require_nonempty(rows, name)
    missing_report = []
    for i, row in enumerate(rows):
        missing = [c for c in required_cols if c not in row]
        if missing:
            missing_report.append(f"row {i} missing {missing}")
    if missing_report:
        raise ValueError(f"{name} rows are missing required columns: {'; '.join(missing_report)}")
    return rows
