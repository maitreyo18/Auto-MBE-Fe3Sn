"""Looks up and analyzes the lab's real experiments with pandas: filtering,
sorting, finding the runs closest to given conditions, comparing specific runs,
summary statistics and correlations. No models are involved.

Called by an LLM, so it tolerates loose input (column names like "film quality",
numbers sent as strings) and reports problems under "error" instead of raising,
so the LLM can correct itself. Every number returned comes from the CSV files.
"""
import difflib
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))

from lab_models import TEST_FILE, TRAIN_FILE  # noqa: E402

DEFAULT_COLUMNS = ["growthtime", "filamentpower", "flux_ratio", "Substrate_quality", "EDS_ratio", "RHEED_Quality_Film"]
DATASETS = ("all", "initial", "later")
# EDS was never measured for these runs; the source data holds a statistical estimate.
IMPUTED_EDS_RUNS = {31, 32, 35}
FEW_ROWS_FOR_CORRELATION = 8
ALIASES = {
    "filmquality": "RHEED_Quality_Film", "film": "RHEED_Quality_Film", "quality": "RHEED_Quality_Film",
    "rheedquality": "RHEED_Quality_Film", "rheedqualityfilm": "RHEED_Quality_Film", "rheed": "RHEED_Quality_Film",
    "eds": "EDS_ratio", "edsratio": "EDS_ratio", "stoichiometry": "EDS_ratio", "fesn": "EDS_ratio", "composition": "EDS_ratio",
    "power": "filamentpower", "filament": "filamentpower", "filamentpower": "filamentpower",
    "flux": "flux_ratio", "fluxratio": "flux_ratio",
    "time": "growthtime", "growthtime": "growthtime",
    "substrate": "Substrate_quality", "substratequality": "Substrate_quality",
    "runs": "run", "runnumber": "run",
}


class AnalysisError(ValueError):
    """A problem with the request; its message is returned to the LLM."""


def _key(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def resolve_column(name, columns, mapping=None):
    """Exact, then case/punctuation-insensitive, then alias match."""
    if name in columns:
        return name
    by_key = {_key(c): c for c in columns}
    key = _key(name)
    match = by_key.get(key) or (ALIASES.get(key) if ALIASES.get(key) in columns else None)
    if match is None:
        close = difflib.get_close_matches(str(name), [str(c) for c in columns], n=3, cutoff=0.5)
        hint = f" Did you mean {close}?" if close else ""
        raise AnalysisError(f"unknown column {name!r}.{hint}")
    if mapping is not None:
        mapping[name] = match
    return match


def _number(value, label):
    """Accepts ints, floats and numeric strings; rejects bools, blanks and non-finite values."""
    if isinstance(value, bool) or value is None:
        raise AnalysisError(f"{label} must be a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise AnalysisError(f"{label} must be a number, got {value!r}") from None
    if not math.isfinite(number):
        raise AnalysisError(f"{label} must be finite, got {value!r}")
    return number


def _integer(value, label):
    number = _number(value, label)
    if number != int(number):
        raise AnalysisError(f"{label} must be a whole number, got {value!r}")
    return int(number)


def load_experiments(dataset="all"):
    dataset = str(dataset).strip().lower()
    if dataset not in DATASETS:
        raise AnalysisError(f"dataset must be one of {list(DATASETS)}, got {dataset!r}")
    frames = []
    for name, path in (("initial", TRAIN_FILE), ("later", TEST_FILE)):
        if dataset in ("all", name):
            if not Path(path).exists():
                raise AnalysisError(f"data file not found: {path}")
            frames.append(pd.read_csv(path).drop(columns=["eds_imputed"], errors="ignore").assign(dataset=name))
    return pd.concat(frames, ignore_index=True)


def _clean(value):
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(value) else round(float(value), 4)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _records(df):
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def _clean_table(table):
    return {col: {k: _clean(v) for k, v in values.items()} for col, values in table.to_dict().items()}


def _resolve_conditions(conditions, label, columns, numeric, mapping):
    """{column: value} with resolved numeric columns and validated numbers."""
    out = {}
    for name, value in (conditions or {}).items():
        col = resolve_column(name, columns, mapping)
        if col not in numeric:
            raise AnalysisError(f"{label} column {col!r} is not numeric")
        out[col] = value
    return out


def _apply_filters(df, filters, min_values, max_values, columns, numeric, mapping):
    for col, bounds in _resolve_conditions(filters, "filters", columns, numeric, mapping).items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise AnalysisError(f"filter for {col} must be [min, max], got {bounds!r}")
        lo, hi = _number(bounds[0], f"{col} minimum"), _number(bounds[1], f"{col} maximum")
        if lo > hi:
            raise AnalysisError(f"filter for {col} has minimum {lo} above maximum {hi}")
        df = df[df[col].between(lo, hi)]
    for col, value in _resolve_conditions(min_values, "min_values", columns, numeric, mapping).items():
        df = df[df[col] >= _number(value, f"{col} minimum")]
    for col, value in _resolve_conditions(max_values, "max_values", columns, numeric, mapping).items():
        df = df[df[col] <= _number(value, f"{col} maximum")]
    return df


def _closest_runs(df, closest_to, all_rows, shown, top_n):
    cols = list(closest_to)
    target = pd.Series(closest_to, dtype=float)
    # Standardized so columns with large values, like growth time, don't dominate.
    spread = all_rows[cols].std().replace(0, 1).fillna(1)
    distance = np.sqrt((((df[cols] - target) / spread) ** 2).sum(axis=1, skipna=False))
    usable = df.assign(distance=distance).dropna(subset=["distance"])
    ranked = usable.sort_values(["distance", "run"]).head(top_n)
    rows = _records(ranked[shown + [c for c in cols if c not in shown] + ["distance"]])
    for row in rows:
        row["difference_from_requested"] = {c: _clean(row[c] - closest_to[c]) for c in cols}
    return {
        "rows": rows,
        "runs": set(ranked["run"]),
        "skipped_blank": len(df) - len(usable),
        "outside": {
            c: {"requested": closest_to[c], "observed_min": _clean(all_rows[c].min()), "observed_max": _clean(all_rows[c].max())}
            for c in cols if not all_rows[c].min() <= closest_to[c] <= all_rows[c].max()
        },
    }


def analyze(question=None, dataset="all", columns=None, filters=None, min_values=None, max_values=None,
            sort_by=None, ascending=False, top_n=5, closest_to=None, compare_runs=None):
    """Returns only what was asked for, plus summary statistics and correlations of
    the rows left after filtering. Problems come back under "error"."""
    try:
        return _analyze(question, dataset, columns, filters, min_values, max_values, sort_by, ascending,
                        top_n, closest_to, compare_runs)
    except AnalysisError as exc:
        try:
            available = list(load_experiments("all").columns)
        except AnalysisError:
            available = []
        return {"error": str(exc), "available_columns": available}


def _analyze(question, dataset, columns, filters, min_values, max_values, sort_by, ascending,
             top_n, closest_to, compare_runs):
    all_rows = load_experiments(dataset)
    available = list(all_rows.columns)
    numeric = list(all_rows.select_dtypes(include="number").columns)
    top_n = _integer(top_n, "top_n")
    if top_n <= 0:
        raise AnalysisError(f"top_n must be a positive integer, got {top_n}")

    mapping = {}
    columns = list(dict.fromkeys(resolve_column(c, available, mapping) for c in columns)) if columns else list(DEFAULT_COLUMNS)
    sort_col = resolve_column(sort_by, available, mapping) if sort_by else None
    closest = {
        col: _number(value, f"closest_to {col}")
        for col, value in _resolve_conditions(closest_to, "closest_to", available, numeric, mapping).items()
    }
    runs_wanted = [_integer(r, "run number") for r in (compare_runs or [])]

    df = _apply_filters(all_rows, filters, min_values, max_values, available, numeric, mapping)
    id_cols = ["run", "dataset"]
    shown = list(dict.fromkeys(id_cols + columns))
    result = {"question": question, "dataset": str(dataset).strip().lower(), "n_runs": len(df), "columns": columns,
              "available_columns": available}
    if mapping:
        result["columns_interpreted_as"] = mapping
    if df.empty:
        result["note"] = "No experiments match these filters."
        return result

    used_runs = set()
    if sort_col:
        usable = df.dropna(subset=[sort_col])
        # run as a tie-break keeps results repeatable.
        ranked = usable.sort_values([sort_col, "run"], ascending=[bool(ascending), True]).head(top_n)
        result["top_runs"] = _records(ranked[list(dict.fromkeys(shown + [sort_col]))])
        result["sorted_by"] = sort_col
        used_runs |= set(ranked["run"])
        if len(usable) < len(df):
            result["sort_note"] = f"{len(df) - len(usable)} run(s) left out: blank {sort_col}."

    if closest:
        found = _closest_runs(df, closest, all_rows, shown, top_n)
        result["closest_runs"] = found["rows"]
        result["closest_to"] = closest
        result["distance_note"] = "Distance is in standard deviations across the requested columns, not a physical unit."
        used_runs |= found["runs"]
        if found["skipped_blank"]:
            result["closest_note"] = f"{found['skipped_blank']} run(s) left out: blank values in the requested columns."
        if found["outside"]:
            result["outside_observed_range"] = found["outside"]
            result["outside_note"] = "These requested values are outside anything grown, so the closest runs may not be close."

    if runs_wanted:
        chosen = df[df["run"].isin(runs_wanted)].sort_values("run")
        result["compared_runs"] = _records(chosen[shown])
        not_found = sorted(set(runs_wanted) - set(int(r) for r in chosen["run"]))
        if not_found:
            result["runs_not_found"] = not_found
        used_runs |= set(chosen["run"])
        if len(chosen) >= 2:
            num = [c for c in shown if c in numeric and c != "run"]
            result["compared_spread"] = {c: _clean(chosen[c].max() - chosen[c].min()) for c in num}

    stats_cols = [c for c in columns if c in numeric and c != "run"]
    if stats_cols:
        result["summary_statistics"] = _clean_table(df[stats_cols].describe())
        if len(df) >= 2:
            result["correlations"] = _clean_table(df[stats_cols].corr())
        if len(df) < FEW_ROWS_FOR_CORRELATION:
            result["statistics_note"] = f"Only {len(df)} run(s): statistics and correlations are not reliable."

    relevant = used_runs if used_runs else set(df["run"])
    flagged = sorted(int(r) for r in set(relevant) & IMPUTED_EDS_RUNS)
    if flagged and "EDS_ratio" in set(columns) | set(closest) | {sort_col}:
        result["eds_imputed_runs"] = flagged
        result["eds_note"] = "EDS_ratio was not measured for these runs; their values are statistical estimates."
    return result
