import sys
from pathlib import Path
from typing import List, Dict, Optional
from langchain_core.tools import tool
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "RHEED_images"))
sys.path.insert(0, str(THIS_DIR.parent / "models"))

from feature_extractor import extract_rheed_quality
from predict import predict_with_uncertainty
from al_loop import run_al_loop
from doe import run_doe, default_ranges as doe_default_ranges
from validation import require_finite, require_positive_int

DATA_PATH = THIS_DIR.parent / "data" / "train_compiled.csv"
PARAM_COLS = [
    "growthtime",
    "filamentpower",
    "flux_ratio",
    "Substrate_quality",
    "EDS_ratio",
    "RHEED_Quality_Film",
]


@tool
def rheed_quality(image_path: str) -> dict:
    """Extracts substrate quality from a RHEED image.
    Use when the user gives an image path. Returns Substrate_quality (0-100)
    plus raw diffraction features."""
    path = Path(image_path)
    if not path.is_absolute():
        path = (THIS_DIR / path).resolve()
    return extract_rheed_quality(path)


@tool
def predict_quality(growthtime: float, filamentpower: float, flux_ratio: float, substrate_quality: float) -> dict:
    """Predicts film quality and stoichiometry from growth parameters.
    Use when growthtime, filamentpower, flux_ratio, and substrate_quality are
    all known as numbers. If substrate_quality is not given but an image is,
    call rheed_quality first to obtain it.
    Returns RHEED_Quality_Film and EDS_ratio (the raw Fe:Sn stoichiometry
    ratio -- target 3.0), both mean model predictions."""
    growthtime = require_finite(growthtime, "growthtime")
    filamentpower = require_finite(filamentpower, "filamentpower")
    flux_ratio = require_finite(flux_ratio, "flux_ratio")
    substrate_quality = require_finite(substrate_quality, "substrate_quality")
    X = [[growthtime, filamentpower, flux_ratio, substrate_quality]]
    eds_mean, _ = predict_with_uncertainty(X, "EDS_ratio")
    film_mean, _ = predict_with_uncertainty(X, "RHEED_Quality_Film")
    return {
        "EDS_ratio": round(float(eds_mean[0]), 3),
        "RHEED_Quality_Film": round(float(film_mean[0]), 2),
    }


@tool
def run_active_learning_loop(initial_data: List[Dict[str, float]], n_iterations: int = 5) -> list:
    """Runs a multi-dimensional Expected Improvement active-learning loop,
    starting from a dataset given directly in the prompt (does not use any
    pre-trained checkpoint). Each row of initial_data must have: growthtime,
    filamentpower, flux_ratio, Substrate_quality, EDS_ratio,
    RHEED_Quality_Film.

    Each iteration: builds a single composite Score_Holistic target (65%
    RHEED film quality + 35% closeness of EDS_ratio to the ideal
    stoichiometry of 3.0, both on a 0-100 scale), fits a fresh RF on that one
    target, and picks the next filamentpower/flux_ratio via Expected
    Improvement using the forest's per-tree (bootstrap) std. Takes the
    forest's own predicted mean as the outcome for that point, appends it to
    the dataset, and repeats. Defaults to 5 iterations unless the user says
    otherwise.

    Saves a plot and the fold models used for every iteration under
    agent/al_runs/iteration_N/. Use when the user asks to optimize, explore,
    or suggest the next experiment(s) -- not for predicting a single known
    configuration."""
    require_positive_int(n_iterations, "n_iterations")
    return run_al_loop(initial_data, n_iterations=n_iterations)


@tool
def analyze_previous_experiments(
    question: Optional[str] = None,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    top_n: int = 5,
) -> dict:
    """Looks up and analyzes previously run MBE experiments recorded in
    data/train_compiled.csv. Use this whenever the user asks about existing,
    previous, prior, historical, or already-present experimental data/runs
    (as opposed to predicting a new, hypothetical configuration).

    Restricts analysis to these six parameters unless the user explicitly
    asks about something else: growthtime, filamentpower, flux_ratio,
    Substrate_quality, EDS_ratio, RHEED_Quality_Film.

    Args:
        question: free-text description of what the user wants (used only
            for context in the returned payload -- the actual filtering is
            done by sort_by/top_n; pass the raw user request here).
        sort_by: one of the six parameter names above to rank runs by
            (e.g. "RHEED_Quality_Film" to find the best-quality runs).
        ascending: sort ascending instead of descending (e.g. True to find
            runs with the lowest EDS_ratio deviation from ideal).
        top_n: number of top rows to return when sort_by is given.

    Returns summary statistics (count, mean, std, min, max, quartiles) for
    each of the six parameters, their pairwise correlations, and -- if
    sort_by is given -- the top_n ranked runs on those six parameters plus
    run/File identifiers."""
    df = pd.read_csv(DATA_PATH)
    cols = [c for c in PARAM_COLS if c in df.columns]
    subset = df[["run"] + cols] if "run" in df.columns else df[cols]

    result = {
        "question": question,
        "n_runs": len(df),
        "parameters": cols,
        "summary_statistics": subset[cols].describe().round(4).to_dict(),
        "correlations": subset[cols].corr().round(4).to_dict(),
    }

    if sort_by:
        if sort_by not in cols:
            result["error"] = f"sort_by must be one of {cols}"
        else:
            ranked = subset.sort_values(sort_by, ascending=ascending).head(top_n)
            result["top_runs"] = ranked.to_dict(orient="records")

    return result


@tool
def get_doe_suggested_ranges() -> dict:
    """Returns the min/max of growthtime, filamentpower, and flux_ratio
    observed in data/train_compiled.csv. Use this FIRST whenever the user
    asks to run a DoE / design of experiments, so you can propose these as
    default ranges and ask the user to confirm or override them (per
    parameter) before calling run_doe_experiment."""
    ranges = doe_default_ranges()
    return {p: {"min": lo, "max": hi} for p, (lo, hi) in ranges.items()}


@tool
def run_doe_experiment(
    growthtime_min: Optional[float] = None,
    growthtime_max: Optional[float] = None,
    filamentpower_min: Optional[float] = None,
    filamentpower_max: Optional[float] = None,
    flux_ratio_min: Optional[float] = None,
    flux_ratio_max: Optional[float] = None,
    n_runs: int = 12,
) -> dict:
    """Runs an autonomous D-optimal Design of Experiments (DoE) over
    growthtime, filamentpower, and flux_ratio to jointly optimize film
    quality (RHEED_Quality_Film) and stoichiometry (EDS_ratio). Use ONLY
    when the user explicitly asks for a DoE, "design of experiments",
    "D-optimal design", or an optimal set of experiments to run next --
    NOT for a single-point prediction (use predict_quality) and NOT for the
    iterative Expected-Improvement search (use run_active_learning_loop).

    Call get_doe_suggested_ranges first and ask the user to confirm or
    override each parameter's (min, max) range before calling this. Any
    bound left as None falls back to that data-driven default. n_runs
    defaults to 12 (quadratic terms + 2, the minimum for a stable fit) and
    is auto-raised to that floor if lower.

    Selects n_runs design points via coordinate-exchange D-optimal search,
    evaluates each with the film-quality/stoichiometry surrogates (mean
    predictions only, combined into the same Score_Holistic as the AL loop),
    fits a quadratic response surface, and returns design_table (every run's
    predictions), quadratic_fit (coefficients with std errors, residual std
    dev, 3-sigma, R-squared), and optimum (the best growthtime/filamentpower/
    flux_ratio within the given ranges)."""
    ranges = {
        "growthtime": (growthtime_min, growthtime_max) if growthtime_min is not None or growthtime_max is not None else None,
        "filamentpower": (filamentpower_min, filamentpower_max) if filamentpower_min is not None or filamentpower_max is not None else None,
        "flux_ratio": (flux_ratio_min, flux_ratio_max) if flux_ratio_min is not None or flux_ratio_max is not None else None,
    }
    defaults = doe_default_ranges()
    resolved = {}
    for p, bounds in ranges.items():
        if bounds is None:
            resolved[p] = defaults[p]
        else:
            lo = bounds[0] if bounds[0] is not None else defaults[p][0]
            hi = bounds[1] if bounds[1] is not None else defaults[p][1]
            resolved[p] = (lo, hi)
    return run_doe(ranges=resolved, n_runs=n_runs)


TOOLS = [
    rheed_quality,
    predict_quality,
    run_active_learning_loop,
    analyze_previous_experiments,
    get_doe_suggested_ranges,
    run_doe_experiment,
]
