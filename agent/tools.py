import sys
from pathlib import Path
from typing import Dict, List, Optional
from langchain_core.tools import tool

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "RHEED_images"))
sys.path.insert(0, str(THIS_DIR.parent / "models"))

from feature_extractor import extract_rheed_quality
from al_loop import run_al_loop
from experiments import analyze
from doe import run_doe, default_ranges as doe_default_ranges
from validation import require_positive_int


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
def run_active_learning_loop(n_iterations: int = 5, n_initial: int = 10) -> list:
    """Runs a multi-dimensional Expected Improvement active-learning loop
    against the lab's ground-truth models (film-quality MLP and stoichiometry
    RFR), which provide the outcome of every suggested experiment.

    It starts from n_initial of the 17 initial experiments (default 10). Each
    iteration fits leave-one-out copies of the same two model types on the rows
    observed so far, picks the next filamentpower/flux_ratio via Expected
    Improvement on Score_Holistic (65% film quality + 35% closeness of EDS_ratio
    to 3.0) using the spread across those folds, asks the ground-truth models for
    the outcome, appends it, and repeats. Each result reports how well the AL
    models track the ground truth. Defaults to 5 iterations.

    This is a simulation: outcomes come from models, not real growths. Saves
    each iteration's models and a plot under agent/al_runs/iteration_N/. Use when
    the user asks to optimize, explore, or simulate the next experiments -- not
    for predicting a single known configuration."""
    require_positive_int(n_iterations, "n_iterations")
    return run_al_loop(n_iterations=n_iterations, n_initial=n_initial)


@tool
def analyze_previous_experiments(
    question: Optional[str] = None,
    dataset: str = "all",
    columns: Optional[List[str]] = None,
    filters: Optional[Dict[str, List[float]]] = None,
    min_values: Optional[Dict[str, float]] = None,
    max_values: Optional[Dict[str, float]] = None,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    top_n: int = 5,
    closest_to: Optional[Dict[str, float]] = None,
    compare_runs: Optional[List[int]] = None,
) -> dict:
    """Reads the lab's real experiments with pandas and analyzes them as needed.
    Use whenever the user asks about existing, previous, past or historical runs,
    as opposed to predicting a new configuration. Combine the options as needed.

    Args:
        question: the user's request, for context.
        dataset: "all" (21 growths, default), "initial" (the first 17) or "later" (the last 4).
        columns: columns to show. Default: growthtime, filamentpower, flux_ratio,
            Substrate_quality, EDS_ratio, RHEED_Quality_Film. Others, such as the raw
            substrate numbers, are listed in available_columns.
        filters: keep rows within ranges, e.g. {"filamentpower": [3.0, 4.0]}.
        min_values: keep rows at or above a value, e.g. {"RHEED_Quality_Film": 60}.
        max_values: keep rows at or below a value, e.g. {"growthtime": 300}.
        sort_by: rank by this column, e.g. RHEED_Quality_Film for the best films.
        ascending: True for lowest first.
        top_n: how many rows to return for sort_by and closest_to.
        closest_to: find the runs nearest these conditions, e.g. {"filamentpower": 3.5,
            "flux_ratio": 0.6} for similar past growths, or {"EDS_ratio": 3.0} for
            the runs closest to ideal stoichiometry.
        compare_runs: run numbers to show side by side, e.g. [14, 24].

    Returns the requested rows (top_runs, closest_runs, compared_runs) plus summary
    statistics and correlations for the rows left after filtering. If "error" is
    returned, fix the request and retry. Always pass on any note: eds_note (those EDS
    values are estimates, not measurements), outside_note, statistics_note,
    columns_interpreted_as and runs_not_found."""
    return analyze(
        question=question, dataset=dataset, columns=columns, filters=filters, min_values=min_values, max_values=max_values, sort_by=sort_by,
        ascending=ascending, top_n=top_n, closest_to=closest_to, compare_runs=compare_runs,
    )


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
    NOT for a single-point prediction (use ask_model_builder) and NOT for the
    iterative Expected-Improvement search (use run_active_learning_loop).

    Call get_doe_suggested_ranges first and ask the user to confirm or
    override each parameter's (min, max) range before calling this. Any
    bound left as None falls back to that data-driven default. n_runs
    defaults to 12 (quadratic terms + 2, the minimum for a stable fit) and
    is auto-raised to that floor if lower.

    Selects n_runs design points via coordinate-exchange D-optimal search,
    evaluates each with the ground-truth film-quality MLP and stoichiometry RFR (mean
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


@tool
def ask_model_builder(request: str) -> str:
    """Delegates to the model-builder agent. It serves ALL predictions of film
    quality (RHEED_Quality_Film) and stoichiometry (EDS_ratio) from the lab's two
    saved models, trains, saves and compares on-the-fly RFR, GPR, KRR and MLP models,
    uses or optimizes any of them, and independently searches for better models when
    given an open-ended task (e.g. "find a better film-quality model in 10 minutes").
    If the user asks to save or write out results, include that in the request: it
    writes them to ./model_builder_results/ only when asked.

    Film quality needs growthtime, filamentpower, flux_ratio plus ALL 7 substrate
    numbers that rheed_quality returns: if a substrate image is given, call
    rheed_quality first and put every returned number in the request. EDS ratio
    needs only growthtime, filamentpower, flux_ratio and Substrate_quality.

    Pass the user's wording as-is in `request`, including their answers to any
    question it asked (it remembers the conversation). Relay its questions to the
    user verbatim. Its reply contains tables and numbers: relay it in full."""
    from model_agent import ask

    return ask(request)


TOOLS = [
    rheed_quality,
    run_active_learning_loop,
    analyze_previous_experiments,
    get_doe_suggested_ranges,
    run_doe_experiment,
    ask_model_builder,
]
