"""D-optimal Design of Experiments over growthtime/filamentpower/flux_ratio.
Evaluates each design point with the film-quality/stoichiometry surrogates,
combines into the same Score_Holistic used by al_loop.py, fits a quadratic
response surface, and returns the optimum. Substrate_quality is held fixed
at the historical median (not a DoE parameter)."""
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "models"))

from predict import predict_with_uncertainty  # noqa: E402
from validation import require_range, require_positive_int  # noqa: E402

DATA_PATH = THIS_DIR.parent / "data" / "train_compiled.csv"

DOE_PARAMS = ["growthtime", "filamentpower", "flux_ratio"]
FIXED_FEATURE = "Substrate_quality"

# Same composite objective as agent/al_loop.py -- kept identical on purpose.
FILM_WEIGHT = 0.65
EDS_WEIGHT = 0.35
EDS_TARGET_RATIO = 3.0
EDS_SIGMA = 0.5

CANDIDATE_RESOLUTION = 15
DEFAULT_N_RUNS = 12
N_RANDOM_STARTS = 10
N_EXCHANGE_CANDIDATES = 200

QUADRATIC_TERM_NAMES = [
    "intercept",
    "growthtime", "filamentpower", "flux_ratio",
    "growthtime^2", "filamentpower^2", "flux_ratio^2",
    "growthtime*filamentpower", "growthtime*flux_ratio", "filamentpower*flux_ratio",
]


def stoichiometry_score(eds_ratio):
    """Same 0-100 closeness-to-target(3.0) transform as al_loop.py."""
    eds_ratio = np.asarray(eds_ratio, dtype=float)
    return np.exp(-((eds_ratio - EDS_TARGET_RATIO) ** 2) / (2 * EDS_SIGMA**2)) * 100


def default_ranges():
    """Min/max of each DoE parameter from the historical dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"training data not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    missing = [p for p in DOE_PARAMS if p not in df.columns]
    if missing:
        raise ValueError(f"{DATA_PATH} is missing required columns: {missing}")
    return {p: (float(df[p].min()), float(df[p].max())) for p in DOE_PARAMS}


def _quadratic_design_matrix(X):
    """X: (n, 3) normalized to [-1, 1]. Returns the (n, 10) quadratic model
    matrix [1, x1, x2, x3, x1^2, x2^2, x3^2, x1*x2, x1*x3, x2*x3]."""
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return np.column_stack([
        np.ones_like(x1), x1, x2, x3,
        x1**2, x2**2, x3**2,
        x1 * x2, x1 * x3, x2 * x3,
    ])


def _normalize(X, lo, hi):
    return 2 * (X - lo) / (hi - lo) - 1


def _denormalize(X_norm, lo, hi):
    return (X_norm + 1) / 2 * (hi - lo) + lo


def _build_candidate_pool(ranges, resolution=CANDIDATE_RESOLUTION):
    grids = [np.linspace(ranges[p][0], ranges[p][1], resolution) for p in DOE_PARAMS]
    return np.array(list(product(*grids)))


def d_optimal_select(candidates_norm, n_runs, n_random_starts=N_RANDOM_STARTS, seed=42):
    """Coordinate-exchange D-optimal selection: maximizes det(F'F) over a
    candidate pool, restarted from several random subsets."""
    rng = np.random.default_rng(seed)
    F_all = _quadratic_design_matrix(candidates_norm)
    n_cand, n_terms = F_all.shape
    n_runs = max(n_runs, n_terms + 2)
    if n_runs > n_cand:
        raise ValueError(
            f"n_runs ({n_runs}) exceeds candidate pool size ({n_cand}); "
            "increase resolution or widen the parameter ranges"
        )
    reg = 1e-10 * np.eye(n_terms)

    best_det = -np.inf
    best_idx = None

    for _ in range(n_random_starts):
        idx = list(rng.choice(n_cand, size=n_runs, replace=False))
        improved = True
        while improved:
            improved = False
            for i in range(n_runs):
                F = F_all[idx]
                cur_det = np.linalg.det(F.T @ F + reg)
                probe = rng.choice(n_cand, size=min(N_EXCHANGE_CANDIDATES, n_cand), replace=False)
                best_j, best_j_det = None, cur_det
                for j in probe:
                    if j in idx:
                        continue
                    trial = idx.copy()
                    trial[i] = j
                    F_trial = F_all[trial]
                    det_trial = np.linalg.det(F_trial.T @ F_trial + reg)
                    if det_trial > best_j_det:
                        best_j_det, best_j = det_trial, j
                if best_j is not None:
                    idx[i] = best_j
                    improved = True

        F = F_all[idx]
        final_det = np.linalg.det(F.T @ F + reg)
        if final_det > best_det:
            best_det, best_idx = final_det, idx.copy()

    return best_idx


def run_doe(ranges=None, n_runs=DEFAULT_N_RUNS, resolution=CANDIDATE_RESOLUTION, seed=42):
    """Runs the full D-optimal DoE workflow, returns a consolidated result.

    ranges: optional {"growthtime": (min, max), ...}; missing params fall
        back to default_ranges(). n_runs bumped up to n_terms + 2 if lower.
    """
    require_positive_int(n_runs, "n_runs")
    require_positive_int(resolution, "resolution")
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}")

    df = pd.read_csv(DATA_PATH)
    resolved_ranges = default_ranges()
    if ranges:
        for p in DOE_PARAMS:
            if p in ranges and ranges[p] is not None:
                lo, hi = require_range(ranges[p][0], ranges[p][1], p)
                resolved_ranges[p] = (lo, hi)

    fixed_substrate_quality = float(df[FIXED_FEATURE].median())

    lo = np.array([resolved_ranges[p][0] for p in DOE_PARAMS])
    hi = np.array([resolved_ranges[p][1] for p in DOE_PARAMS])

    candidates = _build_candidate_pool(resolved_ranges, resolution)
    candidates_norm = _normalize(candidates, lo, hi)

    selected_idx = d_optimal_select(candidates_norm, n_runs, seed=seed)
    design_points = candidates[selected_idx]
    design_norm = _normalize(design_points, lo, hi)

    X_model = np.column_stack([design_points, np.full(len(design_points), fixed_substrate_quality)])
    eds_mean, _ = predict_with_uncertainty(X_model, "EDS_ratio")
    film_mean, _ = predict_with_uncertainty(X_model, "RHEED_Quality_Film")
    score_holistic = FILM_WEIGHT * film_mean + EDS_WEIGHT * stoichiometry_score(eds_mean)

    F = _quadratic_design_matrix(design_norm)
    n_obs, n_terms = F.shape
    coeffs, _, _, _ = np.linalg.lstsq(F, score_holistic, rcond=None)
    y_hat = F @ coeffs
    resid = score_holistic - y_hat
    dof = max(n_obs - n_terms, 1)
    sigma2 = float(np.sum(resid**2) / dof)
    sigma = float(np.sqrt(sigma2))
    three_sigma = 3 * sigma

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((score_holistic - score_holistic.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    try:
        cov = sigma2 * np.linalg.inv(F.T @ F)
    except np.linalg.LinAlgError:
        cov = sigma2 * np.linalg.pinv(F.T @ F)
    std_errors = np.sqrt(np.clip(np.diag(cov), 0, None))

    coefficient_table = [
        {
            "term": name,
            "coefficient": round(float(c), 4),
            "std_error": round(float(se), 4),
        }
        for name, c, se in zip(QUADRATIC_TERM_NAMES, coeffs, std_errors)
    ]

    def neg_score(x_norm):
        f = _quadratic_design_matrix(x_norm.reshape(1, -1))[0]
        return -float(f @ coeffs)

    rng = np.random.default_rng(seed)
    best_val, best_x = -np.inf, None
    for _ in range(40):
        x0 = rng.uniform(-1, 1, size=3)
        res = minimize(neg_score, x0, bounds=[(-1, 1)] * 3, method="L-BFGS-B")
        if -res.fun > best_val:
            best_val, best_x = -res.fun, res.x

    optimum_raw = _denormalize(best_x, lo, hi)
    optimum_X = np.array([[optimum_raw[0], optimum_raw[1], optimum_raw[2], fixed_substrate_quality]])
    optimum_eds_mean, _ = predict_with_uncertainty(optimum_X, "EDS_ratio")
    optimum_film_mean, _ = predict_with_uncertainty(optimum_X, "RHEED_Quality_Film")

    design_table = []
    for i in range(len(design_points)):
        design_table.append({
            "run": i + 1,
            "growthtime": round(float(design_points[i, 0]), 3),
            "filamentpower": round(float(design_points[i, 1]), 4),
            "flux_ratio": round(float(design_points[i, 2]), 4),
            "Substrate_quality_fixed": round(fixed_substrate_quality, 2),
            "predicted_EDS_ratio": round(float(eds_mean[i]), 3),
            "predicted_RHEED_Quality_Film": round(float(film_mean[i]), 2),
            "Score_Holistic": round(float(score_holistic[i]), 3),
        })

    return {
        "ranges_used": {p: [round(resolved_ranges[p][0], 4), round(resolved_ranges[p][1], 4)] for p in DOE_PARAMS},
        "fixed_Substrate_quality": round(fixed_substrate_quality, 2),
        "n_runs": len(design_points),
        "design_table": design_table,
        "quadratic_fit": {
            "objective": "Score_Holistic = 0.65 * RHEED_Quality_Film + 0.35 * stoichiometry_score(EDS_ratio), same composite used by the active-learning loop",
            "coding": "each parameter is coded to [-1, 1] over its range before fitting; coefficients are in coded units",
            "coefficients": coefficient_table,
            "residual_std_dev": round(sigma, 4),
            "three_sigma": round(three_sigma, 4),
            "r_squared": round(float(r_squared), 4),
            "degrees_of_freedom": dof,
        },
        "optimum": {
            "growthtime": round(float(optimum_raw[0]), 3),
            "filamentpower": round(float(optimum_raw[1]), 4),
            "flux_ratio": round(float(optimum_raw[2]), 4),
            "Substrate_quality_fixed": round(fixed_substrate_quality, 2),
            "predicted_Score_Holistic": round(float(best_val), 3),
            "predicted_EDS_ratio": round(float(optimum_eds_mean[0]), 3),
            "predicted_RHEED_Quality_Film": round(float(optimum_film_mean[0]), 2),
        },
    }
