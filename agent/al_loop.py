"""Active learning against the lab's ground-truth models.

The two trained models in models/ (film-quality MLP, stoichiometry RFR) stand in
for real growths. The loop starts from 10 of the 17 initial experiments. Each
iteration fits LOOCV copies of those same two model definitions on the rows
observed so far, picks the next filament power / flux ratio by Expected
Improvement on Score_Holistic, asks the ground-truth models for the outcome, and
appends it. Uncertainty is the spread of Score_Holistic across the LOOCV folds.
"""
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "models"))

from lab_models import (  # noqa: E402
    EDS_FEATURES, EDS_TARGET, FILM_FEATURES, FILM_TARGET, SUBSTRATE_RAW,
    build_eds_model, build_film_model, load_test, load_train,
)
from predict import predict_eds_ratio, predict_film_quality  # noqa: E402
from validation import require_positive_int  # noqa: E402

FIXED_FEATURES = ["growthtime", "Substrate_quality"] + SUBSTRATE_RAW
GRID_RESOLUTION = 60
XI = 0.01
FILM_WEIGHT = 0.65
EDS_WEIGHT = 0.35
EDS_TARGET_RATIO = 3.0
EDS_SIGMA = 0.5
DEFAULT_N_ITERATIONS = 5
DEFAULT_N_INITIAL = 10
MIN_INITIAL = 3


def calculate_ei(mu, std, best_val, xi=XI):
    std = np.maximum(std, 1e-9)
    z = (mu - best_val - xi) / std
    return (mu - best_val - xi) * norm.cdf(z) + std * norm.pdf(z)


def stoichiometry_score(eds_ratio):
    """0-100 closeness of the Fe:Sn ratio to its target."""
    return np.exp(-((eds_ratio - EDS_TARGET_RATIO) ** 2) / (2 * EDS_SIGMA**2)) * 100


def holistic_score(film, eds):
    return FILM_WEIGHT * film + EDS_WEIGHT * stoichiometry_score(eds)


def query_ground_truth(rows):
    """rows: DataFrame with every model input -> (eds, film, score)."""
    eds, _ = predict_eds_ratio(rows[EDS_FEATURES].to_numpy(dtype=float))
    film, _ = predict_film_quality(rows[FILM_FEATURES].to_numpy(dtype=float))
    return eds, film, holistic_score(film, eds)


def fit_loocv_models(df):
    """One (film MLP, EDS RFR) pair per left-out row, plus each target's RMSE
    on its held-out rows."""
    X_f, y_f = df[FILM_FEATURES].to_numpy(dtype=float), df[FILM_TARGET].to_numpy(dtype=float)
    X_e, y_e = df[EDS_FEATURES].to_numpy(dtype=float), df[EDS_TARGET].to_numpy(dtype=float)
    n = len(df)
    film_folds, eds_folds = [], []
    held_f, held_e = np.zeros(n), np.zeros(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n):
            keep = np.arange(n) != i
            film = build_film_model().fit(X_f[keep], y_f[keep])
            eds = build_eds_model().fit(X_e[keep], y_e[keep])
            held_f[i] = film.predict(X_f[i : i + 1])[0]
            held_e[i] = eds.predict(X_e[i : i + 1])[0]
            film_folds.append(film)
            eds_folds.append(eds)
    return film_folds, eds_folds, {
        "film_loocv_rmse": round(float(np.sqrt(np.mean((y_f - held_f) ** 2))), 4),
        "eds_loocv_rmse": round(float(np.sqrt(np.mean((y_e - held_e) ** 2))), 4),
    }


def ensemble_scores(film_folds, eds_folds, rows):
    """Score_Holistic per fold, shape (n_folds, n_rows)."""
    X_f = rows[FILM_FEATURES].to_numpy(dtype=float)
    X_e = rows[EDS_FEATURES].to_numpy(dtype=float)
    return np.array([holistic_score(f.predict(X_f), e.predict(X_e)) for f, e in zip(film_folds, eds_folds)])


def run_al_iteration(df, iteration, out_dir, bounds, eval_rows, eval_truth):
    df = df.copy()
    film_folds, eds_folds, loocv = fit_loocv_models(df)
    observed = holistic_score(df[FILM_TARGET].to_numpy(dtype=float), df[EDS_TARGET].to_numpy(dtype=float))
    best_so_far = float(observed.max())

    top_quartile = df[observed >= np.quantile(observed, 0.75)]
    fixed = top_quartile[FIXED_FEATURES].median()

    power_grid = np.linspace(*bounds["filamentpower"], GRID_RESOLUTION)
    flux_grid = np.linspace(*bounds["flux_ratio"], GRID_RESOLUTION)
    power_mesh, flux_mesh = np.meshgrid(power_grid, flux_grid)
    grid = pd.DataFrame({"filamentpower": power_mesh.ravel(), "flux_ratio": flux_mesh.ravel()})
    for col in FIXED_FEATURES:
        grid[col] = fixed[col]

    fold_scores = ensemble_scores(film_folds, eds_folds, grid)
    mu, std = fold_scores.mean(axis=0), fold_scores.std(axis=0)
    ei = calculate_ei(mu, std, best_so_far)
    best_idx = int(np.argmax(ei))
    point = grid.iloc[[best_idx]]

    eds, film, score = query_ground_truth(point)

    eval_pred = ensemble_scores(film_folds, eds_folds, eval_rows).mean(axis=0)
    ss_res = float(np.sum((eval_truth - eval_pred) ** 2))
    ss_tot = float(np.sum((eval_truth - eval_truth.mean()) ** 2))

    iter_dir = Path(out_dir) / f"iteration_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(film_folds, iter_dir / "mlp_RHEED_Quality_Film_loocv.joblib")
    joblib.dump(eds_folds, iter_dir / "rf_EDS_ratio_loocv.joblib")

    fig, ax = plt.subplots(figsize=(7, 6))
    contour = ax.contourf(flux_mesh, power_mesh, ei.reshape(power_mesh.shape), levels=30, cmap="RdBu_r")
    fig.colorbar(contour, ax=ax, label="Expected Improvement (Score_Holistic)")
    ax.scatter(df["flux_ratio"], df["filamentpower"], facecolors="white", edgecolors="black", s=60, label="Observed")
    ax.scatter(point["flux_ratio"], point["filamentpower"], marker="*", s=300, color="gold",
               edgecolors="black", label="Suggested next point")
    ax.set_xlabel("Flux Ratio (Fe/Sn)")
    ax.set_ylabel("Filament Power")
    ax.set_title(f"AL iteration {iteration}: EI acquisition landscape")
    ax.legend(loc="upper right")
    plot_path = iter_dir / f"al_iteration_{iteration}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    result = {
        "iteration": iteration,
        "n_observed": len(df),
        "suggested_growthtime": round(float(fixed["growthtime"]), 2),
        "suggested_filamentpower": round(float(point["filamentpower"].iloc[0]), 4),
        "suggested_flux_ratio": round(float(point["flux_ratio"].iloc[0]), 4),
        "suggested_Substrate_quality": round(float(fixed["Substrate_quality"]), 2),
        "expected_improvement": round(float(ei[best_idx]), 4),
        "predicted_Score_Holistic_AL": round(float(mu[best_idx]), 3),
        "uncertainty_Score_Holistic_AL": round(float(std[best_idx]), 3),
        "best_score_so_far": round(best_so_far, 3),
        "ground_truth_EDS_ratio": round(float(eds[0]), 3),
        "ground_truth_RHEED_Quality_Film": round(float(film[0]), 2),
        "ground_truth_Score_Holistic": round(float(score[0]), 3),
        **loocv,
        "mae_vs_ground_truth": round(float(np.mean(np.abs(eval_truth - eval_pred))), 4),
        "r2_vs_ground_truth": round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else None,
        "model_dir": str(iter_dir),
        "plot_path": str(plot_path),
    }
    new_row = {**point.iloc[0].to_dict(), EDS_TARGET: float(eds[0]), FILM_TARGET: float(film[0])}
    return result, new_row


def run_al_loop(n_iterations=DEFAULT_N_ITERATIONS, n_initial=DEFAULT_N_INITIAL, out_dir=None, seed=42):
    """Starts from n_initial randomly chosen rows of the 17 initial experiments
    (default 10, with their measured outcomes). New points are labelled by the
    ground-truth models. Each result reports how well the AL models track the
    ground truth over all 21 growths. Saves each iteration's LOOCV models and a
    plot under out_dir/iteration_N/ (default agent/al_runs)."""
    require_positive_int(n_iterations, "n_iterations")
    require_positive_int(n_initial, "n_initial")
    pool = load_train()
    if not MIN_INITIAL <= n_initial <= len(pool):
        raise ValueError(f"n_initial must be between {MIN_INITIAL} and {len(pool)}, got {n_initial}")
    out_dir = Path(out_dir) if out_dir else THIS_DIR / "al_runs"

    columns = FILM_FEATURES + [EDS_TARGET, FILM_TARGET]
    bounds = {c: (float(pool[c].min()), float(pool[c].max())) for c in ("filamentpower", "flux_ratio")}
    eval_rows = pd.concat([pool[FILM_FEATURES], load_test()[FILM_FEATURES]], ignore_index=True)
    _, _, eval_truth = query_ground_truth(eval_rows)

    seed_idx = np.sort(np.random.default_rng(seed).choice(len(pool), size=n_initial, replace=False))
    df = pool.iloc[seed_idx][columns].reset_index(drop=True)

    results = []
    for i in range(1, n_iterations + 1):
        result, new_row = run_al_iteration(df, i, out_dir, bounds, eval_rows, eval_truth)
        results.append(result)
        print(
            f"Iteration {i} (n={result['n_observed']}): power={result['suggested_filamentpower']}, "
            f"flux={result['suggested_flux_ratio']}, EI={result['expected_improvement']}, "
            f"ground truth score={result['ground_truth_Score_Holistic']}, "
            f"R2 vs ground truth={result['r2_vs_ground_truth']}"
        )
        df = pd.concat([df, pd.DataFrame([new_row])[columns]], ignore_index=True)
    return results
