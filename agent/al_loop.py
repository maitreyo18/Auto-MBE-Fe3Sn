import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

THIS_DIR = Path(__file__).resolve().parent

FEATURES = ["growthtime", "filamentpower", "flux_ratio", "Substrate_quality"]
FIXED_FEATURES = ["growthtime", "Substrate_quality"]
GRID_RESOLUTION = 60
XI = 0.01
FILM_WEIGHT = 0.65
EDS_WEIGHT = 0.35
EDS_TARGET_RATIO = 3.0
EDS_SIGMA = 0.5
DEFAULT_N_ITERATIONS = 5


def calculate_ei(mu, std, best_val, xi=XI):
    std = np.maximum(std, 1e-9)
    z = (mu - best_val - xi) / std
    return (mu - best_val - xi) * norm.cdf(z) + std * norm.pdf(z)


def fit_forest(X_scaled, y):
    rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    rf.fit(X_scaled, y)
    return rf


def tree_predict(rf, X_scaled):
    """Per-tree spread of a single fitted forest (bootstrap variance across
    its n_estimators trees) -- the uncertainty source used for EI, matching
    the original paper's method."""
    tree_preds = np.array([t.predict(X_scaled) for t in rf.estimators_])
    return tree_preds.mean(axis=0), tree_preds.std(axis=0)


def stoichiometry_score(eds_ratio):
    """0-100 closeness-to-target score, used only to build the single
    composite training target below -- never exposed as a prediction."""
    return np.exp(-((eds_ratio - EDS_TARGET_RATIO) ** 2) / (2 * EDS_SIGMA**2)) * 100


def run_al_iteration(df, iteration, out_dir):
    df = df.copy()
    df["Score_Holistic"] = FILM_WEIGHT * df["RHEED_Quality_Film"] + EDS_WEIGHT * stoichiometry_score(df["EDS_ratio"])

    X = df[FEATURES].values
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_scaled = (X - X_mean) / (X_std + 1e-8)

    score_rf = fit_forest(X_scaled, df["Score_Holistic"].values)
    eds_rf = fit_forest(X_scaled, df["EDS_ratio"].values)
    film_rf = fit_forest(X_scaled, df["RHEED_Quality_Film"].values)

    iter_dir = out_dir / f"iteration_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(score_rf, iter_dir / "rf_Score_Holistic.joblib")
    joblib.dump(eds_rf, iter_dir / "rf_EDS_ratio.joblib")
    joblib.dump(film_rf, iter_dir / "rf_RHEED_Quality_Film.joblib")

    top_quartile = df[df["Score_Holistic"] >= df["Score_Holistic"].quantile(0.75)]
    fixed_values = top_quartile[FIXED_FEATURES].median()

    power_min, power_max = df["filamentpower"].min(), df["filamentpower"].max()
    flux_min, flux_max = df["flux_ratio"].min(), df["flux_ratio"].max()
    power_grid = np.linspace(power_min, power_max, GRID_RESOLUTION)
    flux_grid = np.linspace(flux_min, flux_max, GRID_RESOLUTION)
    power_mesh, flux_mesh = np.meshgrid(power_grid, flux_grid)

    grid_df = pd.DataFrame({
        "growthtime": fixed_values["growthtime"],
        "filamentpower": power_mesh.ravel(),
        "flux_ratio": flux_mesh.ravel(),
        "Substrate_quality": fixed_values["Substrate_quality"],
    })[FEATURES]

    X_grid_scaled = (grid_df.values - X_mean) / (X_std + 1e-8)

    score_mu, score_std = tree_predict(score_rf, X_grid_scaled)
    eds_ratio_mu, _ = tree_predict(eds_rf, X_grid_scaled)
    film_mu, _ = tree_predict(film_rf, X_grid_scaled)

    ei = calculate_ei(score_mu, score_std, df["Score_Holistic"].max())

    best_idx = np.argmax(ei)
    best_power = grid_df["filamentpower"].values[best_idx]
    best_flux = grid_df["flux_ratio"].values[best_idx]
    best_ei = ei[best_idx]
    best_eds_ratio = eds_ratio_mu[best_idx]
    best_film_quality = film_mu[best_idx]

    ei_mesh = ei.reshape(power_mesh.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    contour = ax.contourf(flux_mesh, power_mesh, ei_mesh, levels=30, cmap="RdBu_r")
    fig.colorbar(contour, ax=ax, label="Expected Improvement (Score_Holistic: 65% film + 35% stoichiometry)")
    ax.scatter(df["flux_ratio"], df["filamentpower"],
               facecolors="white", edgecolors="black", s=60, label="Dataset so far")
    ax.scatter([best_flux], [best_power], marker="*", s=300, color="gold",
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
        "suggested_growthtime": round(float(fixed_values["growthtime"]), 2),
        "suggested_filamentpower": round(float(best_power), 4),
        "suggested_flux_ratio": round(float(best_flux), 4),
        "suggested_Substrate_quality": round(float(fixed_values["Substrate_quality"]), 2),
        "expected_improvement": round(float(best_ei), 4),
        "predicted_EDS_ratio": round(float(best_eds_ratio), 3),
        "predicted_RHEED_Quality_Film": round(float(best_film_quality), 2),
        "model_dir": str(iter_dir),
        "plot_path": str(plot_path),
    }

    new_row = {
        "growthtime": result["suggested_growthtime"],
        "filamentpower": result["suggested_filamentpower"],
        "flux_ratio": result["suggested_flux_ratio"],
        "Substrate_quality": result["suggested_Substrate_quality"],
        "EDS_ratio": result["predicted_EDS_ratio"],
        "RHEED_Quality_Film": result["predicted_RHEED_Quality_Film"],
    }
    return result, new_row


def run_al_loop(initial_data, n_iterations=DEFAULT_N_ITERATIONS, out_dir=None):
    """initial_data: list of dict rows, each with growthtime, filamentpower,
    flux_ratio, Substrate_quality, EDS_ratio, RHEED_Quality_Film.

    Each iteration builds a single composite Score_Holistic target (65% film
    quality + 35% stoichiometry closeness, both on a 0-100 scale) and fits
    one fresh RF on it -- EI uses that forest's per-tree (bootstrap) std,
    matching the original paper's method. Separate forests are also fit just
    to report readable EDS_ratio/RHEED_Quality_Film values for the suggested
    point.

    No pre-trained checkpoints are used. The forest's own predicted mean is
    treated as the outcome for the suggested point, appended to the dataset,
    and the next iteration refits on the growing dataset. Saves a plot and
    all forests used for every iteration under out_dir/iteration_N/."""
    df = pd.DataFrame(initial_data)
    out_dir = Path(out_dir) if out_dir else THIS_DIR / "al_runs"

    results = []
    for i in range(1, n_iterations + 1):
        result, new_row = run_al_iteration(df, i, out_dir)
        results.append(result)
        print(
            f"Iteration {i}: EDS_ratio={result['predicted_EDS_ratio']}, "
            f"RHEED_Quality_Film={result['predicted_RHEED_Quality_Film']}, "
            f"EI={result['expected_improvement']}"
        )
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return results
