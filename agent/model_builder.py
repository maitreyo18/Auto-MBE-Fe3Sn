"""On-the-fly Random Forest / Gaussian Process / Kernel Ridge / MLP model builder.

Pure-Python core (no LLM, no langchain) behind the model-builder agent in
model_agent.py. Trains RFR, GPR, KRR and MLP models on any numeric columns of a
CSV, saves them under model_builder_results/models/<model_id>/ (model.joblib +
meta.json) so they can be reused, compares their train/test RMSEs, and reuses them for prediction
and optimization. Other results are written only when save_session_results is
called, which happens only when the user asks. It also serves the lab's two saved models. auto_search.py
builds on this to explore features and hyperparameters by itself.
"""
import json
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, ConstantKernel, Matern, RationalQuadratic, WhiteKernel,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from validation import require_finite, require_positive_int

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent
LAB_MODELS_DIR = REPO_DIR / "models"
sys.path.insert(0, str(LAB_MODELS_DIR))

from lab_models import CHECKPOINTS, load_test, load_train  # noqa: E402
from predict import load_bundle, predict_with_uncertainty  # noqa: E402
DEFAULT_DATA_PATH = REPO_DIR / "data" / "train_compiled.csv"
RESULTS_ROOT = REPO_DIR / "model_builder_results"
MODEL_DIR = RESULTS_ROOT / "models"

MODEL_TYPES = ("rfr", "gpr", "krr", "mlp")
KERNELS = ("matern52", "matern32", "rbf", "rq")
KRR_KERNELS = ("rbf", "laplacian", "linear")
KRR_ALPHA_GRID = tuple(float(a) for a in np.logspace(-3, 1, 9))
KRR_GAMMA_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)
MIN_TRAIN_ROWS = 3
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


# ----------------------------------------------------------------- data ----

def resolve_data_path(data_path=None):
    """None -> the repo's training CSV. Relative paths resolve against the
    repo root. Must be an existing .csv file."""
    if data_path in (None, ""):
        return DEFAULT_DATA_PATH
    path = Path(data_path)
    if not path.is_absolute():
        path = (REPO_DIR / path).resolve()
    if path.suffix.lower() != ".csv":
        raise ValueError(f"data_path must be a .csv file, got {path.name!r}")
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")
    return path


def _load_df(data_path):
    """data_path="all" means all 21 lab growths (17 initial + 4 later)."""
    if isinstance(data_path, str) and data_path.lower() == "all":
        return Path("train_compiled.csv+test_compiled.csv"), pd.concat([load_train(), load_test()], ignore_index=True)
    path = resolve_data_path(data_path)
    return path, pd.read_csv(path)


def describe_data(data_path=None):
    """Columns the user can choose targets / training variables from."""
    path, df = _load_df(data_path)
    numeric = df.select_dtypes(include="number")
    return {
        "data_path": str(path),
        "n_rows": len(df),
        "numeric_columns": {
            c: {
                "min": round(float(numeric[c].min()), 4),
                "max": round(float(numeric[c].max()), 4),
                "mean": round(float(numeric[c].mean()), 4),
                "n_missing": int(numeric[c].isna().sum()),
            }
            for c in numeric.columns
        },
        "non_numeric_columns_unusable": [c for c in df.columns if c not in numeric.columns],
    }


# ---------------------------------------------------------------- models ----

def _build_kernel(kernel, n_features):
    """Constant * base kernel + learned WhiteKernel noise. Inputs are
    standardized, so length-scale bounds are in standard-deviation units.
    Matern/RBF use one length scale per feature (ARD)."""
    ls = np.ones(n_features)
    bounds = (1e-2, 1e2)
    base = {
        "matern52": lambda: Matern(length_scale=ls, length_scale_bounds=bounds, nu=2.5),
        "matern32": lambda: Matern(length_scale=ls, length_scale_bounds=bounds, nu=1.5),
        "rbf": lambda: RBF(length_scale=ls, length_scale_bounds=bounds),
        "rq": lambda: RationalQuadratic(length_scale=1.0, length_scale_bounds=bounds),
    }[kernel]()
    return ConstantKernel(1.0, (1e-2, 1e2)) * base + WhiteKernel(0.1, (1e-5, 1e1))


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _safe_id(name):
    if not isinstance(name, str) or not SAFE_NAME.match(name):
        raise ValueError(f"invalid model id {name!r}: use only letters, digits, '_', '-', '.'")
    return name


def _unique_id(base):
    model_id, n = base, 2
    while (MODEL_DIR / model_id).exists():
        model_id = f"{base}-{n}"
        n += 1
    return model_id


def _krr(kernel, alpha, gamma):
    """KRR on standardized inputs; targets are standardized inside so the
    kernel model needs no intercept."""
    return TransformedTargetRegressor(
        regressor=KernelRidge(kernel=kernel, alpha=alpha, gamma=gamma), transformer=StandardScaler()
    )


def _fit_krr(Xs, y, kernel, alpha, gamma, random_state):
    """Fits a KRR on already-scaled inputs. Any of alpha / gamma left as None
    is chosen by inner K-fold CV (on THESE rows only, so a test split is never
    touched) over a log-spaced grid. Returns (model, chosen_params, cv_rmse)."""
    alphas = [alpha] if alpha is not None else list(KRR_ALPHA_GRID)
    gammas = [None] if kernel == "linear" else ([gamma] if gamma is not None else list(KRR_GAMMA_GRID))
    cv_rmse = None
    if len(alphas) * len(gammas) > 1:
        search = GridSearchCV(
            _krr(kernel, 1.0, None),
            {"regressor__alpha": alphas, "regressor__gamma": gammas},
            scoring="neg_root_mean_squared_error",
            cv=KFold(n_splits=min(5, len(y)), shuffle=True, random_state=random_state),
        ).fit(Xs, y)
        alpha, gamma = search.best_params_["regressor__alpha"], search.best_params_["regressor__gamma"]
        cv_rmse = round(float(-search.best_score_), 4)
    else:
        alpha, gamma = alphas[0], gammas[0]
    model = _krr(kernel, alpha, gamma).fit(Xs, y)
    return model, {"kernel": kernel, "alpha": float(alpha), "gamma": None if gamma is None else float(gamma)}, cv_rmse


def _mlp(hidden_nodes, alpha, batch_size, learning_rate, max_iter, random_state):
    """One-hidden-layer MLP on standardized inputs and target. Small nodes and
    a strong alpha suit tiny datasets."""
    return TransformedTargetRegressor(
        regressor=MLPRegressor(
            hidden_layer_sizes=(hidden_nodes,), activation="tanh", solver="adam", batch_size=batch_size,
            learning_rate_init=learning_rate, alpha=alpha, max_iter=max_iter, random_state=random_state,
        ),
        transformer=StandardScaler(),
    )


def _fold_models(make, Xs, y, random_state):
    """5 CV fold models (80% each) -- their prediction spread is the uncertainty
    for KRR and MLP, which have no native predictive variance."""
    folds = []
    for tr, _ in KFold(n_splits=min(5, len(y)), shuffle=True, random_state=random_state).split(Xs):
        folds.append(make().fit(Xs[tr], y[tr]))
    return folds


def expand_features(X, features, engineered):
    """Appends engineered columns named log(a), a^2 or a*b to X (columns in the
    order of `features`). Used by auto_search and when predicting with its models."""
    index = {f: i for i, f in enumerate(features)}
    columns = [X]
    for name in engineered or []:
        if name.startswith("log("):
            col = np.log(X[:, index[name[4:-1]]])
        elif name.endswith("^2"):
            col = X[:, index[name[:-2]]] ** 2
        else:
            a, b = name.split("*")
            col = X[:, index[a]] * X[:, index[b]]
        columns.append(col.reshape(-1, 1))
    return np.hstack(columns)


def train_models(
    targets=None,
    features=None,
    model_types=None,
    data_path=None,
    test_size=0.25,
    random_state=42,
    refit_on_all_data=False,
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    max_features=1.0,
    kernel="matern52",
    n_restarts_optimizer=5,
    krr_kernel="rbf",
    alpha=None,
    gamma=None,
    hidden_nodes=3,
    mlp_alpha=1.0,
    batch_size=4,
    learning_rate=0.01,
    mlp_max_iter=300,
    name_prefix=None,
):
    """Trains one model per (target, model type) on a shared train/test split
    and saves each. targets/features are never guessed: if either is missing
    a needs_user_input payload is returned instead of training."""
    path, df = _load_df(data_path)
    numeric_cols = list(df.select_dtypes(include="number").columns)

    if not targets or not features:
        return {
            "needs_user_input": True,
            "message": "Ask the user which target column(s) to predict and which training variables (features) to use.",
            "missing": [n for n, v in (("targets", targets), ("features", features)) if not v],
            "available_numeric_columns": numeric_cols,
        }

    targets = list(targets)
    features = list(features)
    model_types = [m.lower() for m in (model_types or MODEL_TYPES)]
    bad_types = [m for m in model_types if m not in MODEL_TYPES]
    if bad_types:
        raise ValueError(f"unsupported model type(s) {bad_types}: only {list(MODEL_TYPES)} are supported")
    unknown = [c for c in targets + features if c not in numeric_cols]
    if unknown:
        raise ValueError(f"column(s) {unknown} are not numeric columns of {path.name}; available: {numeric_cols}")
    overlap = sorted(set(targets) & set(features))
    if overlap:
        raise ValueError(f"column(s) {overlap} cannot be both target and feature")
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {list(KERNELS)}, got {kernel!r}")
    if krr_kernel not in KRR_KERNELS:
        raise ValueError(f"krr_kernel must be one of {list(KRR_KERNELS)}, got {krr_kernel!r}")
    alpha = None if alpha is None else require_finite(alpha, "alpha")
    gamma = None if gamma is None else require_finite(gamma, "gamma")
    if (alpha is not None and alpha <= 0) or (gamma is not None and gamma <= 0):
        raise ValueError("alpha and gamma must be > 0 when given")
    for name, value in (("hidden_nodes", hidden_nodes), ("batch_size", batch_size), ("mlp_max_iter", mlp_max_iter)):
        require_positive_int(value, name)
    mlp_alpha = require_finite(mlp_alpha, "mlp_alpha")
    learning_rate = require_finite(learning_rate, "learning_rate")
    if mlp_alpha < 0 or learning_rate <= 0:
        raise ValueError("mlp_alpha must be >= 0 and learning_rate must be > 0")
    test_size = require_finite(test_size, "test_size")
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    require_positive_int(n_estimators, "n_estimators")
    if max_depth is not None:
        require_positive_int(max_depth, "max_depth")
    require_positive_int(min_samples_leaf, "min_samples_leaf")
    if not isinstance(n_restarts_optimizer, int) or n_restarts_optimizer < 0:
        raise ValueError(f"n_restarts_optimizer must be an int >= 0, got {n_restarts_optimizer!r}")

    used = df[features + targets]
    n_dropped = int(used.isna().any(axis=1).sum())
    clean = used.dropna()
    idx_all = clean.index.to_numpy()
    if len(clean) < MIN_TRAIN_ROWS + 1:
        raise ValueError(f"only {len(clean)} complete rows -- need at least {MIN_TRAIN_ROWS + 1}")
    train_pos, test_pos = train_test_split(
        np.arange(len(clean)), test_size=test_size, random_state=random_state
    )
    if len(train_pos) < MIN_TRAIN_ROWS:
        raise ValueError(f"test_size {test_size} leaves only {len(train_pos)} training rows")

    X = clean[features].to_numpy(dtype=float)
    feature_ranges = {f: [float(clean[f].min()), float(clean[f].max())] for f in features}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    prefix = f"{_safe_id(name_prefix)}_" if name_prefix else ""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    trained = []
    for target in targets:
        y = clean[target].to_numpy(dtype=float)
        for mtype in model_types:
            krr_info = {}

            def fit_one(Xs, ytr, krr_alpha, krr_gamma):
                """Returns (model, folds-or-None). Only KRR tunes/folds."""
                if mtype == "rfr":
                    m = RandomForestRegressor(
                        n_estimators=n_estimators, max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf, max_features=max_features,
                        random_state=random_state,
                    )
                    return m.fit(Xs, ytr), None
                if mtype == "gpr":
                    m = GaussianProcessRegressor(
                        kernel=_build_kernel(kernel, len(features)), normalize_y=True,
                        n_restarts_optimizer=n_restarts_optimizer, random_state=random_state,
                    )
                    return m.fit(Xs, ytr), None
                if mtype == "mlp":
                    def make():
                        return _mlp(hidden_nodes, mlp_alpha, batch_size, learning_rate, mlp_max_iter, random_state)
                    return make().fit(Xs, ytr), _fold_models(make, Xs, ytr, random_state)
                m, params, inner_cv = _fit_krr(Xs, ytr, krr_kernel, krr_alpha, krr_gamma, random_state)
                krr_info.update(params=params, inner_cv_rmse=inner_cv,
                                tuned=alpha is None or (gamma is None and krr_kernel != "linear"))
                return m, _fold_models(lambda: _krr(params["kernel"], params["alpha"], params["gamma"]), Xs, ytr, random_state)

            scaler = StandardScaler().fit(X[train_pos])
            Xtr = scaler.transform(X[train_pos])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model, folds = fit_one(Xtr, y[train_pos], alpha, gamma)
            fit_warnings = sorted({
                "optimizer did not fully converge (a kernel hyperparameter hit its bound)"
                if issubclass(w.category, ConvergenceWarning) else str(w.message)[:200]
                for w in caught
            })

            pred_tr = model.predict(Xtr)
            pred_te = model.predict(scaler.transform(X[test_pos]))
            metrics = {
                "train_rmse": round(_rmse(y[train_pos], pred_tr), 4),
                "test_rmse": round(_rmse(y[test_pos], pred_te), 4),
                "train_r2": round(float(r2_score(y[train_pos], pred_tr)), 4),
                "test_r2": round(float(r2_score(y[test_pos], pred_te)), 4) if len(test_pos) > 1 else None,
            }

            if refit_on_all_data:
                scaler = StandardScaler().fit(X)
                chosen = krr_info.get("params", {})
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # keep the hyperparameters chosen on the train split for KRR
                    model, folds = fit_one(scaler.transform(X), y, chosen.get("alpha", alpha), chosen.get("gamma", gamma))

            model_id = _unique_id(f"{prefix}{mtype}_{target}_{run_id}")
            hyperparams = (
                {"n_estimators": n_estimators, "max_depth": max_depth,
                 "min_samples_leaf": min_samples_leaf, "max_features": max_features}
                if mtype == "rfr"
                else {"kernel": kernel, "n_restarts_optimizer": n_restarts_optimizer, "normalize_y": True}
                if mtype == "gpr"
                else {"hidden_nodes": hidden_nodes, "alpha": mlp_alpha, "batch_size": batch_size,
                      "learning_rate": learning_rate, "max_iter": mlp_max_iter}
                if mtype == "mlp"
                else {**krr_info["params"], "tuned_by_inner_cv": krr_info["tuned"],
                      "inner_cv_rmse": krr_info["inner_cv_rmse"]}
            )
            meta = {
                "model_id": model_id,
                "model_type": mtype,
                "target": target,
                "features": features,
                "hyperparameters": hyperparams,
                "data_path": str(path),
                "n_rows_used": len(clean),
                "n_rows_dropped_missing": n_dropped,
                "n_train": len(train_pos),
                "n_test": len(test_pos),
                "split": {"test_size": test_size, "random_state": random_state},
                "fitted_on": "all_rows" if refit_on_all_data else "train_split_only",
                "feature_ranges": feature_ranges,
                "metrics": metrics,
                "warnings": fit_warnings,
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source_row_index_test": [int(idx_all[i]) for i in test_pos],
            }
            if mtype == "rfr":
                meta["feature_importances"] = {
                    f: round(float(v), 4) for f, v in zip(features, model.feature_importances_)
                }
            elif mtype == "gpr":
                meta["fitted_kernel"] = str(model.kernel_)
                meta["log_marginal_likelihood"] = round(float(model.log_marginal_likelihood_value_), 4)
            else:
                meta["uncertainty"] = "std across 5 CV fold models (rough; usually under-estimates)"

            out = MODEL_DIR / model_id
            out.mkdir(parents=True)
            joblib.dump({"scaler": scaler, "model": model, "folds": folds}, out / "model.joblib")
            (out / "meta.json").write_text(json.dumps(meta, indent=2))
            trained.append({k: meta[k] for k in
                            ("model_id", "model_type", "target", "n_train", "n_test", "metrics", "warnings")})

    notes = []
    if len(test_pos) < 5:
        notes.append(f"Only {len(test_pos)} test rows: test RMSE is a very noisy estimate.")
    if n_dropped:
        notes.append(f"{n_dropped} row(s) dropped for missing values in the chosen columns.")
    return {"trained": trained, "notes": notes, "model_dir": str(MODEL_DIR)}


# ------------------------------------------------- built-in lab models ----
# The two saved models in models/ (trained by models/train.py on the 17 initial
# growths and tested on the 4 later ones), exposed under ids lab_<target> so they
# can be listed, compared and used exactly like on-the-fly models.

LAB_PREFIX = "lab_"
LAB_EVALUATION = "4 later growths"


def _lab_ids():
    return [f"{LAB_PREFIX}{t}" for t, path in CHECKPOINTS.items() if path.exists()]


def _lab_meta(model_id):
    bundle = load_bundle(model_id[len(LAB_PREFIX):])
    return {
        "model_id": model_id,
        "model_type": bundle["model_type"],
        "built_in": True,
        "target": bundle["target"],
        "features": list(bundle["features"]),
        "hyperparameters": dict(bundle["params"]),
        "data_path": "train_compiled.csv (train), test_compiled.csv (test)",
        "n_rows_used": bundle["n_train"] + bundle["n_test"],
        "n_train": bundle["n_train"],
        "n_test": bundle["n_test"],
        "split": {"test_size": LAB_EVALUATION, "random_state": None},
        "fitted_on": "train_split_only",
        "feature_ranges": dict(bundle["feature_ranges"]),
        "metrics": {k: v for k, v in bundle["metrics"].items() if not k.startswith("mean_baseline")},
        "uncertainty": ("std across 5 CV fold models" if bundle["model_type"] == "mlp" else "std across trees")
        + " (rough; usually under-estimates)",
    }


def _is_lab(model_id):
    return isinstance(model_id, str) and model_id.startswith(LAB_PREFIX)


# ----------------------------------------------------------- bookkeeping ----

def _load_meta(model_id):
    if _is_lab(model_id):
        if model_id not in _lab_ids():
            raise FileNotFoundError(f"built-in model {model_id!r} not found; built-ins: {_lab_ids()}")
        return _lab_meta(model_id)
    path = MODEL_DIR / _safe_id(model_id) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"no saved model {model_id!r}; use list_models to see saved ids")
    return json.loads(path.read_text())


def _load_model(model_id):
    meta = _load_meta(model_id)
    if meta.get("built_in"):
        return meta, None
    return meta, joblib.load(MODEL_DIR / model_id / "model.joblib")


def _summary(m):
    return {k: m.get(k) for k in ("model_id", "model_type", "target", "features", "metrics", "created_utc")} | {
        "built_in": bool(m.get("built_in"))
    }


def list_models():
    models = [_summary(_lab_meta(i)) for i in _lab_ids()]
    if MODEL_DIR.exists():
        models += [_summary(json.loads(p.read_text())) for p in sorted(MODEL_DIR.glob("*/meta.json"))]
    return {"models": models}


def get_model_info(model_id):
    return _load_meta(model_id)


def compare_models(model_ids=None, target=None):
    """Side-by-side train/test RMSE, best (lowest test RMSE) first within
    each target. Only models of the same target are directly comparable, and
    only if they share the same data and split -- flagged in the result."""
    if model_ids:
        metas = [_load_meta(i) for i in model_ids]
    else:
        metas = [_lab_meta(i) for i in _lab_ids()]
        if MODEL_DIR.exists():
            metas += [json.loads(p.read_text()) for p in sorted(MODEL_DIR.glob("*/meta.json"))]
    if target:
        metas = [m for m in metas if m["target"] == target]
    skipped = [m["model_id"] for m in metas if not m.get("metrics")]
    metas = [m for m in metas if m.get("metrics")]
    if not metas:
        return {"comparison": [], "notes": ["no matching saved models"]}

    comparison, notes = [], []
    if skipped:
        notes.append(f"No RMSE available for {skipped} (metrics unavailable).")
    for tgt in sorted({m["target"] for m in metas}):
        group = sorted((m for m in metas if m["target"] == tgt), key=lambda m: m["metrics"]["test_rmse"])
        splits = {(m["data_path"], m["n_rows_used"], m["split"]["test_size"], m["split"]["random_state"],
                   tuple(m["features"])) for m in group}
        if len(splits) > 1:
            notes.append(f"{tgt}: models differ in data/split/features/evaluation (built-in lab models are "
                         "tested on the 4 later growths, new models on a random split), so RMSEs are not strictly comparable.")
        for rank, m in enumerate(group, start=1):
            mt = m["metrics"]
            comparison.append({
                "target": tgt,
                "rank_by_test_rmse": rank,
                "model_id": m["model_id"],
                "model_type": m["model_type"],
                "features": m["features"],
                "train_rmse": mt["train_rmse"],
                "test_rmse": mt["test_rmse"],
                "test_over_train_rmse": round(mt["test_rmse"] / mt["train_rmse"], 2) if mt["train_rmse"] > 0 else None,
                "n_train": m["n_train"],
                "n_test": m["n_test"],
                "evaluation": LAB_EVALUATION if m.get("built_in") else m.get("evaluation", "holdout split"),
            })
    return {"comparison": comparison, "notes": notes}


# ----------------------------------------------------------------- usage ----

def _predict_arrays(meta, bundle, X):
    """(mean, std) for a saved model. Lab models use models/predict.py. On-the-fly
    ones use GPR's predictive std, RFR's per-tree spread, or (KRR, MLP and
    auto_search models) the spread across 5 CV fold models."""
    if meta.get("built_in"):
        return predict_with_uncertainty(X, meta["target"])
    if "pipeline" in bundle:
        X = expand_features(X, meta["features"], meta.get("engineered"))
        return bundle["pipeline"].predict(X), np.array([f.predict(X) for f in bundle["folds"]]).std(axis=0)
    Xs = bundle["scaler"].transform(X)
    model = bundle["model"]
    if meta["model_type"] == "gpr":
        return model.predict(Xs, return_std=True)
    if meta["model_type"] in ("krr", "mlp"):
        folds = bundle.get("folds") or []
        spread = np.array([f.predict(Xs) for f in folds]).std(axis=0) if folds else np.zeros(len(Xs))
        return model.predict(Xs), spread
    tree = np.array([t.predict(Xs) for t in model.estimators_])
    return tree.mean(axis=0), tree.std(axis=0)


def predict(model_id, inputs):
    """inputs: list of {feature: value}. Returns mean and uncertainty (GPR:
    predictive std incl. learned noise; RFR: spread across trees), and flags
    inputs outside the training range."""
    meta, bundle = _load_model(model_id)
    if not inputs:
        raise ValueError("inputs must be a non-empty list of {feature: value} dicts")
    feats = meta["features"]
    rows = []
    for i, row in enumerate(inputs):
        missing = [f for f in feats if f not in row]
        if missing:
            raise ValueError(f"inputs[{i}] is missing feature(s) {missing}; model needs {feats}")
        rows.append([require_finite(row[f], f"inputs[{i}].{f}") for f in feats])
    X = np.array(rows, dtype=float)
    mean, std = _predict_arrays(meta, bundle, X)

    results = []
    for row, m, s in zip(rows, mean, std):
        out_of_range = [f for f, v in zip(feats, row)
                        if not meta["feature_ranges"][f][0] <= v <= meta["feature_ranges"][f][1]]
        entry = {"inputs": dict(zip(feats, row)), f"predicted_{meta['target']}": round(float(m), 4),
                 "uncertainty_std": round(float(s), 4)}
        if out_of_range:
            entry["extrapolation_warning"] = f"outside training range for: {out_of_range}"
        results.append(entry)
    return {"model_id": model_id, "model_type": meta["model_type"], "target": meta["target"],
            "predictions": results}


def optimize(model_id, goal="maximize", target_value=None, bounds=None, fixed=None,
             kappa=0.0, n_candidates=20000, top_k=5, seed=42):
    """Random-search the saved model for the best inputs.

    goal: "maximize" | "minimize" | "target" (get the prediction close to
    target_value). bounds: {feature: [lo, hi]} (default: training range);
    fixed: {feature: value} held constant. kappa trades exploitation for
    exploration: score uses mean -/+ kappa * std (optimistic bound)."""
    if goal not in ("maximize", "minimize", "target"):
        raise ValueError("goal must be 'maximize', 'minimize' or 'target'")
    if goal == "target":
        target_value = require_finite(target_value, "target_value")
    kappa = require_finite(kappa, "kappa")
    require_positive_int(n_candidates, "n_candidates")
    require_positive_int(top_k, "top_k")
    meta, bundle = _load_model(model_id)
    feats = meta["features"]
    bounds, fixed = bounds or {}, fixed or {}
    unknown = [f for f in list(bounds) + list(fixed) if f not in feats]
    if unknown:
        raise ValueError(f"unknown feature(s) {unknown}; model features are {feats}")

    rng = np.random.default_rng(seed)
    cols = []
    for f in feats:
        if f in fixed:
            cols.append(np.full(n_candidates, require_finite(fixed[f], f)))
        else:
            lo, hi = bounds.get(f, meta["feature_ranges"][f])
            lo, hi = require_finite(lo, f"{f} lower bound"), require_finite(hi, f"{f} upper bound")
            if lo >= hi:
                raise ValueError(f"bounds for {f} invalid: {lo} >= {hi}")
            cols.append(rng.uniform(lo, hi, n_candidates))
    X = np.column_stack(cols)
    mean, std = _predict_arrays(meta, bundle, X)

    if goal == "maximize":
        score = mean + kappa * std
    elif goal == "minimize":
        score = -(mean - kappa * std)
    else:
        score = -(np.abs(mean - target_value) - kappa * std)
    order = np.argsort(-score)[:top_k]
    return {
        "model_id": model_id, "goal": goal, "target_value": target_value, "kappa": kappa,
        "n_candidates_searched": n_candidates,
        "best": [
            {**{f: round(float(X[i, j]), 4) for j, f in enumerate(feats)},
             f"predicted_{meta['target']}": round(float(mean[i]), 4),
             "uncertainty_std": round(float(std[i]), 4)}
            for i in order
        ],
        "note": "Predictions come from a surrogate model, not a real experiment; treat as suggestions to verify.",
    }


# ------------------------------------------------------- saving on request ----

session_log = []


def log_result(tool, inputs, output):
    """Remembers a tool's output for this session; returns it unchanged. Nothing
    reaches the disk unless save_session_results is called."""
    session_log.append({"tool": tool, "inputs": inputs, "output": output,
                        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return output


def _session_model_ids():
    """Model ids this session trained or used, in first-seen order."""
    ids = []
    for entry in session_log:
        out = entry["output"] if isinstance(entry["output"], dict) else {}
        found = [t["model_id"] for t in out.get("trained", [])]
        found += [out[k] for k in ("model_id", "saved_model_id") if out.get(k)]
        ids += [i for i in found if i not in ids]
    return ids


def _markdown_table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


def save_session_results(name=None, notes=None):
    """Writes everything from this session to model_builder_results/<name>/:
    results.json (every tool input and output), comparison.csv (train/test RMSE of
    the models involved) and summary.md. Numbers come from the tool outputs, never
    from the LLM; its notes go in a separate, labelled section. Never overwrites."""
    if not session_log:
        raise ValueError("nothing to save yet: no model-builder results in this session")
    name = _safe_id(name) if name else "results_" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if name == MODEL_DIR.name:
        raise ValueError(f"{name!r} is reserved for saved models; choose another name")
    folder, n = RESULTS_ROOT / name, 2
    while folder.exists():
        folder, n = RESULTS_ROOT / f"{name}-{n}", n + 1
    folder.mkdir(parents=True)

    (folder / "results.json").write_text(json.dumps(session_log, indent=2, default=str))
    known = [i for i in _session_model_ids() if _is_lab(i) or (MODEL_DIR / i / "meta.json").exists()]
    comparison = compare_models(model_ids=known)["comparison"] if known else []
    written = ["results.json"]
    if comparison:
        pd.DataFrame(comparison).to_csv(folder / "comparison.csv", index=False)
        written.append("comparison.csv")

    parts = [f"# Model builder results: {name}", "",
             f"Saved {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
             f"{len(session_log)} tool result(s); full detail in results.json.", ""]
    if comparison:
        columns = ["target", "rank_by_test_rmse", "model_id", "model_type", "train_rmse", "test_rmse", "evaluation"]
        parts += ["## Model comparison", "", _markdown_table(comparison, columns), ""]
    searches = [e["output"] for e in session_log if e["tool"] == "run_autonomous_search" and "winner" in e["output"]]
    if searches:
        rows = [{"target": r["target"], "winner": r["winner"]["model_type"], "feature_set": r["winner"]["feature_set"],
                 "confirm_cv_rmse": r["winner"]["confirm_cv_rmse"], "split_rmse": r["winner"]["split_rmse"],
                 "leave_one_out_rmse": r["checks"]["leave_one_out_rmse"], "saved_model_id": r.get("saved_model_id", "")}
                for r in searches]
        parts += ["## Autonomous searches", "", _markdown_table(rows, list(rows[0])), ""]
    predictions = [p for e in session_log if e["tool"] == "predict_with_model" and isinstance(e["output"], dict)
                   for p in [{"model_id": e["output"].get("model_id"), **row} for row in e["output"].get("predictions", [])]]
    if predictions:
        parts += ["## Predictions", ""] + [f"- `{json.dumps(p, default=str)}`" for p in predictions] + [""]
    if notes:
        parts += ["## Assistant's notes (written by the LLM, not computed)", "", str(notes), ""]
    (folder / "summary.md").write_text("\n".join(parts))
    written.append("summary.md")
    return {"saved_to": str(folder), "files": written, "n_tool_results": len(session_log)}
