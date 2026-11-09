"""Autonomous model search: given a target, tries feature sets and hyperparameters
for RFR, GPR, KRR and MLP inside a time budget, stress-tests the best ones, saves
the winner and reports how it held up.

Steps, all time-boxed:
  1. Search: score every (model, feature set) trial with repeated 5-fold CV.
  2. Confirm: rescore the top trials on fresh shuffles and random 80/20 splits.
  3. Check: leave-one-out, re-selecting among the top trials on training rows
     only, a shuffled-target test, and a last-rows holdout.

Run it directly, without any LLM:
    python agent/auto_search.py --target RHEED_Quality_Film --minutes 5
"""
import argparse
import json
import random
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import model_builder as mb  # also puts models/ on sys.path
from lab_models import EDS_FEATURES, EDS_TARGET, FILM_TARGET, SUBSTRATE_RAW
from validation import require_finite

ALL_FAMILIES = ("rfr", "gpr", "krr", "mlp")
POST_GROWTH_COLUMNS = {EDS_TARGET, FILM_TARGET}
TOP_N = 5
MIN_ROWS = 8


def _scaled_y(model):
    return TransformedTargetRegressor(regressor=model, transformer=StandardScaler())


def _gpr(kernel_name, n_features):
    length = np.ones(n_features) if kernel_name == "rbf_ard" else 1.0
    base = {
        "rbf": lambda: RBF(length, (1e-1, 1e2)),
        "rbf_ard": lambda: RBF(length, (1e-1, 1e2)),
        "matern52": lambda: Matern(length, (1e-1, 1e2), nu=2.5),
    }[kernel_name]()
    kernel = ConstantKernel(1.0, (1e-2, 1e1)) * base + WhiteKernel(0.5, (1e-1, 1e1))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=1, random_state=0)


def build_trial(family, params, n_features):
    """Unfitted pipeline that scales its own inputs."""
    if family == "rfr":
        model = RandomForestRegressor(n_estimators=100, random_state=42, **params)
    elif family == "gpr":
        model = _gpr(params["kernel"], n_features)
    elif family == "krr":
        model = _scaled_y(KernelRidge(**params))
    else:
        model = _scaled_y(MLPRegressor(
            hidden_layer_sizes=(params["hidden_nodes"],), alpha=params["alpha"], activation="tanh",
            solver="adam", batch_size=4, learning_rate_init=0.01, max_iter=300, random_state=0,
        ))
    return make_pipeline(StandardScaler(), model)


def hyperparameter_grid(family):
    if family == "rfr":
        return [{"max_depth": d, "min_samples_leaf": leaf, "max_features": m}
                for d in (2, 3, 5, None) for leaf in (1, 3) for m in (0.5, 1.0)]
    if family == "gpr":
        return [{"kernel": k} for k in ("rbf", "matern52", "rbf_ard")]
    if family == "krr":
        return [{"kernel": "rbf", "alpha": a, "gamma": g} for a in (0.01, 0.1, 0.3, 1.0, 3.0)
                for g in (0.01, 0.03, 0.1, 0.3)] + [{"kernel": "linear", "alpha": a} for a in (1.0, 10.0)]
    return [{"hidden_nodes": h, "alpha": a} for h in (2, 3, 5) for a in (0.1, 1.0, 3.0)]


def engineered_names(X, features):
    """log, square and pairwise products of the first 4 inputs."""
    head = features[:4]
    names = [f"log({f})" for f in head if np.all(X[:, features.index(f)] > 0)]
    names += [f"{f}^2" for f in head]
    names += [f"{a}*{b}" for i, a in enumerate(head) for b in head[i + 1:]]
    return names


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def cv_rmse(make, X, y, seeds):
    return float(np.mean([
        _rmse(y, cross_val_predict(make(), X, y, cv=KFold(5, shuffle=True, random_state=s))) for s in seeds
    ]))


def split_rmse(make, X, y, seeds):
    """Test RMSE on each random 80/20 split."""
    out = []
    for s in seeds:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=s)
        out.append(_rmse(y[te], make().fit(X[tr], y[tr]).predict(X[te])))
    return np.array(out)


def _resolve_inputs(target, features, extra_features, df, numeric):
    if target == FILM_TARGET or target == EDS_TARGET:
        features = features or list(EDS_FEATURES)
        if extra_features is None:
            extra_features = [c for c in SUBSTRATE_RAW if c in numeric]
    if not features:
        return None, None
    extra_features = [c for c in (extra_features or []) if c not in features]
    unknown = [c for c in [target] + list(features) + extra_features if c not in numeric]
    if unknown:
        raise ValueError(f"column(s) {unknown} are not numeric columns; available: {numeric}")
    if target in list(features) + extra_features:
        raise ValueError(f"target {target!r} cannot also be an input")
    return list(features), extra_features


def run_search(target=None, features=None, extra_features=None, model_types=None, data_path="all",
               minutes=5.0, engineered=True, save_best=True, holdout_last=4, seed=0, save_report=False):
    """Explores models for `target` for about `minutes` and returns a report dict.
    The winner is saved like any trained model so it can be reused; the report is
    written to a file only with save_report=True. For the lab targets, inputs
    default to the 4 growth inputs plus the 6 raw substrate numbers (all known
    before growth); any other target or dataset needs them from the caller."""
    t0 = time.time()
    minutes = require_finite(minutes, "minutes")
    if not 0 < minutes <= 60:
        raise ValueError(f"minutes must be in (0, 60], got {minutes}")
    budget = minutes * 60
    path, df = mb._load_df(data_path)
    numeric = list(df.select_dtypes(include="number").columns)
    if not target:
        return {"needs_user_input": True, "missing": ["target"], "available_numeric_columns": numeric,
                "message": "Ask the user which column to predict."}
    if target not in numeric:
        raise ValueError(f"target {target!r} is not a numeric column; available: {numeric}")
    features, extra_features = _resolve_inputs(target, features, extra_features, df, numeric)
    if features is None:
        return {"needs_user_input": True, "missing": ["features"], "available_numeric_columns": numeric,
                "message": "Ask the user which training variables to use."}
    families = [m.lower() for m in (model_types or ALL_FAMILIES)]
    bad = [m for m in families if m not in ALL_FAMILIES]
    if bad:
        raise ValueError(f"unsupported model type(s) {bad}: only {list(ALL_FAMILIES)} are supported")

    inputs = features + extra_features
    clean = df[inputs + [target]].dropna()
    if len(clean) < MIN_ROWS:
        raise ValueError(f"only {len(clean)} complete rows -- need at least {MIN_ROWS}")
    n_dropped = len(df) - len(clean)
    y = clean[target].to_numpy(dtype=float)
    X_inputs = clean[inputs].to_numpy(dtype=float)

    eng = engineered_names(X_inputs, inputs) if engineered else []
    feature_sets = {"base": ([f for f in features], [])}
    if extra_features:
        feature_sets["base+extra"] = (inputs, [])
    if eng:
        base_eng = [n for n in eng if all(p in features for p in _parents(n))]
        if base_eng:
            feature_sets["base+engineered"] = (features, base_eng)
        if extra_features:
            feature_sets["base+extra+engineered"] = (inputs, eng)

    def matrix(set_name):
        cols, names = feature_sets[set_name]
        return mb.expand_features(clean[cols].to_numpy(dtype=float), cols, names)

    matrices = {k: matrix(k) for k in feature_sets}
    warnings.filterwarnings("ignore")

    # --- 1. search: first trial of every family runs regardless of the budget, then round-robin
    rng = random.Random(seed)
    queues = {}
    for fam in families:
        trials = [(fam, p, fs) for p in hyperparameter_grid(fam) for fs in feature_sets]
        rng.shuffle(trials)
        queues[fam] = trials
    results, skipped, i = [], 0, 0
    while any(queues.values()):
        fam = families[i % len(families)]
        i += 1
        if not queues[fam]:
            continue
        if results and len(results) >= len(families) and time.time() - t0 > 0.6 * budget:
            skipped = sum(len(q) for q in queues.values())
            break
        family, params, set_name = queues[fam].pop()
        X = matrices[set_name]
        score = cv_rmse(lambda: build_trial(family, params, X.shape[1]), X, y, range(2))
        results.append({"model_type": family, "params": params, "feature_set": set_name, "cv_rmse": round(score, 4)})
    results.sort(key=lambda r: r["cv_rmse"])
    per_family = {f: min((r["cv_rmse"] for r in results if r["model_type"] == f), default=None) for f in families}

    # --- 2. confirm the top trials on data partitions the search did not use
    top = results[:TOP_N]
    confirm_seeds, split_seeds = range(100, 104), range(1000, 1020)
    confirmed = []
    for r in top:
        if confirmed and time.time() - t0 > 0.8 * budget:
            break
        X = matrices[r["feature_set"]]
        make = _maker(r, X)
        cv = cv_rmse(make, X, y, confirm_seeds)
        confirmed.append({**r, "confirm_cv_rmse": round(cv, 4),
                          "split_rmse": round(float(split_rmse(make, X, y, split_seeds).mean()), 4)})
    confirmed.sort(key=lambda r: r["confirm_cv_rmse"])
    winner = confirmed[0]
    win_X = matrices[winner["feature_set"]]
    win_make = _maker(winner, win_X)

    # --- 3. checks
    checks = {}
    loo = cross_val_predict(win_make(), win_X, y, cv=LeaveOneOut())
    checks["leave_one_out_rmse"] = round(_rmse(y, loo), 4)

    if holdout_last and len(y) - holdout_last >= MIN_ROWS:
        tr, te = np.arange(len(y) - holdout_last), np.arange(len(y) - holdout_last, len(y))
        checks[f"last_{holdout_last}_rows_rmse"] = round(_rmse(y[te], win_make().fit(win_X[tr], y[tr]).predict(win_X[te])), 4)

    nested_m = []
    for s in range(2000, 2020):
        if nested_m and time.time() - t0 > 0.9 * budget:
            break
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=s)
        scores = [(cv_rmse(_maker(r, matrices[r["feature_set"]]), matrices[r["feature_set"]][tr], y[tr], [s]), r)
                  for r in confirmed]
        best = min(scores, key=lambda t: t[0])[1]
        B = matrices[best["feature_set"]]
        nested_m.append(_rmse(y[te], _maker(best, B)().fit(B[tr], y[tr]).predict(B[te])))
    checks["selection_on_training_rows_only"] = {
        "rmse": round(float(np.mean(nested_m)), 4), "n_splits": len(nested_m),
    }

    shuffle_rng = np.random.default_rng(seed)
    real = cv_rmse(win_make, win_X, y, [0])
    perm = []
    while (len(perm) < 5 or time.time() - t0 < budget) and len(perm) < 200:
        perm.append(_rmse(z := shuffle_rng.permutation(y), cross_val_predict(win_make(), win_X, z, cv=KFold(5, shuffle=True, random_state=len(perm)))))
    p_value = float((np.array(perm) <= real).mean())
    checks["shuffled_target_test"] = {"real_rmse": round(real, 4), "shuffled_mean_rmse": round(float(np.mean(perm)), 4),
                                      "fraction_shuffled_as_good": round(p_value, 3), "n_shuffles": len(perm)}

    report = {
        "target": target,
        "data": str(path),
        "n_rows": len(y),
        "n_rows_dropped_missing": n_dropped,
        "inputs": {"features": features, "extra_features": extra_features, "engineered_available": eng},
        "budget_minutes": minutes,
        "trials_run": len(results),
        "trials_not_reached": skipped,
        "best_cv_by_family": per_family,
        "top_trials": confirmed,
        "winner": winner,
        "checks": checks,
        "caveat": "The winner is chosen from many trials on these same rows, so its scores are optimistic. "
                  "selection_on_training_rows_only is the most conservative figure, but it only re-selects among "
                  "the top trials, which were themselves picked using all rows, and the shuffle test does not "
                  "account for the search. Confirm on new growths.",
    }
    if n_dropped:
        report["note_dropped"] = f"{n_dropped} row(s) dropped for missing values."
    leaked = sorted(POST_GROWTH_COLUMNS & set(inputs))
    if leaked:
        report["note_post_growth_inputs"] = f"{leaked} are measured after growth, so this model cannot plan a growth."

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    mb.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if save_best:
        report["saved_model_id"] = _save_winner(winner, feature_sets, clean, target, str(path), checks, run_id)
    report["seconds_used"] = round(time.time() - t0, 1)
    if save_report:
        path = mb.RESULTS_ROOT / f"search_{target}_{run_id}.json"
        n = 2
        while path.exists():
            path = mb.RESULTS_ROOT / f"search_{target}_{run_id}-{n}.json"
            n += 1
        path.write_text(json.dumps(report, indent=2, default=str))
        report["report_path"] = str(path)
    return report


def _parents(name):
    if name.startswith("log("):
        return [name[4:-1]]
    if name.endswith("^2"):
        return [name[:-2]]
    return name.split("*")


def _maker(result, X):
    return lambda: build_trial(result["model_type"], result["params"], X.shape[1])


def _save_winner(winner, feature_sets, clean, target, data, checks, run_id):
    """Refits the winner on all rows and saves it so it can be used like any other model."""
    cols, names = feature_sets[winner["feature_set"]]
    X_raw = clean[cols].to_numpy(dtype=float)
    X = mb.expand_features(X_raw, cols, names)
    y = clean[target].to_numpy(dtype=float)
    make = _maker(winner, X)
    pipeline = make().fit(X, y)
    folds = [make().fit(X[tr], y[tr]) for tr, _ in KFold(5, shuffle=True, random_state=0).split(X)]
    model_id = mb._unique_id(f"auto_{winner['model_type']}_{target}_{run_id}")
    meta = {
        "model_id": model_id,
        "model_type": winner["model_type"],
        "target": target,
        "features": cols,
        "engineered": names,
        "hyperparameters": winner["params"],
        "data_path": data,
        "n_rows_used": len(y),
        "n_train": len(y),
        "n_test": len(y) // 5,
        "split": {"test_size": "5-fold CV", "random_state": None},
        "evaluation": "5-fold CV (auto_search)",
        "fitted_on": "all_rows",
        "feature_ranges": {c: [float(clean[c].min()), float(clean[c].max())] for c in cols},
        "metrics": {"train_rmse": round(_rmse(y, pipeline.predict(X)), 4), "test_rmse": winner["confirm_cv_rmse"],
                    "leave_one_out_rmse": checks["leave_one_out_rmse"]},
        "warnings": [],
        "uncertainty": "std across 5 CV fold models (rough; usually under-estimates)",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = mb.MODEL_DIR / model_id
    out.mkdir(parents=True)
    joblib.dump({"pipeline": pipeline, "folds": folds}, out / "model.joblib")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return model_id


def main():
    parser = argparse.ArgumentParser(description="Autonomous RFR/GPR/KRR/MLP search, no LLM needed.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--features", nargs="*", help="main inputs (optional for the lab targets)")
    parser.add_argument("--extra-features", nargs="*", help="optional additional inputs to try")
    parser.add_argument("--models", nargs="*", choices=ALL_FAMILIES)
    parser.add_argument("--data", default="all", help='CSV path, or "all" for the 21 lab growths')
    parser.add_argument("--minutes", type=float, default=5.0)
    parser.add_argument("--no-engineered", action="store_true")
    parser.add_argument("--save-results", action="store_true", help="also write the report to model_builder_results/")
    args = parser.parse_args()
    report = run_search(
        target=args.target, features=args.features, extra_features=args.extra_features, model_types=args.models,
        data_path=args.data, minutes=args.minutes, engineered=not args.no_engineered,
        save_report=args.save_results,
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
