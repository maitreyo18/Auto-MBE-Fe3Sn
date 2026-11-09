"""Trains the lab's two ground-truth models on the 17 initial growths, tests
them on the 4 later growths and saves one checkpoint per model.

    models/mlp_RHEED_Quality_Film.joblib
    models/rf_EDS_ratio.joblib

The 5-fold CV on the 17 rows estimates error too. For the MLP, its 5 fold models
are kept as the prediction uncertainty; the Random Forest uses its per-tree std.
EDS was imputed for runs 32 and 35 in the source data, so EDS test metrics are
reported with and without them.
"""
import warnings

import joblib
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

from lab_models import (
    BUILDERS, CHECKPOINTS, EDS_TARGET, FEATURES, FILM_TARGET, MLP_PARAMS, MODEL_TYPE, RFR_PARAMS,
    load_test, load_train,
)

CV_FOLDS = 5


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def train_one(target, train, test):
    feats = FEATURES[target]
    build = BUILDERS[target]
    X, y = train[feats].to_numpy(dtype=float), train[target].to_numpy(dtype=float)
    X_te, y_te = test[feats].to_numpy(dtype=float), test[target].to_numpy(dtype=float)
    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_pred = cross_val_predict(build(), X, y, cv=kfold)
        model = build().fit(X, y)
        folds = [build().fit(X[tr], y[tr]) for tr, _ in kfold.split(X)] if MODEL_TYPE[target] == "mlp" else None

    mean_cv = np.zeros_like(y)
    for tr, te in kfold.split(X):
        mean_cv[te] = y[tr].mean()
    pred_te = model.predict(X_te)
    metrics = {
        "train_rmse": round(rmse(y, model.predict(X)), 4),
        "test_rmse": round(rmse(y_te, pred_te), 4),
        "train_r2": round(float(r2_score(y, model.predict(X))), 4),
        "test_r2": round(float(r2_score(y_te, pred_te)), 4),
        "mean_baseline_test_rmse": round(rmse(y_te, np.full(len(y_te), y.mean())), 4),
        "cv5_rmse": round(rmse(y, cv_pred), 4),
        "mean_baseline_cv5_rmse": round(rmse(y, mean_cv), 4),
        "evaluation": "train on 17 initial growths, test on 4 later growths",
    }
    if target == EDS_TARGET:
        measured = ~test["eds_imputed"].to_numpy(dtype=bool)
        metrics["test_rmse_measured_eds_only"] = round(rmse(y_te[measured], pred_te[measured]), 4)
        metrics["n_test_measured_eds"] = int(measured.sum())
    params = {**MLP_PARAMS, "hidden_layer_sizes": list(MLP_PARAMS["hidden_layer_sizes"])} \
        if MODEL_TYPE[target] == "mlp" else dict(RFR_PARAMS)
    joblib.dump({
        "model": model, "folds": folds, "target": target, "features": feats,
        "model_type": MODEL_TYPE[target], "params": params, "n_train": len(train), "n_test": len(test),
        "feature_ranges": {f: [float(train[f].min()), float(train[f].max())] for f in feats},
        "metrics": metrics,
    }, CHECKPOINTS[target])
    print(f"[{target}] saved {CHECKPOINTS[target].name}\n  {metrics}")
    print(f"  test predicted {np.round(pred_te, 2).tolist()} actual {np.round(y_te, 2).tolist()}")
    return metrics


def main():
    train, test = load_train(), load_test()
    print(f"Train: {len(train)} growths. Test: {len(test)} growths (runs {test['run'].tolist()}).")
    for target in (FILM_TARGET, EDS_TARGET):
        train_one(target, train, test)


if __name__ == "__main__":
    main()
