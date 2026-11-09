"""Predictions from the lab's two trained models. X columns follow
lab_models.FEATURES[target]. Returns (mean, uncertainty_std).

Uncertainty is rough and usually under-estimates the real error: the spread
across 5 CV fold models for the MLP, and the spread across trees for the RFR."""
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from lab_models import CHECKPOINTS, EDS_TARGET, FEATURES, FILM_TARGET


def _load_checkpoint(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"model checkpoint not found at {path} -- run models/train.py first")
    return joblib.load(path)


@lru_cache(maxsize=None)
def load_bundle(target):
    if target not in CHECKPOINTS:
        raise ValueError(f"no lab model for target {target!r}; known: {list(CHECKPOINTS)}")
    return _load_checkpoint(CHECKPOINTS[target])


def _validate_X(X, target):
    X = np.asarray(X, dtype=float)
    n = len(FEATURES[target])
    if X.ndim != 2 or X.shape[1] != n:
        raise ValueError(f"X must have shape (n_samples, {n}) with features {FEATURES[target]}, got {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values")
    return X


def predict_with_uncertainty(X, target):
    X = _validate_X(X, target)
    bundle = load_bundle(target)
    if bundle["model_type"] == "rfr":
        trees = np.array([t.predict(X) for t in bundle["model"].estimators_])
        return trees.mean(axis=0), trees.std(axis=0)
    return bundle["model"].predict(X), np.array([m.predict(X) for m in bundle["folds"]]).std(axis=0)


def predict_film_quality(X):
    return predict_with_uncertainty(X, FILM_TARGET)


def predict_eds_ratio(X):
    return predict_with_uncertainty(X, EDS_TARGET)
