import numpy as np
import joblib
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent


def load_scaler(scaler_path=None):
    return joblib.load(scaler_path or THIS_DIR / "feature_scaler.joblib")


def scale_features(X, scaler):
    return (np.asarray(X) - scaler["mean"]) / (scaler["std"] + 1e-8)


def predict_with_uncertainty(X, target_name, scaler=None):
    scaler = scaler or load_scaler()
    X_scaled = scale_features(X, scaler)

    loocv_models = joblib.load(THIS_DIR / f"rf_{target_name}_loocv.joblib")
    fold_preds = np.array([m.predict(X_scaled) for m in loocv_models])
    return fold_preds.mean(axis=0), fold_preds.std(axis=0)


def predict_with_tree_uncertainty(X, target_name, scaler=None):
    scaler = scaler or load_scaler()
    X_scaled = scale_features(X, scaler)

    rf_final = joblib.load(THIS_DIR / f"rf_{target_name}.joblib")
    tree_preds = np.array([t.predict(X_scaled) for t in rf_final.estimators_])
    return tree_preds.mean(axis=0), tree_preds.std(axis=0)


if __name__ == "__main__":
    scaler = load_scaler()
    X_example = np.array([scaler["mean"]])
    for target in ["EDS_ratio", "RHEED_Quality_Film"]:
        mean, std = predict_with_uncertainty(X_example, target, scaler)
        print(f"[{target}] LOOCV-ensemble  mean={mean[0]:.3f}  std={std[0]:.3f}")
        mean_t, std_t = predict_with_tree_uncertainty(X_example, target, scaler)
        print(f"[{target}] per-tree        mean={mean_t[0]:.3f}  std={std_t[0]:.3f}")
