"""
Trains Random Forest surrogate models on the Auto-MBE training data.

    Inputs : growthtime, filamentpower, flux_ratio, Substrate_quality
    Targets: EDS_ratio (separate model), RHEED_Quality_Film (separate model)

EDS_ratio is the raw Fe:Sn stoichiometry ratio (target = 3.0), not a
derived closeness score.

Data source: ../data/train_compiled.csv (relative to this file)

Outputs (checkpoints), written to this same directory:
    rf_EDS_ratio.joblib               final model, fit on all data
    rf_EDS_ratio_loocv.joblib         list of N LOOCV fold models (one per left-out row)
    rf_RHEED_Quality_Film.joblib
    rf_RHEED_Quality_Film_loocv.joblib
    feature_scaler.joblib

Uncertainty at inference time: predict with every model in the *_loocv.joblib
list and take mean/std across them (see models/predict.py).
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error

THIS_DIR = Path(__file__).resolve().parent
DATA_FILE = THIS_DIR.parent / "data" / "train_compiled.csv"

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Training data not found at {DATA_FILE}")

FEATURES = ["growthtime", "filamentpower", "flux_ratio", "Substrate_quality"]
TARGETS = ["EDS_ratio", "RHEED_Quality_Film"]

df = pd.read_csv(DATA_FILE)
df_filled = df.copy()
for col in ["growthtime", "filamentpower", "flux_ratio", "Substrate_quality", "EDS_ratio", "RHEED_Quality_Film"]:
    if col in df_filled.columns:
        df_filled[col] = df_filled[col].fillna(df_filled[col].mean())

X = df_filled[FEATURES].values

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_scaled = (X - X_mean) / (X_std + 1e-8)

joblib.dump({"mean": X_mean, "std": X_std, "features": FEATURES}, THIS_DIR / "feature_scaler.joblib")


def train_and_save(target_name):
    y = df_filled[target_name].values

    loo = LeaveOneOut()
    y_pred_cv = np.zeros_like(y, dtype=float)
    loocv_models = []

    for train_index, test_index in loo.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train = y[train_index]
        rf_fold = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
        rf_fold.fit(X_train, y_train)
        y_pred_cv[test_index] = rf_fold.predict(X_test)
        loocv_models.append(rf_fold)

    mae = mean_absolute_error(y, y_pred_cv)
    print(f"[{target_name}] LOOCV MAE = {mae:.3f}")

    loocv_path = THIS_DIR / f"rf_{target_name}_loocv.joblib"
    joblib.dump(loocv_models, loocv_path)
    print(f"[{target_name}] Saved {len(loocv_models)} LOOCV fold models -> {loocv_path}")

    rf_final = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    rf_final.fit(X_scaled, y)

    out_path = THIS_DIR / f"rf_{target_name}.joblib"
    joblib.dump(rf_final, out_path)
    print(f"[{target_name}] Saved checkpoint -> {out_path}")

    return mae


def main():
    print(f"Loaded {len(df)} training rows from {DATA_FILE}")
    print(f"Inputs : {FEATURES}")
    print(f"Targets: {TARGETS}\n")

    for target_name in TARGETS:
        train_and_save(target_name)
        print()

    print("Done. Checkpoints and feature_scaler.joblib saved in:", THIS_DIR)


if __name__ == "__main__":
    main()
