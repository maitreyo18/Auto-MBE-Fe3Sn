"""The lab's two models and their data split.

    RHEED_Quality_Film -> small MLP (one hidden layer of 3 nodes, alpha=1, batch
                          size 4) on 10 inputs known before growth.
    EDS_ratio          -> Random Forest (100 trees, max_depth=3) on 4 inputs.

models/train.py fits them on the 17 initial growths (data/train_compiled.csv)
and tests them on the 4 later ones (data/test_compiled.csv). These trained
models are the "ground truth" for predictions, the AL loop and the DoE. The AL
loop refits these same model definitions on the rows it has observed.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
TRAIN_FILE = THIS_DIR.parent / "data" / "train_compiled.csv"
TEST_FILE = THIS_DIR.parent / "data" / "test_compiled.csv"

FILM_TARGET = "RHEED_Quality_Film"
EDS_TARGET = "EDS_ratio"
GROWTH_INPUTS = ["growthtime", "filamentpower", "flux_ratio"]
SUBSTRATE_RAW = ["subcfwhm", "sublfwhm", "subrfwhm", "submean_width", "subwidth_variance", "subcurve_variance"]
EDS_FEATURES = GROWTH_INPUTS + ["Substrate_quality"]
FILM_FEATURES = EDS_FEATURES + SUBSTRATE_RAW
FEATURES = {FILM_TARGET: FILM_FEATURES, EDS_TARGET: EDS_FEATURES}
CHECKPOINTS = {
    FILM_TARGET: THIS_DIR / "mlp_RHEED_Quality_Film.joblib",
    EDS_TARGET: THIS_DIR / "rf_EDS_ratio.joblib",
}
MODEL_TYPE = {FILM_TARGET: "mlp", EDS_TARGET: "rfr"}
MLP_PARAMS = {
    "hidden_layer_sizes": (3,), "activation": "tanh", "solver": "adam", "batch_size": 4,
    "learning_rate_init": 0.01, "alpha": 1.0, "max_iter": 300,
}
RFR_PARAMS = {"n_estimators": 100, "max_depth": 3}


def build_film_model(seed=0):
    """Unfitted film-quality MLP; scales its own inputs and target."""
    return make_pipeline(
        StandardScaler(),
        TransformedTargetRegressor(regressor=MLPRegressor(random_state=seed, **MLP_PARAMS), transformer=StandardScaler()),
    )


def build_eds_model(seed=42):
    """Unfitted stoichiometry Random Forest."""
    return RandomForestRegressor(random_state=seed, **RFR_PARAMS)


BUILDERS = {FILM_TARGET: build_film_model, EDS_TARGET: build_eds_model}


def load_train():
    return pd.read_csv(TRAIN_FILE)


def load_test():
    return pd.read_csv(TEST_FILE)
