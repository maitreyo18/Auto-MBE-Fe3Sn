import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
for sub in ("agent", "models", "RHEED_images"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture
def synthetic_df():
    """Small, deterministic synthetic dataset shaped like data/train_compiled.csv
    (only the columns the agents actually consume)."""
    rng = np.random.default_rng(0)
    n = 12
    growthtime = rng.uniform(210, 660, n)
    filamentpower = rng.uniform(3.145, 5.414, n)
    flux_ratio = rng.uniform(0.467, 1.292, n)
    substrate_quality = rng.uniform(38, 92, n)
    eds_ratio = rng.uniform(1.5, 5.0, n)
    film_quality = rng.uniform(31, 88, n)
    return pd.DataFrame({
        "run": range(1, n + 1),
        "growthtime": growthtime,
        "filamentpower": filamentpower,
        "flux_ratio": flux_ratio,
        "Substrate_quality": substrate_quality,
        "EDS_ratio": eds_ratio,
        "RHEED_Quality_Film": film_quality,
    })


@pytest.fixture
def synthetic_csv(tmp_path, synthetic_df):
    path = tmp_path / "train_compiled.csv"
    synthetic_df.to_csv(path, index=False)
    return path
