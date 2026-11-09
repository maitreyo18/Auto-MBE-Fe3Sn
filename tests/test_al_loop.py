import joblib
import numpy as np
import pytest

import al_loop
from lab_models import EDS_FEATURES, FILM_FEATURES, load_train


def test_stoichiometry_score_peaks_at_target():
    assert al_loop.stoichiometry_score(al_loop.EDS_TARGET_RATIO) == pytest.approx(100.0)
    assert al_loop.stoichiometry_score(np.array([0.0, 3.0]))[1] == pytest.approx(100.0)


def test_calculate_ei_nonnegative_and_zero_std_edge_case():
    ei = al_loop.calculate_ei(mu=np.array([10.0]), std=np.array([0.0]), best_val=5.0)
    assert np.all(ei >= 0)


@pytest.mark.parametrize("n_iterations", [0, -1, 1.5])
def test_run_al_loop_rejects_invalid_n_iterations(n_iterations, tmp_path):
    with pytest.raises(ValueError):
        al_loop.run_al_loop(n_iterations=n_iterations, out_dir=tmp_path)


@pytest.mark.parametrize("n_initial", [0, 2, 18])
def test_run_al_loop_rejects_bad_n_initial(n_initial, tmp_path):
    with pytest.raises(ValueError):
        al_loop.run_al_loop(n_iterations=1, n_initial=n_initial, out_dir=tmp_path)


def test_defaults_start_from_ten_experiments():
    assert al_loop.DEFAULT_N_INITIAL == 10


def test_loocv_uses_the_same_models_with_one_pair_per_row():
    df = load_train().iloc[:6]
    film, eds, metrics = al_loop.fit_loocv_models(df)
    assert len(film) == len(eds) == 6
    assert type(film[0].steps[-1][1].regressor).__name__ == "MLPRegressor"
    assert type(eds[0]).__name__ == "RandomForestRegressor"
    assert metrics["film_loocv_rmse"] > 0 and metrics["eds_loocv_rmse"] > 0


def test_end_to_end_grows_from_ten_and_labels_with_ground_truth(tmp_path):
    results = al_loop.run_al_loop(n_iterations=2, out_dir=tmp_path)
    assert [r["n_observed"] for r in results] == [10, 11]
    for i, r in enumerate(results, start=1):
        folds = joblib.load(tmp_path / f"iteration_{i}" / "mlp_RHEED_Quality_Film_loocv.joblib")
        assert len(folds) == r["n_observed"]
        assert len(joblib.load(tmp_path / f"iteration_{i}" / "rf_EDS_ratio_loocv.joblib")) == r["n_observed"]
        assert (tmp_path / f"iteration_{i}" / f"al_iteration_{i}.png").exists()
    # the appended outcome is exactly what the ground-truth models say for that point
    row = load_train().iloc[[0]].copy()
    row["filamentpower"] = results[0]["suggested_filamentpower"]
    row["flux_ratio"] = results[0]["suggested_flux_ratio"]
    eds, film, score = al_loop.query_ground_truth(row)
    assert score.shape == (1,) and eds.shape == film.shape == (1,)
    assert set(EDS_FEATURES) <= set(FILM_FEATURES)
