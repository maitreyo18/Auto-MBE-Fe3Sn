import numpy as np
import pytest

import doe


@pytest.fixture(autouse=True)
def patched_data_path(monkeypatch, synthetic_csv):
    monkeypatch.setattr(doe, "DATA_PATH", synthetic_csv)


@pytest.fixture
def stub_predictions(monkeypatch):
    """Deterministic stand-in for the RF surrogates."""
    def fake_predict(X, target_name, scaler=None):
        X = np.asarray(X, dtype=float)
        if target_name == "EDS_ratio":
            mean = 3.0 + 0.001 * X[:, 0] - 0.01 * X[:, 1]
        else:
            mean = 50 + 0.02 * X[:, 0] + 5 * X[:, 1] - 10 * X[:, 2]
        return mean, np.zeros_like(mean)

    monkeypatch.setattr(doe, "predict_with_uncertainty", fake_predict)
    return fake_predict


def test_default_ranges_matches_csv(synthetic_df, patched_data_path):
    ranges = doe.default_ranges()
    for p in doe.DOE_PARAMS:
        assert ranges[p][0] == pytest.approx(synthetic_df[p].min())
        assert ranges[p][1] == pytest.approx(synthetic_df[p].max())


def test_default_ranges_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(doe, "DATA_PATH", tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError):
        doe.default_ranges()


def test_stoichiometry_score_peaks_at_target():
    assert doe.stoichiometry_score(doe.EDS_TARGET_RATIO) == pytest.approx(100.0)
    assert doe.stoichiometry_score(0.0) < doe.stoichiometry_score(2.5)


@pytest.mark.parametrize("bad_range", [(5.0, 5.0), (5.0, 1.0)])
def test_run_doe_rejects_invalid_range(bad_range, stub_predictions):
    with pytest.raises(ValueError):
        doe.run_doe(ranges={"growthtime": bad_range})


@pytest.mark.parametrize("n_runs", [0, -3, 1.5])
def test_run_doe_rejects_invalid_n_runs(n_runs, stub_predictions):
    with pytest.raises(ValueError):
        doe.run_doe(n_runs=n_runs)


def test_d_optimal_select_returns_unique_valid_indices(stub_predictions):
    ranges = doe.default_ranges()
    lo = np.array([ranges[p][0] for p in doe.DOE_PARAMS])
    hi = np.array([ranges[p][1] for p in doe.DOE_PARAMS])
    candidates = doe._build_candidate_pool(ranges, resolution=8)
    candidates_norm = doe._normalize(candidates, lo, hi)

    idx = doe.d_optimal_select(candidates_norm, n_runs=12, n_random_starts=2, seed=1)

    assert len(idx) == len(set(idx))
    assert all(0 <= i < len(candidates) for i in idx)


def test_d_optimal_select_improves_or_matches_det_of_a_random_subset(stub_predictions):
    ranges = doe.default_ranges()
    lo = np.array([ranges[p][0] for p in doe.DOE_PARAMS])
    hi = np.array([ranges[p][1] for p in doe.DOE_PARAMS])
    candidates = doe._build_candidate_pool(ranges, resolution=8)
    candidates_norm = doe._normalize(candidates, lo, hi)
    F_all = doe._quadratic_design_matrix(candidates_norm)

    idx = doe.d_optimal_select(candidates_norm, n_runs=12, n_random_starts=3, seed=2)
    det_selected = np.linalg.det(F_all[idx].T @ F_all[idx])

    rng = np.random.default_rng(99)
    random_idx = rng.choice(len(candidates), size=12, replace=False)
    det_random = np.linalg.det(F_all[random_idx].T @ F_all[random_idx])
    assert det_selected >= det_random


def test_run_doe_end_to_end_with_stubbed_models(stub_predictions):
    result = doe.run_doe(n_runs=12, resolution=8, seed=7)

    assert result["n_runs"] == 12
    assert len(result["design_table"]) == 12

    fit = result["quadratic_fit"]
    assert len(fit["coefficients"]) == len(doe.QUADRATIC_TERM_NAMES)
    for term in fit["coefficients"]:
        assert set(term) == {"term", "coefficient", "std_error"}
        assert term["std_error"] >= 0

    assert fit["residual_std_dev"] < 1.0
    assert fit["r_squared"] > 0.95
    assert fit["three_sigma"] == pytest.approx(3 * fit["residual_std_dev"], abs=1e-3)

    optimum = result["optimum"]
    for p in doe.DOE_PARAMS:
        lo, hi = result["ranges_used"][p]
        assert lo - 1e-3 <= optimum[p] <= hi + 1e-3


def test_run_doe_is_deterministic_given_seed(stub_predictions):
    r1 = doe.run_doe(n_runs=12, resolution=8, seed=123)
    r2 = doe.run_doe(n_runs=12, resolution=8, seed=123)
    assert r1["design_table"] == r2["design_table"]
    assert r1["optimum"] == r2["optimum"]


def test_run_doe_n_runs_floor_applies(stub_predictions):
    result = doe.run_doe(n_runs=1, resolution=8, seed=3)
    # auto-raised to n_terms + 2 = 12
    assert result["n_runs"] == 12
