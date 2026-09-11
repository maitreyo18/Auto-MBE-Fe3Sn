import numpy as np
import pytest

import al_loop


def test_stoichiometry_score_peaks_at_target():
    assert al_loop.stoichiometry_score(al_loop.EDS_TARGET_RATIO) == pytest.approx(100.0)
    assert al_loop.stoichiometry_score(np.array([0.0, 3.0]))[1] == pytest.approx(100.0)


def test_calculate_ei_nonnegative_and_zero_std_edge_case():
    ei = al_loop.calculate_ei(mu=np.array([10.0]), std=np.array([0.0]), best_val=5.0)
    assert np.all(ei >= 0)


def test_run_al_loop_rejects_empty_initial_data():
    with pytest.raises(ValueError):
        al_loop.run_al_loop([], n_iterations=1)


def test_run_al_loop_rejects_single_row():
    row = {
        "growthtime": 400, "filamentpower": 4.0, "flux_ratio": 0.8,
        "Substrate_quality": 70, "EDS_ratio": 3.0, "RHEED_Quality_Film": 60,
    }
    with pytest.raises(ValueError):
        al_loop.run_al_loop([row], n_iterations=1)


def test_run_al_loop_rejects_missing_columns():
    rows = [
        {"growthtime": 400, "filamentpower": 4.0, "flux_ratio": 0.8, "Substrate_quality": 70},
        {"growthtime": 450, "filamentpower": 4.2, "flux_ratio": 0.9, "Substrate_quality": 72},
    ]
    with pytest.raises(ValueError):
        al_loop.run_al_loop(rows, n_iterations=1)


@pytest.mark.parametrize("n_iterations", [0, -1, 1.5])
def test_run_al_loop_rejects_invalid_n_iterations(n_iterations):
    rows = [
        {"growthtime": 400, "filamentpower": 4.0, "flux_ratio": 0.8,
         "Substrate_quality": 70, "EDS_ratio": 3.0, "RHEED_Quality_Film": 60},
        {"growthtime": 450, "filamentpower": 4.2, "flux_ratio": 0.9,
         "Substrate_quality": 72, "EDS_ratio": 2.8, "RHEED_Quality_Film": 65},
    ]
    with pytest.raises(ValueError):
        al_loop.run_al_loop(rows, n_iterations=n_iterations)


def test_run_al_loop_end_to_end(tmp_path, synthetic_df):
    rows = synthetic_df.to_dict(orient="records")
    results = al_loop.run_al_loop(rows, n_iterations=2, out_dir=tmp_path)

    assert len(results) == 2
    for i, result in enumerate(results, start=1):
        assert result["iteration"] == i
        assert "suggested_filamentpower" in result
        assert "suggested_flux_ratio" in result
        assert "predicted_EDS_ratio" in result
        assert "predicted_RHEED_Quality_Film" in result
        assert (tmp_path / f"iteration_{i}").exists()
        assert (tmp_path / f"iteration_{i}" / f"al_iteration_{i}.png").exists()
