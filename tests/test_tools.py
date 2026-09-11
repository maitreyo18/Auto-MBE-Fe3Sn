import pytest

import doe
import tools


def test_tools_registered():
    names = {t.name for t in tools.TOOLS}
    assert names == {
        "rheed_quality",
        "predict_quality",
        "run_active_learning_loop",
        "analyze_previous_experiments",
        "get_doe_suggested_ranges",
        "run_doe_experiment",
    }


def test_predict_quality_happy_path():
    result = tools.predict_quality.invoke({
        "growthtime": 400.0, "filamentpower": 4.0, "flux_ratio": 0.8, "substrate_quality": 70.0,
    })
    assert set(result) == {"EDS_ratio", "RHEED_Quality_Film"}
    assert isinstance(result["EDS_ratio"], float)
    assert isinstance(result["RHEED_Quality_Film"], float)


@pytest.mark.parametrize("bad_kwargs", [
    {"growthtime": float("nan"), "filamentpower": 4.0, "flux_ratio": 0.8, "substrate_quality": 70.0},
    {"growthtime": float("inf"), "filamentpower": 4.0, "flux_ratio": 0.8, "substrate_quality": 70.0},
])
def test_predict_quality_rejects_non_finite(bad_kwargs):
    with pytest.raises(Exception):
        tools.predict_quality.invoke(bad_kwargs)


def test_analyze_previous_experiments_summary():
    result = tools.analyze_previous_experiments.invoke({})
    assert result["n_runs"] > 0
    assert "summary_statistics" in result
    assert "correlations" in result


def test_analyze_previous_experiments_invalid_sort_by():
    result = tools.analyze_previous_experiments.invoke({"sort_by": "not_a_real_column"})
    assert "error" in result


def test_analyze_previous_experiments_sort_by_and_top_n():
    result = tools.analyze_previous_experiments.invoke({
        "sort_by": "RHEED_Quality_Film", "top_n": 3, "ascending": False,
    })
    assert len(result["top_runs"]) == 3


def test_get_doe_suggested_ranges_shape():
    result = tools.get_doe_suggested_ranges.invoke({})
    assert set(result) == set(doe.DOE_PARAMS)
    for p in doe.DOE_PARAMS:
        assert result[p]["min"] < result[p]["max"]


def test_run_doe_experiment_rejects_min_ge_max():
    with pytest.raises(ValueError):
        tools.run_doe_experiment.invoke({"growthtime_min": 500.0, "growthtime_max": 400.0})


def test_run_doe_experiment_uses_defaults_when_unset(monkeypatch):
    calls = {}

    def fake_run_doe(ranges, n_runs):
        calls["ranges"] = ranges
        calls["n_runs"] = n_runs
        return {"ok": True}

    monkeypatch.setattr(tools, "run_doe", fake_run_doe)
    result = tools.run_doe_experiment.invoke({})
    assert result == {"ok": True}
    defaults = doe.default_ranges()
    assert calls["ranges"] == defaults
    assert calls["n_runs"] == 12


def test_run_doe_experiment_partial_override(monkeypatch):
    calls = {}

    def fake_run_doe(ranges, n_runs):
        calls["ranges"] = ranges
        return {"ok": True}

    monkeypatch.setattr(tools, "run_doe", fake_run_doe)
    tools.run_doe_experiment.invoke({"growthtime_min": 300.0})
    defaults = doe.default_ranges()
    assert calls["ranges"]["growthtime"] == (300.0, defaults["growthtime"][1])
    assert calls["ranges"]["filamentpower"] == defaults["filamentpower"]


def test_run_active_learning_loop_rejects_bad_n_iterations():
    rows = [
        {"growthtime": 400, "filamentpower": 4.0, "flux_ratio": 0.8,
         "Substrate_quality": 70, "EDS_ratio": 3.0, "RHEED_Quality_Film": 60},
        {"growthtime": 450, "filamentpower": 4.2, "flux_ratio": 0.9,
         "Substrate_quality": 72, "EDS_ratio": 2.8, "RHEED_Quality_Film": 65},
    ]
    with pytest.raises(ValueError):
        tools.run_active_learning_loop.invoke({"initial_data": rows, "n_iterations": 0})
