import pytest

import doe
import tools


def test_tools_registered():
    names = {t.name for t in tools.TOOLS}
    assert names == {
        "rheed_quality",
        "run_active_learning_loop",
        "analyze_previous_experiments",
        "get_doe_suggested_ranges",
        "run_doe_experiment",
        "ask_model_builder",
    }


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


def test_analyze_covers_all_21_growths_by_default():
    assert tools.analyze_previous_experiments.invoke({})["n_runs"] == 21
    assert tools.analyze_previous_experiments.invoke({"dataset": "initial"})["n_runs"] == 17
    assert tools.analyze_previous_experiments.invoke({"dataset": "later"})["n_runs"] == 4


def test_analyze_sort_matches_pandas():
    result = tools.analyze_previous_experiments.invoke({"sort_by": "RHEED_Quality_Film", "top_n": 2})
    assert [r["run"] for r in result["top_runs"]] == [24, 14]
    low = tools.analyze_previous_experiments.invoke({"sort_by": "RHEED_Quality_Film", "ascending": True, "top_n": 1})
    assert low["top_runs"][0]["run"] == 31


def test_analyze_closest_to_exact_run_conditions_finds_that_run():
    result = tools.analyze_previous_experiments.invoke({
        "closest_to": {"growthtime": 570.0, "filamentpower": 5.413968, "flux_ratio": 0.578034682}, "top_n": 3,
    })
    rows = result["closest_runs"]
    assert rows[0]["run"] == 14 and rows[0]["distance"] == pytest.approx(0, abs=1e-3)
    assert [r["distance"] for r in rows] == sorted(r["distance"] for r in rows)
    assert rows[0]["difference_from_requested"]["growthtime"] == pytest.approx(0)


def test_analyze_closest_to_target_value():
    rows = tools.analyze_previous_experiments.invoke({"closest_to": {"EDS_ratio": 3.0}, "top_n": 3})["closest_runs"]
    gaps = [abs(r["EDS_ratio"] - 3.0) for r in rows]
    assert gaps == sorted(gaps) and gaps[0] <= 0.05


def test_analyze_filters_and_empty_result():
    result = tools.analyze_previous_experiments.invoke({"filters": {"filamentpower": [5.0, 6.0]}, "sort_by": "EDS_ratio"})
    assert 0 < result["n_runs"] < 21 and all(r["filamentpower"] >= 5 for r in result["top_runs"])
    empty = tools.analyze_previous_experiments.invoke({"filters": {"filamentpower": [100.0, 200.0]}})
    assert empty["n_runs"] == 0 and "note" in empty


def test_analyze_compare_runs_reports_spread_and_missing():
    result = tools.analyze_previous_experiments.invoke({"compare_runs": [14, 24, 999]})
    assert [r["run"] for r in result["compared_runs"]] == [14, 24]
    assert result["runs_not_found"] == [999] and result["compared_spread"]["RHEED_Quality_Film"] > 0


def test_analyze_flags_only_relevant_imputed_eds():
    result = tools.analyze_previous_experiments.invoke({"compare_runs": [31, 33]})
    assert result["eds_imputed_runs"] == [31]
    assert "eds_note" not in tools.analyze_previous_experiments.invoke({"compare_runs": [14, 33]})


def test_analyze_other_columns_and_bad_inputs():
    result = tools.analyze_previous_experiments.invoke({"columns": ["subcurve_variance"], "compare_runs": [14]})
    assert "subcurve_variance" in result["compared_runs"][0]
    assert "error" in tools.analyze_previous_experiments.invoke({"closest_to": {"bogus": 1.0}})
    assert "error" in tools.analyze_previous_experiments.invoke({"filters": {"flux_ratio": [2.0, 1.0]}})
    assert "error" in tools.analyze_previous_experiments.invoke({"top_n": 0})


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
    with pytest.raises(ValueError):
        tools.run_active_learning_loop.invoke({"n_iterations": 0})
