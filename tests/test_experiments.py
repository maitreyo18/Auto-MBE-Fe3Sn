import json

import numpy as np
import pandas as pd
import pytest

import experiments as ex


def test_errors_are_returned_not_raised():
    for kwargs in [
        {"sort_by": "bogus"}, {"columns": ["bogus"]}, {"closest_to": {"bogus": 1}}, {"filters": {"bogus": [1, 2]}},
        {"dataset": "nope"}, {"top_n": 0}, {"top_n": 2.5}, {"top_n": True}, {"filters": {"flux_ratio": [2, 1]}},
        {"filters": {"flux_ratio": [1]}}, {"filters": {"flux_ratio": 1}}, {"closest_to": {"flux_ratio": "abc"}},
        {"closest_to": {"flux_ratio": None}}, {"closest_to": {"flux_ratio": float("nan")}},
        {"closest_to": {"File": 1}}, {"compare_runs": ["x"]}, {"min_values": {"flux_ratio": "abc"}},
    ]:
        out = ex.analyze(**kwargs)
        assert "error" in out, kwargs
        assert out["available_columns"]


def test_unknown_column_suggests_close_match():
    assert "RHEED_Quality_Film" in ex.analyze(sort_by="RHEED_Qualty_Film")["error"] or \
        ex.analyze(sort_by="RHEED_Qualty_Film")["sorted_by"] == "RHEED_Quality_Film"
    assert "growthtime" in ex.analyze(sort_by="growthtim_e_x")["error"]


@pytest.mark.parametrize("name, expected", [
    ("film quality", "RHEED_Quality_Film"), ("EDS", "EDS_ratio"), ("stoichiometry", "EDS_ratio"),
    ("filament power", "filamentpower"), ("Flux Ratio", "flux_ratio"), ("growth_time", "growthtime"),
    ("substrate quality", "Substrate_quality"),
])
def test_loose_column_names_resolve_and_are_reported(name, expected):
    out = ex.analyze(sort_by=name, top_n=1)
    assert out["sorted_by"] == expected and out["columns_interpreted_as"] == {name: expected}


def test_numbers_as_strings_and_whole_floats_accepted():
    out = ex.analyze(closest_to={"flux_ratio": "0.6"}, top_n=2.0, compare_runs=["14", 24.0])
    assert len(out["closest_runs"]) == 2
    assert [r["run"] for r in out["compared_runs"]] == [14, 24]


def test_one_sided_filters():
    out = ex.analyze(min_values={"film quality": 60}, max_values={"growthtime": 500}, sort_by="film quality")
    assert out["n_runs"] > 0
    assert all(r["RHEED_Quality_Film"] >= 60 for r in out["top_runs"])
    df = ex.load_experiments()
    assert out["n_runs"] == len(df[(df.RHEED_Quality_Film >= 60) & (df.growthtime <= 500)])


def test_sort_is_repeatable_and_matches_pandas():
    df = ex.load_experiments()
    out = ex.analyze(sort_by="flux_ratio", ascending=True, top_n=21)
    assert [r["flux_ratio"] for r in out["top_runs"]] == sorted(round(v, 4) for v in df.flux_ratio)
    assert out == ex.analyze(sort_by="flux_ratio", ascending=True, top_n=21)


def test_closest_scaling_stops_large_columns_dominating():
    df = ex.load_experiments()
    row = df[df.run == 14].iloc[0]
    out = ex.analyze(closest_to={"growthtime": row.growthtime, "flux_ratio": row.flux_ratio}, top_n=1)
    assert out["closest_runs"][0]["run"] == 14 and out["closest_runs"][0]["distance"] == 0


def test_far_request_is_flagged_outside_range():
    out = ex.analyze(closest_to={"filamentpower": 50, "flux_ratio": 0.8})
    assert set(out["outside_observed_range"]) == {"filamentpower"} and "outside_note" in out
    assert "outside_observed_range" not in ex.analyze(closest_to={"filamentpower": 4, "flux_ratio": 0.8})


def test_top_n_larger_than_rows_and_small_groups():
    assert len(ex.analyze(sort_by="run", top_n=500)["top_runs"]) == 21
    one = ex.analyze(filters={"run": [14, 14]}, sort_by="run")
    assert one["n_runs"] == 1 and "statistics_note" in one and "correlations" not in one
    assert "statistics_note" not in ex.analyze()


def test_blank_values_are_handled_and_output_is_valid_json(monkeypatch, tmp_path):
    good = ex.load_experiments("initial").drop(columns=["dataset"])
    bad = good.copy()
    bad.loc[bad.run == 24, "RHEED_Quality_Film"] = np.nan
    bad.loc[bad.run == 14, "flux_ratio"] = np.nan
    path = tmp_path / "t.csv"
    bad.to_csv(path, index=False)
    monkeypatch.setattr(ex, "TRAIN_FILE", path)
    monkeypatch.setattr(ex, "TEST_FILE", tmp_path / "t.csv")
    out = ex.analyze(dataset="initial", sort_by="RHEED_Quality_Film", top_n=1, closest_to={"flux_ratio": 0.6}, compare_runs=[14, 24])
    assert out["top_runs"][0]["run"] != 24 and "sort_note" in out and "closest_note" in out
    assert all(r["run"] != 14 for r in out["closest_runs"])
    text = json.dumps(out, allow_nan=False)
    assert "NaN" not in text


def test_every_output_is_strict_json():
    out = ex.analyze(sort_by="EDS_ratio", closest_to={"EDS_ratio": 3}, compare_runs=[31, 14], columns=["EDS_ratio", "File"])
    json.dumps(out, allow_nan=False)


def test_missing_data_file_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(ex, "TEST_FILE", tmp_path / "missing.csv")
    assert "not found" in ex.analyze()["error"]


def test_imputed_flag_covers_only_returned_runs_and_eds_requests():
    assert ex.analyze(compare_runs=[31, 33])["eds_imputed_runs"] == [31]
    assert "eds_note" not in ex.analyze(compare_runs=[14, 33])
    assert "eds_note" not in ex.analyze(compare_runs=[31], columns=["flux_ratio"])
    assert ex.analyze(columns=["flux_ratio"], sort_by="EDS_ratio", top_n=21)["eds_imputed_runs"] == [31, 32, 35]
    assert ex.analyze()["eds_imputed_runs"] == [31, 32, 35]


def test_data_files_are_not_modified():
    before = pd.read_csv(ex.TRAIN_FILE).copy()
    ex.analyze(sort_by="EDS_ratio", filters={"flux_ratio": [0, 1]}, closest_to={"flux_ratio": 0.5})
    pd.testing.assert_frame_equal(before, pd.read_csv(ex.TRAIN_FILE))


def test_later_dataset_labels_rows():
    assert {r["dataset"] for r in ex.analyze(dataset="LATER", sort_by="run")["top_runs"]} == {"later"}
