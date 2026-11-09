import json

import numpy as np
import pandas as pd
import pytest

import auto_search as auto
import model_agent
import model_builder as mb

FEATS = ["a", "b", "c"]


@pytest.fixture(autouse=True)
def isolated_model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(mb, "RESULTS_ROOT", tmp_path / "model_builder_results")
    monkeypatch.setattr(mb, "MODEL_DIR", tmp_path / "model_builder_results" / "models")
    mb.session_log.clear()


@pytest.fixture
def csv_path(tmp_path):
    rng = np.random.default_rng(3)
    n = 30
    df = pd.DataFrame({"a": rng.uniform(1, 5, n), "b": rng.uniform(1, 5, n), "c": rng.uniform(1, 5, n), "extra": rng.uniform(1, 5, n)})
    df["y"] = 3 * df.a - 2 * df.b + df.extra + rng.normal(0, 0.2, n)
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_missing_target_or_features_asks_user(csv_path):
    assert auto.run_search(target=None, data_path=csv_path)["missing"] == ["target"]
    assert auto.run_search(target="y", data_path=csv_path)["missing"] == ["features"]


@pytest.mark.parametrize("kwargs", [
    {"target": "nope", "features": FEATS},
    {"target": "y", "features": ["y"]},
    {"target": "y", "features": ["missing"]},
    {"target": "y", "features": FEATS, "model_types": ["xgboost"]},
    {"target": "y", "features": FEATS, "minutes": 0},
    {"target": "y", "features": FEATS, "minutes": 100},
])
def test_rejects_bad_inputs(csv_path, kwargs):
    with pytest.raises(ValueError):
        auto.run_search(data_path=csv_path, **kwargs)


def test_engineered_names_skip_log_of_nonpositive():
    X = np.array([[1.0, -1.0], [2.0, 3.0]])
    names = auto.engineered_names(X, ["p", "q"])
    assert "log(p)" in names and "log(q)" not in names and "p*q" in names and "q^2" in names


def test_expand_features_matches_formulas():
    X = np.array([[2.0, 3.0]])
    out = mb.expand_features(X, ["p", "q"], ["log(p)", "q^2", "p*q"])
    assert out[0].tolist() == pytest.approx([2.0, 3.0, np.log(2.0), 9.0, 6.0])


def test_search_covers_every_family_and_reports_no_mean_comparison(csv_path):
    r = auto.run_search("y", FEATS, ["extra"], data_path=csv_path, minutes=0.05, holdout_last=4)
    assert all(v is not None for v in r["best_cv_by_family"].values()) and len(r["best_cv_by_family"]) == 4
    assert r["winner"]["confirm_cv_rmse"] < 0.5 * np.std(pd.read_csv(csv_path)["y"])  # clear signal in this data
    assert "mean" not in json.dumps({k: v for k, v in r.items() if k != "checks"}).replace("shuffled_mean_rmse", "")
    assert not [k for k in r["checks"] if "mean" in k]
    assert "selection_on_training_rows_only" in r["checks"] and "shuffled_target_test" in r["checks"]
    assert "report_path" not in r


def test_saved_winner_can_predict_from_its_base_inputs(csv_path):
    r = auto.run_search("y", FEATS, ["extra"], data_path=csv_path, minutes=0.05)
    info = mb.get_model_info(r["saved_model_id"])
    row = {f: 2.0 for f in info["features"]}
    p = mb.predict(r["saved_model_id"], [row])["predictions"][0]
    assert np.isfinite(p["predicted_y"]) and p["uncertainty_std"] >= 0
    assert r["saved_model_id"] in {m["model_id"] for m in mb.list_models()["models"]}


def test_lab_target_defaults_use_inputs_known_before_growth():
    r = auto.run_search("RHEED_Quality_Film", data_path="all", minutes=0.03, model_types=["krr"], engineered=False, save_best=False)
    assert r["n_rows"] == 21 and r["inputs"]["features"][0] == "growthtime"
    assert "EDS_ratio" not in r["inputs"]["features"] + r["inputs"]["extra_features"]
    assert "saved_model_id" not in r


def test_post_growth_input_is_flagged():
    r = auto.run_search("RHEED_Quality_Film", ["growthtime", "EDS_ratio"], [], data_path="all", minutes=0.02,
                        model_types=["krr"], engineered=False, save_best=False)
    assert "EDS_ratio" in r["note_post_growth_inputs"]


def test_tool_wrapper_reaches_search(csv_path):
    out = model_agent.run_autonomous_search.invoke({"target": "y", "data_path": csv_path})
    assert out["needs_user_input"]


def test_report_written_only_when_requested(csv_path):
    quiet = auto.run_search("y", FEATS, data_path=csv_path, minutes=0.02, model_types=["krr"])
    assert "report_path" not in quiet and not list(mb.RESULTS_ROOT.glob("search_*.json"))
    loud = auto.run_search("y", FEATS, data_path=csv_path, minutes=0.02, model_types=["krr"], save_report=True)
    assert json.loads(open(loud["report_path"]).read())["target"] == "y"
    assert mb.RESULTS_ROOT in __import__("pathlib").Path(loud["report_path"]).parents
