import json
import types

import numpy as np
import pandas as pd
import pytest

import model_agent
import model_builder as mb

FEATS = ["growthtime", "filamentpower", "flux_ratio"]


@pytest.fixture(autouse=True)
def isolated_model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(mb, "RESULTS_ROOT", tmp_path / "model_builder_results")
    monkeypatch.setattr(mb, "MODEL_DIR", tmp_path / "model_builder_results" / "models")
    mb.session_log.clear()


@pytest.fixture
def csv_path(tmp_path):
    rng = np.random.default_rng(1)
    n = 40
    df = pd.DataFrame({
        "growthtime": rng.uniform(200, 700, n),
        "filamentpower": rng.uniform(3, 5.5, n),
        "flux_ratio": rng.uniform(0.4, 1.3, n),
    })
    df["y"] = 0.05 * df["growthtime"] + 8 * df["filamentpower"] - 20 * df["flux_ratio"] ** 2 + rng.normal(0, 0.5, n)
    df["label"] = ["a"] * n
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_missing_target_or_features_asks_user_and_trains_nothing(csv_path):
    out = mb.train_models(targets=None, features=FEATS, data_path=csv_path)
    assert out["needs_user_input"] and out["missing"] == ["targets"]
    assert "y" in out["available_numeric_columns"] and "label" not in out["available_numeric_columns"]
    out = mb.train_models(targets=["y"], features=[], data_path=csv_path)
    assert out["missing"] == ["features"]
    assert [m for m in mb.list_models()["models"] if not m["built_in"]] == []


def test_train_both_models_saves_and_reports_rmse(csv_path):
    out = mb.train_models(["y"], FEATS, data_path=csv_path)
    assert {t["model_type"] for t in out["trained"]} == {"rfr", "gpr", "krr", "mlp"}
    for t in out["trained"]:
        assert t["metrics"]["train_rmse"] >= 0 and t["metrics"]["test_rmse"] >= 0
        assert (mb.MODEL_DIR / t["model_id"] / "model.joblib").exists()
        meta = json.loads((mb.MODEL_DIR / t["model_id"] / "meta.json").read_text())
        assert meta["features"] == FEATS and meta["n_train"] + meta["n_test"] == 40
    gpr = next(t for t in out["trained"] if t["model_type"] == "gpr")
    rfr = next(t for t in out["trained"] if t["model_type"] == "rfr")
    # smooth synthetic data: GP should beat a small RF on held-out error
    assert gpr["metrics"]["test_rmse"] < rfr["metrics"]["test_rmse"]


def test_multiple_targets_and_variants_then_compare(csv_path):
    df = pd.read_csv(csv_path)
    df["z"] = df["y"] * 2
    df.to_csv(csv_path, index=False)
    mb.train_models(["y", "z"], FEATS, data_path=csv_path)
    mb.train_models(["y"], FEATS, model_types=["rfr"], max_depth=2, name_prefix="shallow", data_path=csv_path)
    comp = mb.compare_models()
    custom = [r for r in comp["comparison"] if not r["model_id"].startswith("lab_")]
    assert len(custom) == 9 and len(comp["comparison"]) == 11  # + 2 built-in lab models
    y_rows = [r for r in comp["comparison"] if r["target"] == "y"]
    assert [r["rank_by_test_rmse"] for r in y_rows] == [1, 2, 3, 4, 5]
    assert [r["test_rmse"] for r in y_rows] == sorted(r["test_rmse"] for r in y_rows)
    assert any(r["model_id"].startswith("shallow_rfr_y_") for r in y_rows)
    assert len(mb.compare_models(target="z")["comparison"]) == 4


@pytest.mark.parametrize("kwargs", [
    {"model_types": ["xgboost"]},
    {"targets": ["nope"]},
    {"targets": ["label"]},
    {"features": ["y"]},
    {"kernel": "linear"},
    {"krr_kernel": "poly"},
    {"alpha": -1.0},
    {"gamma": 0.0},
    {"test_size": 1.5},
    {"name_prefix": "../evil"},
])
def test_train_rejects_bad_inputs(csv_path, kwargs):
    args = {"targets": ["y"], "features": FEATS, "data_path": csv_path}
    args.update(kwargs)
    with pytest.raises(ValueError):
        mb.train_models(**args)


def test_predict_uses_saved_model_and_flags_extrapolation(csv_path):
    out = mb.train_models(["y"], FEATS, model_types=["gpr"], data_path=csv_path)
    mid = out["trained"][0]["model_id"]
    res = mb.predict(mid, [{"growthtime": 400, "filamentpower": 4.0, "flux_ratio": 0.8},
                           {"growthtime": 5000, "filamentpower": 4.0, "flux_ratio": 0.8}])
    p = res["predictions"]
    expected = 0.05 * 400 + 8 * 4.0 - 20 * 0.8**2
    assert p[0]["predicted_y"] == pytest.approx(expected, abs=2.5)
    assert "extrapolation_warning" not in p[0] and "growthtime" in p[1]["extrapolation_warning"]
    assert p[1]["uncertainty_std"] > p[0]["uncertainty_std"]
    with pytest.raises(ValueError):
        mb.predict(mid, [{"growthtime": 400}])
    with pytest.raises(FileNotFoundError):
        mb.predict("rfr_missing", [{}])
    with pytest.raises(ValueError):
        mb.predict("../../etc", [{}])


def test_optimize_respects_bounds_fixed_and_goal(csv_path):
    mid = mb.train_models(["y"], FEATS, model_types=["gpr"], data_path=csv_path)["trained"][0]["model_id"]
    hi = mb.optimize(mid, "maximize", fixed={"growthtime": 600}, bounds={"filamentpower": [4.0, 5.0]}, top_k=3)
    assert len(hi["best"]) == 3
    assert all(b["growthtime"] == 600 and 4.0 <= b["filamentpower"] <= 5.0 for b in hi["best"])
    lo = mb.optimize(mid, "minimize", top_k=1)
    assert lo["best"][0]["predicted_y"] < hi["best"][0]["predicted_y"]
    tgt = mb.optimize(mid, "target", target_value=30.0, top_k=1)
    assert abs(tgt["best"][0]["predicted_y"] - 30.0) < 2.0
    with pytest.raises(ValueError):
        mb.optimize(mid, "target")
    with pytest.raises(ValueError):
        mb.optimize(mid, "maximize", bounds={"bogus": [0, 1]})


def test_refit_on_all_data_and_get_info(csv_path):
    out = mb.train_models(["y"], FEATS, data_path=csv_path, refit_on_all_data=True)
    for t in out["trained"]:
        info = mb.get_model_info(t["model_id"])
        assert info["fitted_on"] == "all_rows"
        assert ("feature_importances" in info) == (info["model_type"] == "rfr")
        assert ("fitted_kernel" in info) == (info["model_type"] == "gpr")


def test_describe_data_default_csv():
    out = mb.describe_data()
    assert out["n_rows"] == 17 and "EDS_ratio" in out["numeric_columns"]
    assert "File" in out["non_numeric_columns_unusable"]


# ------------------------------------------------------------- agent glue ----

def test_skills_file_loaded_and_scoped_to_rfr_gpr():
    prompt = model_agent.build_system_prompt()
    assert "---" not in prompt.split("Follow the skills below exactly.")[1][:10]  # frontmatter stripped
    assert "RandomForestRegressor" in prompt and "GaussianProcessRegressor" in prompt and "KernelRidge" in prompt
    assert "MLPRegressor" in prompt and "run_autonomous_search" in prompt
    assert "ONLY RFR, GPR, KRR and MLP" in model_agent.SKILLS_PATH.read_text()


def test_model_tools_registered():
    names = {t.name for t in model_agent.MODEL_TOOLS}
    assert names == {"describe_training_data", "train_models", "list_models", "compare_models",
                     "get_model_info", "predict_with_model", "optimize_with_model", "run_autonomous_search",
                     "save_results"}


def test_tool_wrapper_reaches_builder(csv_path):
    out = model_agent.train_models.invoke({"features": FEATS, "data_path": csv_path})
    assert out["needs_user_input"]


def test_ask_keeps_history_across_turns():
    model_agent.reset_history()
    seen = []

    class FakeAgent:
        def invoke(self, state):
            seen.append(len(state["messages"]))
            return {"messages": [types.SimpleNamespace(type="ai", content=f"reply{len(seen)}")]}

    fake = FakeAgent()
    assert model_agent.ask("train a model", agent=fake) == "reply1"
    assert model_agent.ask("target is y", agent=fake) == "reply2"
    assert seen == [1, 3]
    model_agent.reset_history()
    assert model_agent._memory.turns == []


def test_ask_model_builder_tool_registered_on_main_agent():
    import tools
    assert "ask_model_builder" in {t.name for t in tools.TOOLS}


# ----------------------------------------------------- built-in lab models ----

FOUR = {"growthtime": 400.0, "filamentpower": 4.0, "flux_ratio": 0.8, "Substrate_quality": 70.0}
TEN = {**FOUR, "subcfwhm": 45.0, "sublfwhm": 40.0, "subrfwhm": 40.0, "submean_width": 50.0,
       "subwidth_variance": 300.0, "subcurve_variance": 0.04}


def test_lab_models_listed_and_predict_from_their_checkpoints():
    from predict import predict_eds_ratio, predict_film_quality

    listed = {m["model_id"]: m for m in mb.list_models()["models"]}
    assert listed["lab_EDS_ratio"]["model_type"] == "rfr" and listed["lab_RHEED_Quality_Film"]["model_type"] == "mlp"

    res = mb.predict("lab_EDS_ratio", [FOUR])["predictions"][0]
    mean, std = predict_eds_ratio([list(FOUR.values())])
    assert res["predicted_EDS_ratio"] == pytest.approx(round(float(mean[0]), 4))
    assert res["uncertainty_std"] == pytest.approx(round(float(std[0]), 4))

    res = mb.predict("lab_RHEED_Quality_Film", [TEN])["predictions"][0]
    mean, std = predict_film_quality([list(TEN.values())])
    assert res["predicted_RHEED_Quality_Film"] == pytest.approx(round(float(mean[0]), 4))
    assert res["uncertainty_std"] == pytest.approx(round(float(std[0]), 4))


def test_film_model_requires_the_raw_substrate_inputs():
    with pytest.raises(ValueError, match="subcfwhm"):
        mb.predict("lab_RHEED_Quality_Film", [FOUR])


def test_film_model_info_reports_mlp_and_17_4_split():
    info = mb.get_model_info("lab_RHEED_Quality_Film")
    assert info["built_in"] and len(info["features"]) == 10 and info["hyperparameters"]["batch_size"] == 4
    assert info["n_train"] == 17 and info["n_test"] == 4
    assert info["metrics"]["test_rmse"] > 0
    assert not [k for k in info["metrics"] if k.startswith("mean_baseline")]


def test_lab_model_requires_all_four_inputs_and_finite_values():
    with pytest.raises(ValueError):
        mb.predict("lab_EDS_ratio", [{"growthtime": 400.0, "filamentpower": 4.0, "flux_ratio": 0.8}])
    with pytest.raises(ValueError):
        mb.predict("lab_EDS_ratio", [{"growthtime": float("nan"), "filamentpower": 4.0,
                                      "flux_ratio": 0.8, "Substrate_quality": 70.0}])
    with pytest.raises(FileNotFoundError):
        mb.predict("lab_nonexistent", [{}])


def test_lab_models_in_compare_with_their_evaluation(csv_path):
    comp = mb.compare_models(target="EDS_ratio")["comparison"]
    assert [r["model_id"] for r in comp] == ["lab_EDS_ratio"]
    assert comp[0]["evaluation"] == mb.LAB_EVALUATION and comp[0]["test_rmse"] > 0
    df = pd.read_csv(csv_path).rename(columns={"y": "EDS_ratio"})
    df.to_csv(csv_path, index=False)
    mb.train_models(["EDS_ratio"], FEATS, model_types=["gpr"], data_path=csv_path)
    out = mb.compare_models(target="EDS_ratio")
    assert len(out["comparison"]) == 2 and any("not strictly comparable" in n for n in out["notes"])


def test_lab_model_optimize_and_info():
    res = mb.optimize("lab_RHEED_Quality_Film", "maximize", top_k=2, n_candidates=2000,
                      fixed={"Substrate_quality": 75.0})
    assert all("subcfwhm" in b for b in res["best"])
    assert len(res["best"]) == 2 and all(b["Substrate_quality"] == 75.0 for b in res["best"])
    info = mb.get_model_info("lab_EDS_ratio")
    assert info["built_in"] and info["model_type"] == "rfr" and info["features"][0] == "growthtime"
    assert info["hyperparameters"]["max_depth"] == 3 and info["metrics"]["test_rmse"] > 0


# ------------------------------------------------------------------- KRR ----

def test_krr_trains_tunes_on_train_split_only_and_predicts(csv_path):
    out = mb.train_models(["y"], FEATS, model_types=["krr"], data_path=csv_path)
    t = out["trained"][0]
    assert t["model_type"] == "krr" and t["metrics"]["test_rmse"] < 3.0  # smooth data -> KRR should fit well
    info = mb.get_model_info(t["model_id"])
    hp = info["hyperparameters"]
    assert hp["tuned_by_inner_cv"] and hp["inner_cv_rmse"] > 0 and hp["alpha"] > 0 and hp["gamma"] > 0
    res = mb.predict(t["model_id"], [{"growthtime": 400, "filamentpower": 4.0, "flux_ratio": 0.8}])
    p = res["predictions"][0]
    assert p["predicted_y"] == pytest.approx(0.05 * 400 + 8 * 4.0 - 20 * 0.8**2, abs=3.0)
    assert p["uncertainty_std"] >= 0


def test_krr_fixed_hyperparameters_are_not_tuned(csv_path):
    out = mb.train_models(["y"], FEATS, model_types=["krr"], data_path=csv_path, alpha=0.1, gamma=0.1)
    hp = mb.get_model_info(out["trained"][0]["model_id"])["hyperparameters"]
    assert (hp["alpha"], hp["gamma"]) == (0.1, 0.1) and hp["tuned_by_inner_cv"] is False


def test_krr_linear_kernel_and_refit_on_all_keeps_chosen_params(csv_path):
    lin = mb.train_models(["y"], FEATS, model_types=["krr"], krr_kernel="linear", data_path=csv_path)
    assert mb.get_model_info(lin["trained"][0]["model_id"])["hyperparameters"]["gamma"] is None
    out = mb.train_models(["y"], FEATS, model_types=["krr"], data_path=csv_path, refit_on_all_data=True)
    info = mb.get_model_info(out["trained"][0]["model_id"])
    assert info["fitted_on"] == "all_rows" and info["hyperparameters"]["alpha"] > 0


def test_krr_optimize_works(csv_path):
    mid = mb.train_models(["y"], FEATS, model_types=["krr"], data_path=csv_path)["trained"][0]["model_id"]
    best = mb.optimize(mid, "maximize", top_k=1, n_candidates=2000)["best"][0]
    low = mb.optimize(mid, "minimize", top_k=1, n_candidates=2000)["best"][0]
    assert best["predicted_y"] > low["predicted_y"]


# ------------------------------------------------------------------- MLP ----

def test_mlp_trains_saves_predicts_and_records_settings(csv_path):
    out = mb.train_models(["y"], FEATS, model_types=["mlp"], data_path=csv_path,
                          hidden_nodes=2, mlp_alpha=0.5, batch_size=8)
    t = out["trained"][0]
    assert t["model_type"] == "mlp"
    info = mb.get_model_info(t["model_id"])
    assert info["hyperparameters"]["hidden_nodes"] == 2 and info["hyperparameters"]["batch_size"] == 8
    p = mb.predict(t["model_id"], [{"growthtime": 400, "filamentpower": 4.0, "flux_ratio": 0.8}])["predictions"][0]
    assert p["uncertainty_std"] > 0


def test_train_models_on_all_21_growths():
    out = mb.train_models(["RHEED_Quality_Film"], ["growthtime", "flux_ratio"], model_types=["krr"], data_path="all")
    assert out["trained"][0]["n_train"] + out["trained"][0]["n_test"] == 21


@pytest.mark.parametrize("kwargs", [{"hidden_nodes": 0}, {"batch_size": -1}, {"learning_rate": 0.0}, {"mlp_alpha": -1.0}])
def test_mlp_rejects_bad_settings(csv_path, kwargs):
    with pytest.raises(ValueError):
        mb.train_models(["y"], FEATS, model_types=["mlp"], data_path=csv_path, **kwargs)


# ---------------------------------------------------------------- saving ----

def test_default_location_is_model_builder_results():
    import importlib
    fresh = importlib.reload(mb)
    assert fresh.RESULTS_ROOT.name == "model_builder_results" and fresh.RESULTS_ROOT.parent == fresh.REPO_DIR
    assert fresh.MODEL_DIR.parent == fresh.RESULTS_ROOT


def test_only_models_are_written_without_being_asked(csv_path):
    out = mb.train_models(["y"], FEATS, model_types=["krr"], data_path=csv_path)
    mb.log_result("train_models", {}, out)
    mb.log_result("compare_models", {}, mb.compare_models())
    assert (mb.MODEL_DIR / out["trained"][0]["model_id"] / "model.joblib").exists()
    assert [p.name for p in mb.RESULTS_ROOT.iterdir()] == ["models"]


def test_save_session_results_writes_everything_from_tool_outputs(csv_path):
    out = mb.train_models(["y"], FEATS, model_types=["krr", "rfr"], data_path=csv_path)
    mb.log_result("train_models", {"targets": ["y"]}, out)
    mid = out["trained"][0]["model_id"]
    mb.log_result("predict_with_model", {}, mb.predict(mid, [{"growthtime": 400, "filamentpower": 4.0, "flux_ratio": 0.8}]))
    res = mb.save_session_results(name="run1", notes="looks fine")
    folder = mb.RESULTS_ROOT / "run1"
    assert res["saved_to"] == str(folder) and set(res["files"]) == {"results.json", "comparison.csv", "summary.md"}
    logged = json.loads((folder / "results.json").read_text())
    assert [e["tool"] for e in logged] == ["train_models", "predict_with_model"]
    table = pd.read_csv(folder / "comparison.csv")
    assert set(table["model_id"]) == {t["model_id"] for t in out["trained"]}
    assert table["test_rmse"].tolist() == sorted(t["metrics"]["test_rmse"] for t in out["trained"])
    summary = (folder / "summary.md").read_text()
    assert mid in summary and "## Predictions" in summary
    assert "not computed" in summary and "looks fine" in summary


def test_save_never_overwrites_and_rejects_bad_names(csv_path):
    mb.log_result("describe_training_data", {}, mb.describe_data(csv_path))
    first = mb.save_session_results(name="same")["saved_to"]
    second = mb.save_session_results(name="same")["saved_to"]
    assert first != second and second.endswith("same-2")
    with pytest.raises(ValueError):
        mb.save_session_results(name="../escape")
    with pytest.raises(ValueError):
        mb.save_session_results(name="models")


def test_save_with_nothing_to_save_is_a_clear_error():
    with pytest.raises(ValueError, match="nothing to save"):
        mb.save_session_results()


def test_tools_log_results_and_clear_resets(csv_path):
    model_agent.describe_training_data.invoke({"data_path": csv_path})
    model_agent.train_models.invoke({"targets": ["y"], "features": FEATS, "model_types": ["krr"], "data_path": csv_path})
    assert [e["tool"] for e in mb.session_log] == ["describe_training_data", "train_models"]
    res = model_agent.save_results.invoke({"name": "via_tool"})
    assert (mb.RESULTS_ROOT / "via_tool" / "results.json").exists() and res["n_tool_results"] == 2
    model_agent.reset_history()
    assert mb.session_log == []


def test_prompt_forbids_saving_unless_asked():
    prompt = model_agent.build_system_prompt()
    assert "only when the user explicitly asks" in prompt
