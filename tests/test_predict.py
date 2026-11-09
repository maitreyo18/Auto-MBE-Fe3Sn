import numpy as np
import pandas as pd
import pytest

import lab_models as lm
import predict


@pytest.mark.parametrize("target", [lm.FILM_TARGET, lm.EDS_TARGET])
def test_checkpoint_is_trained_on_17_tested_on_4(target):
    bundle = predict.load_bundle(target)
    assert bundle["n_train"] == 17 and bundle["n_test"] == 4
    assert bundle["features"] == lm.FEATURES[target]
    assert bundle["metrics"]["test_rmse"] > 0


def test_model_types_and_inputs():
    assert predict.load_bundle(lm.FILM_TARGET)["model_type"] == "mlp" and len(lm.FILM_FEATURES) == 10
    assert predict.load_bundle(lm.EDS_TARGET)["model_type"] == "rfr" and len(lm.EDS_FEATURES) == 4


def test_only_the_two_checkpoints_remain():
    assert {p.name for p in predict.Path(lm.THIS_DIR).glob("*.joblib")} == {
        "mlp_RHEED_Quality_Film.joblib", "rf_EDS_ratio.joblib",
    }


@pytest.mark.parametrize("target", [lm.FILM_TARGET, lm.EDS_TARGET])
def test_stored_metrics_reproduce_on_the_test_rows(target):
    test = lm.load_test()
    mean, std = predict.predict_with_uncertainty(test[lm.FEATURES[target]].to_numpy(dtype=float), target)
    assert mean.shape == std.shape == (4,) and np.all(std >= 0)
    rmse = np.sqrt(np.mean((test[target].to_numpy() - mean) ** 2))
    assert rmse == pytest.approx(predict.load_bundle(target)["metrics"]["test_rmse"], abs=1e-3)


def test_train_and_test_files_do_not_overlap():
    train, test = lm.load_train(), lm.load_test()
    assert len(train) == 17 and len(test) == 4
    assert not set(train["File"]) & set(test["File"])
    assert test.loc[test["eds_imputed"], "run"].tolist() == [32, 35]


@pytest.mark.parametrize("bad, target", [
    (np.zeros((1, 4)), lm.FILM_TARGET),
    (np.zeros((1, 10)), lm.EDS_TARGET),
    (np.zeros(4), lm.EDS_TARGET),
    (np.full((1, 4), np.nan), lm.EDS_TARGET),
])
def test_rejects_bad_input(bad, target):
    with pytest.raises(ValueError):
        predict.predict_with_uncertainty(bad, target)


def test_unknown_target_rejected():
    with pytest.raises(ValueError):
        predict.load_bundle("not_a_target")


def test_missing_checkpoint_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict._load_checkpoint(tmp_path / "missing.joblib")


def test_train_file_is_unchanged_shape():
    assert isinstance(lm.load_train(), pd.DataFrame)
