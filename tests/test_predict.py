import numpy as np
import pytest

import predict


def test_load_scaler_has_expected_features():
    scaler = predict.load_scaler()
    assert list(scaler["features"]) == ["growthtime", "filamentpower", "flux_ratio", "Substrate_quality"]
    assert len(scaler["mean"]) == 4
    assert len(scaler["std"]) == 4


def test_scale_features_roundtrip():
    scaler = predict.load_scaler()
    X = np.array([scaler["mean"]])
    scaled = predict.scale_features(X, scaler)
    assert scaled == pytest.approx(np.zeros_like(scaled), abs=1e-6)


@pytest.mark.parametrize("target", ["EDS_ratio", "RHEED_Quality_Film"])
def test_predict_with_uncertainty_shapes_and_finite(target):
    scaler = predict.load_scaler()
    X = np.array([scaler["mean"], scaler["mean"] + scaler["std"]])
    mean, std = predict.predict_with_uncertainty(X, target, scaler)
    assert mean.shape == (2,)
    assert std.shape == (2,)
    assert np.all(np.isfinite(mean))
    assert np.all(std >= 0)


@pytest.mark.parametrize("target", ["EDS_ratio", "RHEED_Quality_Film"])
def test_predict_with_tree_uncertainty_shapes_and_finite(target):
    scaler = predict.load_scaler()
    X = np.array([scaler["mean"]])
    mean, std = predict.predict_with_tree_uncertainty(X, target, scaler)
    assert mean.shape == (1,)
    assert std.shape == (1,)
    assert np.isfinite(mean[0])


def test_predict_with_uncertainty_rejects_wrong_feature_count():
    scaler = predict.load_scaler()
    X = np.array([[1.0, 2.0, 3.0]])  # only 3 columns, scaler expects 4
    with pytest.raises(ValueError):
        predict.predict_with_uncertainty(X, "EDS_ratio", scaler)


def test_predict_with_uncertainty_rejects_nan():
    scaler = predict.load_scaler()
    X = np.array([[np.nan, 2.0, 3.0, 4.0]])
    with pytest.raises(ValueError):
        predict.predict_with_uncertainty(X, "EDS_ratio", scaler)


def test_predict_with_uncertainty_rejects_1d_input():
    scaler = predict.load_scaler()
    X = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        predict.predict_with_uncertainty(X, "EDS_ratio", scaler)


def test_missing_checkpoint_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "THIS_DIR", tmp_path)
    scaler = {"mean": np.zeros(4), "std": np.ones(4), "features": ["a", "b", "c", "d"]}
    with pytest.raises(FileNotFoundError):
        predict.predict_with_uncertainty(np.zeros((1, 4)), "EDS_ratio", scaler)
