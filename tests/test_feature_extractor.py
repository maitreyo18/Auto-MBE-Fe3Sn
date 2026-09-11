import numpy as np
import pytest

import feature_extractor as fe


def test_lorentzian_bg_peaks_at_center():
    x = np.array([0.0, 5.0, 10.0])
    y = fe.lorentzian_bg(x, amp=10, ctr=5, hwhm=2, m=0, c=0)
    assert y[1] == pytest.approx(10.0)
    assert y[1] > y[0]
    assert y[1] > y[2]


def test_double_lorentzian_bg_two_peaks():
    x = np.linspace(0, 20, 500)
    y = fe.double_lorentzian_bg(x, a1=10, c1=5, w1=1, a2=8, c2=15, w2=1, m=0, c=0)
    assert y[np.argmin(np.abs(x - 5))] > y[np.argmin(np.abs(x - 10))]
    assert y[np.argmin(np.abs(x - 15))] > y[np.argmin(np.abs(x - 10))]


def test_compute_substrate_quality_uses_compiled_data(monkeypatch, synthetic_csv):
    monkeypatch.setattr(fe, "COMPILED_DATA_PATH", synthetic_csv)
    # synthetic_csv doesn't have the sub* feature columns fe expects, so this
    # exercises the "missing columns -> fallback" branch.
    score = fe.compute_substrate_quality(40.0, 40.0, 40.0, 50.0, 0.05)
    assert score == 50.0


def test_compute_substrate_quality_falls_back_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(fe, "COMPILED_DATA_PATH", tmp_path / "does_not_exist.csv")
    score = fe.compute_substrate_quality(40.0, 40.0, 40.0, 50.0, 0.05)
    assert score == 50.0


def test_compute_substrate_quality_in_bounds_with_real_data():
    df_path = fe.COMPILED_DATA_PATH
    if not df_path.exists():
        pytest.skip("real train_compiled.csv not present")
    score = fe.compute_substrate_quality(45.0, 40.0, 40.0, 50.0, 0.04)
    assert 0.0 <= score <= 100.0


def test_extract_rheed_quality_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError):
        fe.extract_rheed_quality(tmp_path / "nope.png")
