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


def test_existing_image_reproduces_its_stored_substrate_quality():
    from pathlib import Path

    import pandas as pd

    image = Path(fe.__file__).parent / "substrate" / "train" / "10.3.png"
    row = fe.extract_rheed_quality(image)
    stored = pd.read_csv(fe.COMPILED_DATA_PATH).set_index("run").loc[10]
    assert fe.TEST_DATA_PATH.exists()
    assert row["Substrate_quality"] == pytest.approx(stored["Substrate_quality"], abs=0.01)
    assert {"subcfwhm", "sublfwhm", "subrfwhm", "submean_width", "subwidth_variance",
            "subcurve_variance"} <= set(row)


# ---- film / substrate identification --------------------------------------

@pytest.mark.parametrize("name,kind", [
    ("32.1.png", "film"),
    ("32.3.png", "substrate"),
    ("film/test/x.png", "film"),
    ("substrate/train/x.png", "substrate"),
])
def test_detect_image_kind(name, kind):
    assert fe.detect_image_kind(name) == kind


def test_detect_image_kind_ambiguous_raises():
    with pytest.raises(ValueError):
        fe.detect_image_kind("mystery.png")


def test_invalid_kind_rejected(tmp_path):
    with pytest.raises(ValueError):
        fe.extract_rheed_quality(tmp_path / "x.png", kind="bogus")


def test_film_images_all_extract_with_bounded_score():
    from pathlib import Path

    images = sorted((Path(fe.__file__).parent / "film").glob("*/*.png"))
    assert len(images) == 21
    for image in images:
        row = fe.extract_rheed_quality(image)
        assert row["image_kind"] == "film"
        assert 0 <= row["RHEED_Film_Score"] <= 100
        assert {"cfwhm", "lfwhm", "rfwhm", "mean_width", "width_variance",
                "curve_variance"} <= set(row)


def test_film_score_reproduces_stored_film_quality():
    from pathlib import Path

    import pandas as pd

    root = Path(fe.__file__).parent
    stored = pd.concat([pd.read_csv(fe.COMPILED_DATA_PATH), pd.read_csv(fe.TEST_DATA_PATH)]).set_index("run")
    xs, ys = [], []
    for image in sorted((root / "film").glob("*/*.png")):
        run = int(image.name.split(".")[0])
        xs.append(fe.extract_rheed_quality(image)["RHEED_Film_Score"])
        ys.append(stored.loc[run, "RHEED_Quality_Film"])
    # the stored labels come from this exact method
    assert np.max(np.abs(np.array(xs) - np.array(ys))) < 0.1


def test_substrate_kind_unchanged_and_forceable():
    from pathlib import Path

    image = Path(fe.__file__).parent / "substrate" / "train" / "10.3.png"
    auto = fe.extract_rheed_quality(image)
    forced = fe.extract_rheed_quality(image, kind="substrate")
    assert auto == forced and auto["image_kind"] == "substrate"
    assert "RHEED_Film_Score" not in auto


# ---- exact agreement with the reference repo's saved outputs ---------------
# Values copied from raghu0415/Fe3SnMLfolder fe3snrheedanalysis outputs:
#   film: rheed_analysis/v6, streak_widths/v7, vertical_profiles/v14
#   substrate: substrate/v13, substrate_streak_widths/v6, substrate_vertical_profiles/v7

def _img(rel):
    from pathlib import Path

    import cv2

    return cv2.resize(cv2.imread(str(Path(fe.__file__).parent / rel), 0), (1024, 1024))


def test_film_pipeline_matches_reference_outputs_for_run_10():
    img = _img("film/train/10.1.png")
    fwhm = fe.process_fwhm_film(img)
    assert (fwhm["fwhm_L"], fwhm["fwhm_C"], fwhm["fwhm_R"]) == (31.75, 40.27, 29.57)
    assert fe.process_vertical_streak_film(img)["curve_variance"] == pytest.approx(0.0305)


def test_film_width_matches_reference_for_run_32():
    w = fe.process_streak_width_film(_img("film/test/32.1.png"))
    assert w["mean_width"] == pytest.approx(37.09, abs=0.01)


def test_film_and_substrate_use_different_width_algorithms():
    # same streak, different published algorithms -> different numbers
    film = fe.process_streak_width_film(_img("film/test/33.1.png"))["mean_width"]
    sub = fe.process_streak_width(_img("film/test/33.1.png"))["mean_width"]
    assert film != sub
