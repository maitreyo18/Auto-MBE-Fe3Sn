"""
Reads one RHEED image -- substrate (*.3.png) or film (*.1.png) -- and writes a
one-row CSV.

Substrate image:
    subcfwhm, sublfwhm, subrfwhm, submean_width, subwidth_variance,
    subcurve_variance, Substrate_quality

Film image:
    cfwhm, lfwhm, rfwhm, mean_width, width_variance, curve_variance,
    RHEED_Film_Score   (film quality %, from the image alone)

The image type is identified from the file name (.1 = film, .3 = substrate),
else from the parent folder name ("film" / "substrate"), or forced with --kind.

Output: image_<run_number>.csv (substrate) or film_image_<run_number>.csv
written next to this script, e.g. 10.3.png -> image_10.csv.

Usage:
    python feature_extractor.py 10.3.png
    python feature_extractor.py film/test/32.1.png
    python feature_extractor.py /path/to/img.png --kind film
"""

import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler

CROP_BOTTOM = 950
SEARCH_RADIUS = 90

THIS_DIR = Path(__file__).resolve().parent

# All 21 growths (17 training + 4 test), the set the stored Substrate_quality
# values are scaled over. Used to normalize the new image's raw features.
COMPILED_DATA_PATH = THIS_DIR.parent / "data" / "train_compiled.csv"
TEST_DATA_PATH = THIS_DIR.parent / "data" / "test_compiled.csv"


def process_streak_width(img):
    h_profile = np.mean(img[450:574, :], axis=0)
    smoothed_h = savgol_filter(h_profile, 21, 3)

    peaks, _ = find_peaks(smoothed_h, prominence=10, distance=50)
    if len(peaks) == 0:
        return None

    center_area_peaks = [p for p in peaks if 350 < p < 674]
    if not center_area_peaks:
        center_area_peaks = peaks
    cx = max(center_area_peaks, key=lambda p: smoothed_h[p])

    center_line = np.mean(img[:, cx - 2 : cx + 3], axis=1)
    smoothed_center = savgol_filter(center_line, 11, 3)

    void_baseline = np.mean(smoothed_center[0:20])
    edge_pixels = np.where(smoothed_center > (void_baseline + 5))[0]

    start_y = edge_pixels[0] if len(edge_pixels) > 0 else 50
    start_y = min(max(start_y, 10), 400)

    raw_left_edges = []
    raw_right_edges = []

    for y in range(start_y, CROP_BOTTOM):
        row_slice = img[y, cx - SEARCH_RADIUS : cx + SEARCH_RADIUS].astype(float)
        smoothed_row = savgol_filter(row_slice, 31, 3)

        local_bg_left = np.min(smoothed_row[:15])
        local_bg_right = np.min(smoothed_row[-15:])
        local_baseline = np.linspace(local_bg_left, local_bg_right, len(smoothed_row))

        corrected_row = smoothed_row - local_baseline
        corrected_row = np.clip(corrected_row, 0, None)

        max_val = np.max(corrected_row)

        if max_val > 5:
            half_max = max_val / 2.0
            above_half = np.where(corrected_row >= half_max)[0]

            if len(above_half) > 0:
                left_idx = above_half[0]
                right_idx = above_half[-1]

                abs_left = (cx - SEARCH_RADIUS) + left_idx
                abs_right = (cx - SEARCH_RADIUS) + right_idx

                raw_left_edges.append(abs_left)
                raw_right_edges.append(abs_right)
            else:
                raw_left_edges.append(np.nan)
                raw_right_edges.append(np.nan)
        else:
            raw_left_edges.append(np.nan)
            raw_right_edges.append(np.nan)

    left_series = pd.Series(raw_left_edges).interpolate().bfill().ffill().values
    right_series = pd.Series(raw_right_edges).interpolate().bfill().ffill().values

    smooth_left = savgol_filter(left_series, 71, 3)
    smooth_right = savgol_filter(right_series, 71, 3)

    final_widths = smooth_right - smooth_left

    return {
        "cx": cx,
        "mean_width": round(np.mean(final_widths), 2),
        "width_variance": round(np.var(final_widths), 2),
    }


def process_vertical_streak(img):
    h_profile = np.mean(img[450:574, :], axis=0)
    smoothed_h = savgol_filter(h_profile, 21, 3)

    peaks, _ = find_peaks(smoothed_h, prominence=10, distance=50)
    if len(peaks) == 0:
        return None

    center_area_peaks = [p for p in peaks if 350 < p < 674]
    if not center_area_peaks:
        center_area_peaks = peaks
    cx = max(center_area_peaks, key=lambda p: smoothed_h[p])

    search_radius = 90
    left_bound = max(0, cx - search_radius)
    right_bound = min(1024, cx + search_radius)

    left_valley = np.argmin(smoothed_h[left_bound:cx]) + left_bound
    right_valley = np.argmin(smoothed_h[cx:right_bound]) + cx

    if (cx - left_valley) < 10 or (right_valley - cx) < 10:
        left_valley, right_valley = cx - 40, cx + 40

    box_left = int(left_valley)
    box_right = int(right_valley)

    streak_box = img[0:CROP_BOTTOM, box_left:box_right]
    v_profile = np.mean(streak_box, axis=1)
    smoothed_v = savgol_filter(v_profile, 11, 3)

    void_baseline = np.mean(smoothed_v[0:20])

    edge_pixels = np.where(smoothed_v > (void_baseline + 5))[0]
    start_y = edge_pixels[0] if len(edge_pixels) > 0 else 50
    start_y = min(max(start_y, 10), 400)

    cropped_streak = smoothed_v[start_y:CROP_BOTTOM]

    peak_val = np.max(cropped_streak) - void_baseline
    if peak_val == 0:
        peak_val = 1

    norm_profile = (cropped_streak - void_baseline) / peak_val

    curve_variance = round(np.var(norm_profile), 4) if len(norm_profile) > 0 else np.nan

    return {
        "cx": cx,
        "curve_variance": curve_variance,
    }


def lorentzian_bg(x, amp, ctr, hwhm, m, c):
    return (amp * hwhm**2 / ((x - ctr) ** 2 + hwhm**2)) + (m * x + c)


def double_lorentzian_bg(x, a1, c1, w1, a2, c2, w2, m, c):
    return (
        (a1 * w1**2 / ((x - c1) ** 2 + w1**2))
        + (a2 * w2**2 / ((x - c2) ** 2 + w2**2))
        + (m * x + c)
    )


def process_fwhm(img):
    """Adaptive hybrid singlet/doublet Lorentzian-plus-linear-background fit
    for the left/center/right diffraction peaks."""
    profile = np.mean(img[480:540, :], axis=0)
    smoothed = savgol_filter(profile, 11, 3)
    x = np.arange(1024)

    peaks, _ = find_peaks(smoothed, prominence=8, distance=100)
    if len(peaks) < 3:
        return None

    best_peaks = sorted(peaks[np.argsort(smoothed[peaks])[-3:]])
    center_idx_val = np.argmin(np.abs(np.array(best_peaks) - 512))

    res = {}

    for i, p_loc in enumerate(best_peaks):
        # 140px window for the wide central doublet, 80px for side singlets
        win = 140 if i == center_idx_val else 80

        mask = (x >= p_loc - win) & (x <= p_loc + win)
        x_loc = x[mask]

        y_loc_smooth = smoothed[mask]
        y_loc_raw = profile[mask]

        valid_mask = y_loc_smooth > 0.5
        if not np.any(valid_mask):
            continue
        x_val = x_loc[valid_mask]
        y_val_smooth = y_loc_smooth[valid_mask]
        y_val_raw = y_loc_raw[valid_mask]

        m_guess = (y_val_smooth[-1] - y_val_smooth[0]) / (x_val[-1] - x_val[0] + 1e-6)
        c_guess = np.min(y_val_smooth)

        peak_threshold = c_guess + (np.max(y_val_raw) - c_guess) * 0.80
        fit_weights = np.where(y_val_raw > peak_threshold, 0.2, 1.0)

        try:
            if i == center_idx_val:
                sub_peaks, _ = find_peaks(y_val_smooth, prominence=2, distance=5)
                is_doublet = False

                if len(sub_peaks) >= 2:
                    p1, p2 = sub_peaks[0], sub_peaks[-1]
                    valley_min = np.min(y_val_smooth[p1 : p2 + 1])
                    peak_max = max(y_val_smooth[p1], y_val_smooth[p2])
                    dip_ratio = (peak_max - valley_min) / (peak_max - c_guess + 1e-6)

                    if dip_ratio > 0.15:
                        is_doublet = True

                if is_doublet:
                    c1_g, c2_g = x_val[sub_peaks[0]], x_val[sub_peaks[-1]]

                    a1_g = y_val_raw[sub_peaks[0]] - c_guess
                    a2_g = y_val_raw[sub_peaks[-1]] - c_guess

                    p0 = [a1_g, c1_g, 15, a2_g, c2_g, 15, m_guess, c_guess]
                    bounds = (
                        [0, 0, 0.1, 0, 0, 0.1, -np.inf, -np.inf],
                        [np.inf, 1024, 80, np.inf, 1024, 80, np.inf, np.inf],
                    )

                    popt, _ = curve_fit(
                        double_lorentzian_bg,
                        x_val,
                        y_val_raw,
                        p0=p0,
                        bounds=bounds,
                        sigma=fit_weights,
                        absolute_sigma=False,
                        maxfev=5000,
                    )

                    dist = abs(popt[1] - popt[4])
                    avg_w = abs(popt[2]) + abs(popt[5])
                    res["fwhm_C"] = round(dist + avg_w, 2)

                else:
                    amp_g = np.max(y_val_raw) - c_guess
                    p0 = [amp_g, p_loc, 15, m_guess, c_guess]
                    bounds = (
                        [0, 0, 0.1, -np.inf, -np.inf],
                        [np.inf, 1024, 150, np.inf, np.inf],
                    )

                    popt, _ = curve_fit(
                        lorentzian_bg,
                        x_val,
                        y_val_raw,
                        p0=p0,
                        bounds=bounds,
                        sigma=fit_weights,
                        absolute_sigma=False,
                        maxfev=5000,
                    )
                    res["fwhm_C"] = round(abs(popt[2] * 2), 2)

            else:
                amp_g = np.max(y_val_raw) - c_guess
                p0 = [amp_g, p_loc, 10, m_guess, c_guess]
                bounds = (
                    [0, 0, 0.1, -np.inf, -np.inf],
                    [np.inf, 1024, 100, np.inf, np.inf],
                )

                popt, _ = curve_fit(
                    lorentzian_bg,
                    x_val,
                    y_val_raw,
                    p0=p0,
                    bounds=bounds,
                    sigma=fit_weights,
                    absolute_sigma=False,
                    maxfev=5000,
                )

                label = "L" if i < center_idx_val else "R"
                res[f"fwhm_{label}"] = round(abs(popt[2] * 2), 2)

        except RuntimeError:
            continue

    if "fwhm_L" not in res or "fwhm_C" not in res or "fwhm_R" not in res:
        return None

    return res


def compute_substrate_quality(subcfwhm, sublfwhm, subrfwhm, submean_width, subcurve_variance):
    """Normalizes the new image's raw features against the existing compiled
    dataset, then applies a 40/20/20/20 weighted penalty formula."""
    score_cols = ["subcfwhm", "sublfwhm", "subrfwhm", "submean_width", "subcurve_variance"]
    new_row = {
        "subcfwhm": subcfwhm,
        "sublfwhm": sublfwhm,
        "subrfwhm": subrfwhm,
        "submean_width": submean_width,
        "subcurve_variance": subcurve_variance,
    }

    if COMPILED_DATA_PATH.exists():
        df = pd.read_csv(COMPILED_DATA_PATH)
        if all(c in df.columns for c in score_cols):
            frames = [df[score_cols], pd.DataFrame([new_row])]
            if TEST_DATA_PATH.exists():
                frames.insert(1, pd.read_csv(TEST_DATA_PATH)[score_cols])
            combined = pd.concat(frames, ignore_index=True)
            scaler = MinMaxScaler()
            scaled = pd.DataFrame(
                scaler.fit_transform(combined[score_cols]), columns=score_cols
            )
            s = scaled.iloc[-1]
            sub_penalty = (
                ((s["sublfwhm"] + s["subrfwhm"]) / 2 * 0.40)
                + (s["submean_width"] * 0.20)
                + (s["subcfwhm"] * 0.20)
                + (s["subcurve_variance"] * 0.20)
            )
            return round((1.0 - sub_penalty) * 100, 2)

    return 50.0


KINDS = ("substrate", "film")
FILM_DIR = THIS_DIR / "film"
FILM_SCORE_COLS = ["cfwhm", "lfwhm", "rfwhm", "mean_width", "curve_variance"]


def detect_image_kind(image_path):
    """'film' or 'substrate', from the file name (<run>.1.png = film,
    <run>.3.png = substrate), else from the parent folder name. Raises
    ValueError if neither identifies it (then pass kind explicitly)."""
    p = Path(image_path)
    parts = p.stem.split(".")  # "32.1.png" -> stem "32.1" -> ["32", "1"]
    if len(parts) >= 2:
        if parts[-1] == "1":
            return "film"
        if parts[-1] == "3":
            return "substrate"
    names = {q.name.lower() for q in p.parents}
    if "film" in names and "substrate" not in names:
        return "film"
    if "substrate" in names and "film" not in names:
        return "substrate"
    raise ValueError(
        f"cannot tell whether {p.name} is a film or substrate image; "
        "name it <run>.1.png / <run>.3.png or pass kind='film'|'substrate'"
    )


def _lorentzian(x, amp, ctr, hwhm):
    return amp * hwhm**2 / ((x - ctr) ** 2 + hwhm**2)


def process_fwhm_film(img):
    """FWHM of the left/center/right streaks of a FILM image (reference repo's
    mbeanalyzer.py). Film streaks are dim and sit on a curved phosphor glow, so
    (unlike process_fwhm) this flattens the central-belt profile with a morphological background and uses
    a prominence relative to the flattened maximum, then fits one Lorentzian
    per streak in a +/-60 px window. Returns None if L, C and R are not found."""
    profile = np.mean(img[450:574, :], axis=0)
    x = np.arange(len(profile))
    smoothed = savgol_filter(profile, 15, 3)
    bg = uniform_filter1d(
        maximum_filter1d(minimum_filter1d(smoothed, size=150), size=150), size=150
    )
    flat = np.clip(smoothed - bg, 0, None)
    if np.max(flat) <= 0:
        return None

    peaks, _ = find_peaks(flat, prominence=np.max(flat) * 0.1, distance=40)
    if len(peaks) < 3:
        return None
    center_area = [p for p in peaks if 350 < p < 674] or list(peaks)
    c_pixel = max(center_area, key=lambda p: flat[p])
    c_idx = int(np.where(peaks == c_pixel)[0][0])
    if c_idx == 0 or c_idx == len(peaks) - 1:
        return None

    res = {}
    for label, loc in zip("LCR", (peaks[c_idx - 1], peaks[c_idx], peaks[c_idx + 1])):
        mask = (x >= loc - 60) & (x <= loc + 60)
        try:
            popt, _ = curve_fit(
                _lorentzian, x[mask], flat[mask], p0=[flat[loc], loc, 10]
            )
        except (RuntimeError, ValueError):
            return None
        res[f"fwhm_{label}"] = round(abs(popt[2] * 2), 2)
    return res


def _find_central_streak_x(img):
    """Column of the central streak: the brightest horizontal-profile peak in
    pixels 350-674 (else the brightest overall). None if no peak."""
    h_profile = np.mean(img[450:574, :], axis=0)
    smoothed_h = savgol_filter(h_profile, 21, 3)
    peaks, _ = find_peaks(smoothed_h, prominence=10, distance=50)
    if len(peaks) == 0:
        return None
    center_area_peaks = [p for p in peaks if 350 < p < 674] or list(peaks)
    return max(center_area_peaks, key=lambda p: smoothed_h[p])


def process_streak_width_film(img):
    """Film streak width (their widthprofiler.py): per row, subtract a sloping
    local baseline, take the peak closest to the streak axis and its half-height
    width (scipy peak_widths), then smooth the edge envelope."""
    cx = _find_central_streak_x(img)
    if cx is None:
        return None

    center_line = np.mean(img[:, cx - 2 : cx + 3], axis=1)
    smoothed_center = savgol_filter(center_line, 11, 3)
    void_baseline = np.mean(smoothed_center[0:20])
    edge_pixels = np.where(smoothed_center > (void_baseline + 5))[0]
    start_y = edge_pixels[0] if len(edge_pixels) > 0 else 50
    start_y = min(max(start_y, 10), 400)

    raw_left_edges, raw_right_edges = [], []
    for y in range(start_y, CROP_BOTTOM):
        row_slice = img[y, cx - SEARCH_RADIUS : cx + SEARCH_RADIUS].astype(float)
        smoothed_row = savgol_filter(row_slice, 15, 3)

        local_bg_left = np.min(smoothed_row[:15])
        local_bg_right = np.min(smoothed_row[-15:])
        local_baseline = np.linspace(local_bg_left, local_bg_right, len(smoothed_row))
        corrected_row = np.clip(smoothed_row - local_baseline, 0, None)

        row_peaks, _ = find_peaks(corrected_row, prominence=2)
        if len(row_peaks) > 0:
            closest_peak = row_peaks[np.argmin(np.abs(row_peaks - SEARCH_RADIUS))]
            _, _, left_ips, right_ips = peak_widths(
                corrected_row, [closest_peak], rel_height=0.5
            )
            raw_left_edges.append((cx - SEARCH_RADIUS) + left_ips[0])
            raw_right_edges.append((cx - SEARCH_RADIUS) + right_ips[0])
        else:
            raw_left_edges.append(np.nan)
            raw_right_edges.append(np.nan)

    left_series = pd.Series(raw_left_edges).interpolate().bfill().ffill().values
    right_series = pd.Series(raw_right_edges).interpolate().bfill().ffill().values
    final_widths = savgol_filter(right_series, 71, 3) - savgol_filter(left_series, 71, 3)

    return {
        "cx": cx,
        "mean_width": round(np.mean(final_widths), 2),
        "width_variance": round(np.var(final_widths), 2),
    }


def process_vertical_streak_film(img):
    """Film curve variance (their verticalprof2.py): variance of the normalized
    intensity along a 3-px-wide column through the central streak."""
    cx = _find_central_streak_x(img)
    if cx is None:
        return None

    v_profile = np.mean(img[0:CROP_BOTTOM, cx - 1 : cx + 2], axis=1)
    smoothed_v = savgol_filter(v_profile, 11, 3)
    void_baseline = np.mean(smoothed_v[0:20])
    edge_pixels = np.where(smoothed_v > (void_baseline + 5))[0]
    start_y = edge_pixels[0] if len(edge_pixels) > 0 else 50
    start_y = min(max(start_y, 10), 400)

    cropped_streak = smoothed_v[start_y:CROP_BOTTOM]
    peak_val = np.max(cropped_streak) - void_baseline
    if peak_val == 0:
        peak_val = 1
    norm_profile = (cropped_streak - void_baseline) / peak_val
    curve_variance = round(np.var(norm_profile), 4) if len(norm_profile) > 0 else np.nan
    return {"cx": cx, "curve_variance": curve_variance}


def _load_resized(img_path):
    if not img_path.exists():
        raise ValueError(f"could not find image at {img_path}")
    img_raw = cv2.imread(str(img_path), 0)
    if img_raw is None:
        raise ValueError(f"could not read image at {img_path}")
    return cv2.resize(img_raw, (1024, 1024))


def _film_features(img_path, img_name=None):
    img = _load_resized(img_path)
    name = img_name or img_path.name
    width_result = process_streak_width_film(img)
    vert_result = process_vertical_streak_film(img)
    fwhm_result = process_fwhm_film(img)
    if width_result is None or vert_result is None or fwhm_result is None:
        raise ValueError(f"could not locate a central streak in {name}")
    return {
        "cfwhm": fwhm_result["fwhm_C"],
        "lfwhm": fwhm_result["fwhm_L"],
        "rfwhm": fwhm_result["fwhm_R"],
        "mean_width": width_result["mean_width"],
        "width_variance": width_result["width_variance"],
        "curve_variance": vert_result["curve_variance"],
    }


@lru_cache(maxsize=1)
def _film_reference_features():
    """Raw film features of every film image shipped in RHEED_images/film
    (train + test); the set the film score is min-max scaled over."""
    rows = []
    for p in sorted(FILM_DIR.glob("*/*.png")):
        try:
            rows.append({k: float(v) for k, v in _film_features(p).items()})
        except ValueError:
            continue
    return pd.DataFrame(rows, columns=FILM_SCORE_COLS + ["width_variance"])


def compute_film_score(cfwhm, lfwhm, rfwhm, mean_width, curve_variance):
    """0-100 film-quality estimate: min-max scale the 5 features against the
    reference film images, then 1 - (0.40*mean(L,R FWHM) + 0.20*mean_width +
    0.20*center FWHM + 0.20*curve_variance). Same 40/20/20/20 weights as the
    substrate score. With the reference film algorithms this reproduces the
    stored RHEED_Quality_Film labels of all 21 growths to within 0.1."""
    new_row = {
        "cfwhm": cfwhm,
        "lfwhm": lfwhm,
        "rfwhm": rfwhm,
        "mean_width": mean_width,
        "curve_variance": curve_variance,
    }
    ref = _film_reference_features()
    if ref.empty:
        return 50.0
    combined = pd.concat([ref[FILM_SCORE_COLS], pd.DataFrame([new_row])], ignore_index=True)
    scaled = pd.DataFrame(
        MinMaxScaler().fit_transform(combined[FILM_SCORE_COLS]), columns=FILM_SCORE_COLS
    )
    s = scaled.iloc[-1]
    penalty = (
        (s["lfwhm"] + s["rfwhm"]) / 2 * 0.40
        + s["mean_width"] * 0.20
        + s["cfwhm"] * 0.20
        + s["curve_variance"] * 0.20
    )
    return round((1.0 - penalty) * 100, 2)


def extract_rheed_quality(image_path, kind="auto"):
    """Runs the full pipeline on one RHEED image and returns its feature row.

    kind: "auto" (identify from the name/folder), "substrate" or "film".
    Substrate rows include Substrate_quality; film rows include
    RHEED_Film_Score and an "image_kind" key. Raises ValueError if the image
    can't be processed or its kind can't be identified."""
    img_path = Path(image_path)
    if not img_path.is_absolute():
        img_path = THIS_DIR / img_path

    if kind == "auto":
        kind = detect_image_kind(img_path)
    elif kind not in KINDS:
        raise ValueError(f"kind must be 'auto', 'substrate' or 'film', got {kind!r}")

    if kind == "film":
        row = _film_features(img_path)
        row["RHEED_Film_Score"] = compute_film_score(
            row["cfwhm"], row["lfwhm"], row["rfwhm"],
            row["mean_width"], row["curve_variance"],
        )
        row["image_kind"] = "film"
        return row

    img = _load_resized(img_path)

    width_result = process_streak_width(img)
    vert_result = process_vertical_streak(img)
    fwhm_result = process_fwhm(img)

    if width_result is None or vert_result is None or fwhm_result is None:
        raise ValueError(f"could not locate a central streak in {img_path.name}")

    subcfwhm = fwhm_result["fwhm_C"]
    sublfwhm = fwhm_result["fwhm_L"]
    subrfwhm = fwhm_result["fwhm_R"]
    submean_width = width_result["mean_width"]
    subwidth_variance = width_result["width_variance"]
    subcurve_variance = vert_result["curve_variance"]

    substrate_quality = compute_substrate_quality(
        subcfwhm, sublfwhm, subrfwhm, submean_width, subcurve_variance
    )

    return {
        "subcfwhm": subcfwhm,
        "sublfwhm": sublfwhm,
        "subrfwhm": subrfwhm,
        "submean_width": submean_width,
        "subwidth_variance": subwidth_variance,
        "subcurve_variance": subcurve_variance,
        "Substrate_quality": substrate_quality,
        "image_kind": "substrate",
    }


def main():
    args = sys.argv[1:]
    kind = "auto"
    if "--kind" in args:
        i = args.index("--kind")
        kind = args[i + 1] if i + 1 < len(args) else ""
        del args[i : i + 2]
    img_arg = args[0] if args else "10.3.png"
    img_path = Path(img_arg)
    if not img_path.is_absolute():
        img_path = THIS_DIR / img_path

    try:
        row = extract_rheed_quality(img_path, kind=kind)
    except ValueError as e:
        print(f"Error: {e}")
        return

    run_number = img_path.name.split(".")[0]
    prefix = "film_image_" if row["image_kind"] == "film" else "image_"
    out_path = THIS_DIR / f"{prefix}{run_number}.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False)

    print(f"Processed {img_path.name} ({row['image_kind']})")
    for k, v in row.items():
        print(f"  {k} = {v}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
