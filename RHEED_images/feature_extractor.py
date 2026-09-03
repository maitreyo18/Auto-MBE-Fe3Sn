"""
Reads one substrate RHEED image and writes a one-row CSV of:

    subcfwhm, sublfwhm, subrfwhm, submean_width, subwidth_variance,
    subcurve_variance, Substrate_quality

Output: image_<run_number>.csv written next to this script, e.g.
    10.3.png -> image_10.csv

Usage:
    python feature_extractor.py 10.3.png
    python feature_extractor.py /path/to/10.3.png
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler

CROP_BOTTOM = 950
SEARCH_RADIUS = 90

THIS_DIR = Path(__file__).resolve().parent

# Existing compiled dataset, used to normalize the new image's raw features
# against the same scale as the rest of the training data.
COMPILED_DATA_PATH = THIS_DIR.parent / "data" / "train_compiled.csv"


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
            combined = pd.concat(
                [df[score_cols], pd.DataFrame([new_row])], ignore_index=True
            )
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


def extract_rheed_quality(image_path):
    """Runs the full pipeline on one RHEED image and returns the feature row
    (including Substrate_quality), or raises ValueError if the image can't
    be processed."""
    img_path = Path(image_path)
    if not img_path.is_absolute():
        img_path = THIS_DIR / img_path

    if not img_path.exists():
        raise ValueError(f"could not find image at {img_path}")

    img_raw = cv2.imread(str(img_path), 0)
    if img_raw is None:
        raise ValueError(f"could not read image at {img_path}")

    img = cv2.resize(img_raw, (1024, 1024))

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
    }


def main():
    img_arg = sys.argv[1] if len(sys.argv) > 1 else "10.3.png"
    img_path = Path(img_arg)
    if not img_path.is_absolute():
        img_path = THIS_DIR / img_path

    try:
        row = extract_rheed_quality(img_path)
    except ValueError as e:
        print(f"Error: {e}")
        return

    run_number = img_path.name.split(".")[0]
    out_path = THIS_DIR / f"image_{run_number}.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False)

    print(f"Processed {img_path.name}")
    for k, v in row.items():
        print(f"  {k} = {v}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
