"""
Reads one substrate RHEED image and plots its transversal (W) and
longitudinal (L) intensity profiles.

Usage:
    python images_process.py 10.3.png
    python images_process.py /path/to/10.3.png
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

CROP_BOTTOM = 950
SEARCH_RADIUS = 90

THIS_DIR = Path(__file__).resolve().parent


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
    y_pixels = np.arange(start_y, CROP_BOTTOM)

    return {
        "cx": cx,
        "start_y": start_y,
        "y_pixels": y_pixels,
        "final_widths": final_widths,
        "x_pixels": np.arange(len(smoothed_h)),
        "transversal_intensity": smoothed_h,
        "transversal_peaks": peaks,
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
    y_pixels = np.arange(start_y, CROP_BOTTOM)

    curve_variance = round(np.var(norm_profile), 4) if len(norm_profile) > 0 else np.nan

    return {
        "cx": cx,
        "box_left": box_left,
        "box_right": box_right,
        "start_y": start_y,
        "y_pixels": y_pixels,
        "norm_profile": norm_profile,
        "curve_variance": curve_variance,
    }


def save_transversal_plot(width_result, out_path, img_name):
    x_pixels = width_result["x_pixels"]
    intensity = width_result["transversal_intensity"]
    peaks = width_result["transversal_peaks"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_pixels, intensity, color="purple", lw=2, label="Intensity (W)")
    ax.plot(x_pixels[peaks], intensity[peaks], "rx", ms=8, label="Detected Peaks")
    ax.set_title(f"(d) Transversal Cut (W): {img_name}")
    ax.set_xlabel("Horizontal Pixel Position")
    ax.set_ylabel("Intensity")
    ax.set_xlim(0, 1024)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def save_longitudinal_plot(vert_result, out_path, img_name):
    y_pixels = vert_result["y_pixels"]
    norm_profile = vert_result["norm_profile"]
    curve_variance = vert_result["curve_variance"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        y_pixels,
        norm_profile,
        color="blue",
        lw=2,
        label=r"Intensity (L), $\sigma^2$=" + f"{curve_variance}",
    )
    ax.set_title(f"(e) Longitudinal Cut (L): {img_name}")
    ax.set_xlabel("Vertical Pixel Position (Top -> Bottom)")
    ax.set_ylabel("Intensity")
    ax.set_xlim(0, 1024)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    img_arg = sys.argv[1] if len(sys.argv) > 1 else "10.3.png"
    img_path = Path(img_arg)
    if not img_path.is_absolute():
        img_path = THIS_DIR / img_path

    if not img_path.exists():
        print(f"Error: could not find image at {img_path}")
        return

    img_raw = cv2.imread(str(img_path), 0)
    if img_raw is None:
        print(f"Error: could not read image at {img_path}")
        return

    img = cv2.resize(img_raw, (1024, 1024))

    width_result = process_streak_width(img)
    vert_result = process_vertical_streak(img)

    if width_result is None or vert_result is None:
        print(f"Error: could not locate a central streak in {img_path.name}")
        return

    stem = img_path.stem

    out_d = THIS_DIR / f"fig2d_transversal_{stem}.png"
    out_e = THIS_DIR / f"fig2e_longitudinal_{stem}.png"

    save_transversal_plot(width_result, out_d, img_path.name)
    save_longitudinal_plot(vert_result, out_e, img_path.name)

    print(f"Processed {img_path.name}")
    print(f"  mean_width     = {width_result['mean_width']}")
    print(f"  width_variance = {width_result['width_variance']}")
    print(f"  curve_variance = {vert_result['curve_variance']}")
    print(f"Saved: {out_d}")
    print(f"Saved: {out_e}")


if __name__ == "__main__":
    main()
