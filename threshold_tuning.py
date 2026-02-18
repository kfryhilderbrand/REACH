from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def main():
    #plot_magnitude_with_thresholds(
    plot_magnitudes_and_suggest_thresholds(
    "Processed_CSVs/Participant 1/20251017-091403_Free_Form/"
    "20251017-091403_Free_Form_sensor-21263_preprocessed.csv"
    )







def robust_noise_floor(x: np.ndarray) -> float:
    """
    Robust estimate of noise floor using median absolute deviation (MAD).
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return med + 1.4826 * mad  # MAD -> std-equivalent


def plot_magnitude_with_thresholds(csv_relative_path: str, show_thresholds: bool = True):
    """
    Load a processed CSV and:
      - compute accel & gyro magnitudes
      - print suggested thresholds
      - plot magnitudes (optionally with threshold lines)
    """
    csv_path = Path(csv_relative_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    accel_cols = ["accel_0_lin", "accel_1_lin", "accel_2_lin"]
    gyro_cols  = ["gyro_0_corr", "gyro_1_corr", "gyro_2_corr"]

    for c in accel_cols + gyro_cols + ["time"]:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}'")

    t = df["time"].to_numpy()

    accel_mag = np.linalg.norm(df[accel_cols].to_numpy(), axis=1)
    gyro_mag  = np.linalg.norm(df[gyro_cols].to_numpy(), axis=1)

    # ---- Estimate noise floors ----
    accel_noise = robust_noise_floor(accel_mag)
    gyro_noise  = robust_noise_floor(gyro_mag)

    sets = suggest_threshold_sets(
        accel_mag, gyro_mag,
        rest_quantile=0.10,
        accel_sigma_floor=0.05,
        gyro_sigma_floor=0.05
    )
    print_threshold_sets(sets)

    # # ---- Suggested thresholds ----
    # accel_end_thresh   = accel_noise * 1.2
    # accel_start_thresh = accel_noise * 2.5

    # gyro_end_thresh    = gyro_noise * 1.2
    # gyro_start_thresh  = gyro_noise * 2.5

    # # ---- Print suggestions ----
    # print("\n=== Suggested Reach Thresholds ===")
    # print("Acceleration (linear magnitude):")
    # print(f"  Noise floor        ≈ {accel_noise:.4f}")
    # print(f"  accel_end_thresh   ≈ {accel_end_thresh:.4f}")
    # print(f"  accel_start_thresh ≈ {accel_start_thresh:.4f}")

    # print("\nGyroscope (corrected magnitude):")
    # print(f"  Noise floor        ≈ {gyro_noise:.4f}")
    # print(f"  gyro_end_thresh    ≈ {gyro_end_thresh:.4f}")
    # print(f"  gyro_start_thresh  ≈ {gyro_start_thresh:.4f}")
    # print("=================================\n")

    # ---- Plot acceleration magnitude ----
    plt.figure()
    plt.plot(t, accel_mag, label="accel |lin|")
    # if show_thresholds:
    #     plt.axhline(accel_start_thresh, linestyle="--", label="start thresh")
    #     plt.axhline(accel_end_thresh, linestyle=":", label="end thresh")
    plt.xlabel("time [s]")
    plt.ylabel("accel magnitude")
    plt.title("Acceleration magnitude (gravity-compensated)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---- Plot gyroscope magnitude ----
    plt.figure()
    plt.plot(t, gyro_mag, label="gyro |corr|")
    # if show_thresholds:
    #     plt.axhline(gyro_start_thresh, linestyle="--", label="start thresh")
    #     plt.axhline(gyro_end_thresh, linestyle=":", label="end thresh")
    plt.xlabel("time [s]")
    plt.ylabel("gyro magnitude")
    plt.title("Gyroscope magnitude (drift-corrected)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def mad_sigma(x: np.ndarray) -> float:
    """Robust std estimate from MAD."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def suggest_thresholds_from_rest_tail(
    accel_mag: np.ndarray,
    gyro_mag: np.ndarray,
    rest_quantile: float = 0.10,   # use lowest 10% as "rest-like"
    end_k: float = 3.0,            # end threshold = baseline + end_k*sigma
    start_k: float = 8.0,          # start threshold = baseline + start_k*sigma
    require_both: bool = True,     # rest mask uses BOTH accel+gyro low
):
    """
    Returns suggested (accel_end, accel_start, gyro_end, gyro_start) based on
    low-quantile "rest-like" samples.
    """
    a_q = np.quantile(accel_mag, rest_quantile)
    g_q = np.quantile(gyro_mag, rest_quantile)

    if require_both:
        rest_mask = (accel_mag <= a_q) & (gyro_mag <= g_q)
        # If too few samples, fall back to accel-only
        if rest_mask.sum() < max(50, 0.01 * len(accel_mag)):
            rest_mask = (accel_mag <= a_q)
    else:
        rest_mask = (accel_mag <= a_q)

    a_rest = accel_mag[rest_mask]
    g_rest = gyro_mag[rest_mask]

    a_base = float(np.median(a_rest))
    g_base = float(np.median(g_rest))

    a_sig = float(mad_sigma(a_rest))
    g_sig = float(mad_sigma(g_rest))

    # Avoid degenerate zero sigma
    a_sig = max(a_sig, 1e-6)
    g_sig = max(g_sig, 1e-6)

    accel_end = a_base + end_k * a_sig
    accel_start = a_base + start_k * a_sig

    gyro_end = g_base + end_k * g_sig
    gyro_start = g_base + start_k * g_sig

    return accel_end, accel_start, gyro_end, gyro_start, a_base, a_sig, g_base, g_sig


def plot_magnitudes_and_suggest_thresholds(csv_relative_path: str):
    csv_path = Path(csv_relative_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    accel_cols = ["accel_0_lin", "accel_1_lin", "accel_2_lin"]
    gyro_cols  = ["gyro_0_corr", "gyro_1_corr", "gyro_2_corr"]

    for c in ["time"] + accel_cols + gyro_cols:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}'")

    t = df["time"].to_numpy()
    accel_mag = np.linalg.norm(df[accel_cols].to_numpy(), axis=1)
    gyro_mag  = np.linalg.norm(df[gyro_cols].to_numpy(), axis=1)

    sets = suggest_threshold_sets(
        accel_mag, gyro_mag,
        rest_quantile=0.15,
        accel_sigma_floor=0.25,
        gyro_sigma_floor=0.25
    )
    print_threshold_sets(sets)

    # # --- Threshold suggestions based on rest tail ---
    # accel_end, accel_start, gyro_end, gyro_start, a_base, a_sig, g_base, g_sig = (
    #     suggest_thresholds_from_rest_tail(
    #         accel_mag, gyro_mag,
    #         rest_quantile=0.10,   # try 0.05–0.15
    #         end_k=3.0,
    #         start_k=8.0,
    #         require_both=True
    #     )
    # )

    # print("\n=== Threshold suggestions (rest-tail based) ===")
    # print(f"Accel baseline ~ {a_base:.4f}, sigma ~ {a_sig:.4f}")
    # print(f"  accel_end_thresh   ~ {accel_end:.4f}")
    # print(f"  accel_start_thresh ~ {accel_start:.4f}")
    # print(f"Gyro baseline  ~ {g_base:.4f}, sigma ~ {g_sig:.4f}")
    # print(f"  gyro_end_thresh    ~ {gyro_end:.4f}")
    # print(f"  gyro_start_thresh  ~ {gyro_start:.4f}")
    # print("Tip: if starts are missed, lower *_start; if reaches split, raise *_end or end_hold.\n")

    # --- Plot accel magnitude + thresholds ---
    plt.figure()
    plt.plot(t, accel_mag, label="accel |lin|")
    # plt.axhline(accel_start, linestyle="--", label="accel start")
    # plt.axhline(accel_end, linestyle=":", label="accel end")
    plt.xlabel("time [s]")
    plt.ylabel("accel magnitude")
    plt.title("Acceleration magnitude (gravity-compensated)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot gyro magnitude + thresholds ---
    plt.figure()
    plt.plot(t, gyro_mag, label="gyro |corr|")
    # plt.axhline(gyro_start, linestyle="--", label="gyro start")
    # plt.axhline(gyro_end, linestyle=":", label="gyro end")
    plt.xlabel("time [s]")
    plt.ylabel("gyro magnitude")
    plt.title("Gyroscope magnitude (drift-corrected)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def suggest_threshold_sets(
    accel_mag: np.ndarray,
    gyro_mag: np.ndarray,
    *,
    rest_quantile: float = 0.10,
    require_both: bool = True,
    # floors prevent "too small" thresholds
    accel_sigma_floor: float = 0.05,
    gyro_sigma_floor: float = 0.05,
):
    """
    Returns dict of presets -> thresholds.
    Presets are expressed as (end_k, start_k).
    """
    a_q = np.quantile(accel_mag, rest_quantile)
    g_q = np.quantile(gyro_mag, rest_quantile)

    if require_both:
        rest_mask = (accel_mag <= a_q) & (gyro_mag <= g_q)
        if rest_mask.sum() < max(50, 0.01 * len(accel_mag)):
            rest_mask = (accel_mag <= a_q)
    else:
        rest_mask = (accel_mag <= a_q)

    a_rest = accel_mag[rest_mask]
    g_rest = gyro_mag[rest_mask]

    a_base = float(np.median(a_rest))
    g_base = float(np.median(g_rest))

    a_sig = float(mad_sigma(a_rest))
    g_sig = float(mad_sigma(g_rest))

    # Clamp sigma so it doesn't collapse
    a_sig = max(a_sig, accel_sigma_floor)
    g_sig = max(g_sig, gyro_sigma_floor)

    presets = {
        "sensitive":   (2.0, 5.0),   # more detections, risk false positives
        "balanced":    (3.0, 7.0),   # good starting point
        "conservative":(4.0, 9.0),   # fewer detections, safer
    }

    out = {}
    for name, (end_k, start_k) in presets.items():
        out[name] = {
            "accel_end_thresh":   a_base + end_k * a_sig,
            "accel_start_thresh": a_base + start_k * a_sig,
            "gyro_end_thresh":    g_base + end_k * g_sig,
            "gyro_start_thresh":  g_base + start_k * g_sig,
            "meta": {
                "accel_baseline": a_base, "accel_sigma_used": a_sig,
                "gyro_baseline": g_base,  "gyro_sigma_used": g_sig,
                "rest_quantile": rest_quantile,
            }
        }
    return out

def print_threshold_sets(threshold_sets: dict):
    meta = next(iter(threshold_sets.values()))["meta"]
    print("\n=== Rest-baseline stats (after sigma floors) ===")
    print(f"rest_quantile: {meta['rest_quantile']}")
    print(f"accel baseline: {meta['accel_baseline']:.4f}, sigma used: {meta['accel_sigma_used']:.4f}")
    print(f"gyro  baseline: {meta['gyro_baseline']:.4f}, sigma used: {meta['gyro_sigma_used']:.4f}")
    print("\n=== Suggested threshold presets ===")
    for name, vals in threshold_sets.items():
        print(f"\n[{name}]")
        print(f"  accel_end_thresh   = {vals['accel_end_thresh']:.4f}")
        print(f"  accel_start_thresh = {vals['accel_start_thresh']:.4f}")
        print(f"  gyro_end_thresh    = {vals['gyro_end_thresh']:.4f}")
        print(f"  gyro_start_thresh  = {vals['gyro_start_thresh']:.4f}")
    print()


if __name__ == "__main__":

    main()

# What to do if thresholds too small
# Increase sigma floors (most direct fix):

# accel_sigma_floor: try 0.10, 0.20

# gyro_sigma_floor: try 0.10, 0.20

# Use a larger rest_quantile (makes “rest” include more small motion → larger sigma):

# try 0.15 instead of 0.10

# Pick a less sensitive preset:

# start with balanced

# if too many false reaches, use conservative