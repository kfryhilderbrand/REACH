from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def collect_pairs(project_root: Path) -> list[tuple[Path, Path]]:
    output_root = project_root / "Project/Output_CSVs"
    processed_root = project_root / "Project/Processed_CSVs"

    pairs: list[tuple[Path, Path]] = []
    for raw_csv in sorted(output_root.rglob("*_sensor-*.csv")):
        participant_dir = raw_csv.parents[1].name
        trial_dir = raw_csv.parent.name
        processed_csv = (
            processed_root
            / participant_dir
            / trial_dir
            / f"{raw_csv.stem}_preprocessed.csv"
        )
        if processed_csv.exists():
            pairs.append((raw_csv, processed_csv))

    return pairs


def load_pair(raw_csv: Path, processed_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(raw_csv), pd.read_csv(processed_csv)


def magnitude(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.linalg.norm(df[cols].to_numpy(dtype=float), axis=1)


def plot_pair(raw_csv: Path, processed_csv: Path, t_max: float | None = None) -> None:
    raw_df, proc_df = load_pair(raw_csv, processed_csv)

    raw_time = (raw_df["time"].to_numpy(dtype=float) - float(raw_df["time"].iloc[0])) / 1e6
    proc_time = proc_df["time"].to_numpy(dtype=float)

    raw_accel_cols = [f"accel_Accelerometer_{i}" for i in range(3)]
    raw_gyro_cols = [f"gyro_Gyroscope_{i}" for i in range(3)]
    proc_accel_cols = [f"accel_{i}_lin" for i in range(3)]
    proc_gyro_cols = [f"gyro_{i}_corr" for i in range(3)]
    proc_filt_cols = [f"accel_{i}_filt" for i in range(3)]

    raw_accel_mag = magnitude(raw_df, raw_accel_cols)
    raw_gyro_mag = magnitude(raw_df, raw_gyro_cols)
    proc_accel_mag = magnitude(proc_df, proc_accel_cols)
    proc_gyro_mag = magnitude(proc_df, proc_gyro_cols)

    if t_max is not None:
        raw_mask = raw_time <= t_max
        proc_mask = proc_time <= t_max
    else:
        raw_mask = slice(None)
        proc_mask = slice(None)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex="col")
    fig.suptitle(f"Raw vs Processed: {raw_csv.name}")

    axes[0, 0].plot(raw_time[raw_mask], raw_accel_mag[raw_mask], label="raw accel |a|", alpha=0.8)
    axes[0, 0].plot(proc_time[proc_mask], proc_accel_mag[proc_mask], label="processed linear accel |a|", alpha=0.8)
    axes[0, 0].set_ylabel("Acceleration")
    axes[0, 0].set_title("Acceleration Magnitude")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(raw_time[raw_mask], raw_gyro_mag[raw_mask], label="raw gyro |g|", alpha=0.8)
    axes[0, 1].plot(proc_time[proc_mask], proc_gyro_mag[proc_mask], label="processed corrected gyro |g|", alpha=0.8)
    axes[0, 1].set_ylabel("Angular Velocity")
    axes[0, 1].set_title("Gyroscope Magnitude")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    for axis in range(3):
        axes[1, 0].plot(
            raw_time[raw_mask],
            raw_df[f"accel_Accelerometer_{axis}"].to_numpy(dtype=float)[raw_mask],
            label=f"raw accel_{axis}",
            alpha=0.6,
        )
        axes[1, 0].plot(
            proc_time[proc_mask],
            proc_df[f"accel_{axis}_lin"].to_numpy(dtype=float)[proc_mask],
            label=f"linear accel_{axis}",
            linestyle="--",
            alpha=0.8,
        )
        axes[1, 1].plot(
            proc_time[proc_mask],
            proc_df[f"accel_{axis}_filt"].to_numpy(dtype=float)[proc_mask],
            label=f"filtered accel_{axis}",
            alpha=0.7,
        )
        axes[1, 1].plot(
            proc_time[proc_mask],
            proc_df[f"accel_{axis}_lin"].to_numpy(dtype=float)[proc_mask],
            label=f"linear accel_{axis}",
            linestyle="--",
            alpha=0.8,
        )

    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Acceleration")
    axes[1, 0].set_title("Raw vs Linear Accel by Axis")
    axes[1, 0].grid(True)
    axes[1, 0].legend(ncol=2, fontsize=8)

    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("Acceleration")
    axes[1, 1].set_title("Filtered vs Linear Accel by Axis")
    axes[1, 1].grid(True)
    axes[1, 1].legend(ncol=2, fontsize=8)

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot raw sensor data versus processed sensor data for a random example sensor."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project folder containing Output_CSVs and Processed_CSVs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible selection.",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=None,
        help="Optional time limit in seconds for the plot.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    rng = random.Random(args.seed)
    pairs = collect_pairs(project_root)

    if not pairs:
        raise FileNotFoundError(
            f"No matching raw/processed CSV pairs found under {project_root}"
        )

    raw_csv, processed_csv = rng.choice(pairs)
    print(f"Selected raw CSV: {raw_csv}")
    print(f"Selected processed CSV: {processed_csv}")
    plot_pair(raw_csv, processed_csv, t_max=args.t_max)


if __name__ == "__main__":
    main()
