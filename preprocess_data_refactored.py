"""
Refactor note:
- Original script hard-coded a single participant name and walked Output_CSVs/<participant>.
- The preprocessing core (preprocess_sensor_df, batch_preprocess_csvs) is unchanged.
- Added preprocess_participant_output_csvs() so main.py can call this per participant.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Union, List
from scipy.signal import butter, filtfilt


def preprocess_participant_output_csvs(
    participant_output_csv_root: Union[str, Path],
    participant_processed_root: Union[str, Path],
    *,
    assume_time_is_epoch_us: bool = True,
    pattern: str = "*.csv",
):
    """
    Preprocess all extracted sensor CSVs for one participant.

    Input layout (from extract_data_multi):
        Output_CSVs/<participant>/<h5_stem>/*.csv

    Output layout (used by detect_reaches):
        Processed_CSVs/<participant>/<h5_stem>/*_preprocessed.csv
    """
    participant_output_csv_root = Path(participant_output_csv_root)
    participant_processed_root = Path(participant_processed_root)
    participant_processed_root.mkdir(parents=True, exist_ok=True)

    if not participant_output_csv_root.exists():
        print(f"[SKIP] missing {participant_output_csv_root}")
        return

    for test_dir in sorted([p for p in participant_output_csv_root.iterdir() if p.is_dir()]):
        batch_preprocess_csvs(
            input_path=test_dir,
            output_dir=participant_processed_root / test_dir.name,
            pattern=pattern,
            assume_time_is_epoch_us=assume_time_is_epoch_us,
        )


def main():
    # Small example. main.py is the recommended entrypoint now.
    preprocess_participant_output_csvs(
        participant_output_csv_root="Output_CSVs/Participant 1",
        participant_processed_root="Processed_CSVs/Participant 1",
        assume_time_is_epoch_us=True,
    )


# -----------------------------
# Time conversion: epoch -> elapsed
# -----------------------------
def epoch_to_elapsed_seconds(epoch_times) -> np.ndarray:
    epoch_times = np.asarray(epoch_times, dtype=float)
    if epoch_times.size == 0:
        raise ValueError("Empty time array.")
    t0 = epoch_times[0]
    return (epoch_times - t0) / 1e6  # microseconds -> seconds


def estimate_sampling_rate(time: np.ndarray) -> float:
    dt = np.diff(time)
    dt = dt[np.isfinite(dt)]
    if len(dt) == 0:
        raise ValueError("Could not estimate sampling rate from time array.")
    median_dt = np.median(dt)
    if median_dt <= 0:
        raise ValueError("Non-positive time step detected.")
    return 1.0 / median_dt


def butter_lowpass(cutoff_hz: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype="low")
    return b, a


def butter_highpass(cutoff_hz: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype="high")
    return b, a


def apply_filter(series: pd.Series, b, a) -> pd.Series:
    data = series.to_numpy().astype(float)

    nans = ~np.isfinite(data)
    if np.any(nans):
        valid_idx = np.flatnonzero(~nans)
        if len(valid_idx) == 0:
            return series
        data[nans] = np.interp(np.flatnonzero(nans), valid_idx, data[valid_idx])

    filtered = filtfilt(b, a, data)
    return pd.Series(filtered, index=series.index)


def preprocess_sensor_df(
    df: pd.DataFrame,
    accel_cutoff_hz: float = 20.0,
    gyro_cutoff_hz: float = 20.0,
    gravity_cutoff_hz: float = 0.4,
    gyro_drift_cutoff_hz: float = 0.05,
    assume_time_is_epoch_us: bool = True,
) -> pd.DataFrame:
    if "time" not in df.columns:
        raise KeyError("Expected a 'time' column.")

    df_out = df.copy()

    if assume_time_is_epoch_us:
        df_out.rename(columns={"time": "time_epoch"}, inplace=True)
        elapsed = epoch_to_elapsed_seconds(df_out["time_epoch"].to_numpy())
        df_out["time"] = elapsed
    else:
        df_out["time"] = df_out["time"].astype(float)

    time = df_out["time"].to_numpy()
    fs = estimate_sampling_rate(time)
    print(f"Estimated sampling rate: {fs:.2f} Hz")

    rename_map = {}
    for c in df_out.columns:
        if c.startswith("accel_Accelerometer_"):
            idx = c.split("_")[-1]
            rename_map[c] = f"accel_{idx}"
        if c.startswith("gyro_Gyroscope_"):
            idx = c.split("_")[-1]
            rename_map[c] = f"gyro_{idx}"

    df_out = df_out.rename(columns=rename_map)

    accel_cols = sorted([c for c in df_out.columns if c.startswith("accel_") and c[-1].isdigit()])
    gyro_cols  = sorted([c for c in df_out.columns if c.startswith("gyro_")  and c[-1].isdigit()])

    b_accel_lp, a_accel_lp = butter_lowpass(accel_cutoff_hz, fs)
    b_gyro_lp, a_gyro_lp = butter_lowpass(gyro_cutoff_hz, fs)
    b_grav_lp, a_grav_lp = butter_lowpass(gravity_cutoff_hz, fs)
    b_gyro_hp, a_gyro_hp = butter_highpass(gyro_drift_cutoff_hz, fs)

    for col in accel_cols:
        axis = col.split("_")[-1]
        filt_col = f"accel_{axis}_filt"
        grav_col = f"accel_{axis}_grav"
        lin_col  = f"accel_{axis}_lin"

        df_out[filt_col] = apply_filter(df_out[col], b_accel_lp, a_accel_lp)
        df_out[grav_col] = apply_filter(df_out[filt_col], b_grav_lp, a_grav_lp)
        df_out[lin_col]  = df_out[filt_col] - df_out[grav_col]

    for col in gyro_cols:
        axis = col.split("_")[-1]
        filt_col = f"gyro_{axis}_filt"
        corr_col = f"gyro_{axis}_corr"

        df_out[filt_col] = apply_filter(df_out[col], b_gyro_lp, a_gyro_lp)
        df_out[corr_col] = apply_filter(df_out[filt_col], b_gyro_hp, a_gyro_hp)

    desired_order = [
        "time",
        "accel_0_filt", "accel_1_filt", "accel_2_filt",
        "accel_0_lin",  "accel_1_lin",  "accel_2_lin",
        "gyro_0_filt",  "gyro_1_filt",  "gyro_2_filt",
        "gyro_0_corr",  "gyro_1_corr",  "gyro_2_corr",
    ]

    df_new = pd.DataFrame()
    for col in desired_order:
        if col in df_out.columns:
            df_new[col] = df_out[col]

    return df_new


def preprocess_csv_file(
    csv_path: Union[str, Path],
    output_path: Union[str, Path] = None,
    **pre_kwargs,
):
    csv_path = Path(csv_path)
    if output_path is None:
        output_path = csv_path.with_name(csv_path.stem + "_preprocessed.csv")
    else:
        output_path = Path(output_path)

    df = pd.read_csv(csv_path)
    df_pre = preprocess_sensor_df(df, **pre_kwargs)
    df_pre.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


def batch_preprocess_csvs(
    input_path: Union[str, Path],
    output_dir: Union[str, Path] = None,
    pattern: str = "*.csv",
    **pre_kwargs,
):
    input_path = Path(input_path)

    if input_path.is_file():
        if output_dir is None:
            preprocess_csv_file(input_path, **pre_kwargs)
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / (input_path.stem + "_preprocessed.csv")
            preprocess_csv_file(input_path, output_path=out_path, **pre_kwargs)
        return

    if not input_path.is_dir():
        raise FileNotFoundError(f"{input_path} is not a valid file or directory")

    csv_files: List[Path] = sorted(input_path.glob(pattern))
    if not csv_files:
        print(f"No CSV files in {input_path} matching {pattern}")
        return

    if output_dir is None:
        output_dir = input_path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in csv_files:
        out_path = output_dir / (csv_file.stem + "_preprocessed.csv")
        preprocess_csv_file(csv_file, output_path=out_path, **pre_kwargs)


if __name__ == "__main__":
    main()
