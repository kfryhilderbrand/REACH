import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np


def main():
    # raw_df, proc_df = load_raw_and_processed("Output_CSVs/20250915-123550_Free_Form_sensor-17738.csv")

    # # Plot accel axis 0 for first 10 seconds
    # plot_accel_gravity_compensation(raw_df, proc_df, axis=0, t_max=10.0)

    # # Plot gyro axis 2 for full duration
    # plot_gyro_drift_correction(raw_df, proc_df, axis=ss2, t_max=None)

    # Plot magnitude of the acceleration and gyroscope
    plot_accel_and_gyro_magnitude("Processed_CSVs/Participant 1/20251017-090642_Free_Form/20251017-090642_Free_Form_sensor-21263_preprocessed.csv")

def plot_accel_gravity_compensation(
    raw_df: pd.DataFrame,
    proc_df: pd.DataFrame,
    axis: int = 0,
    t_max: float | None = None,
):
    """
    Plot accelerometer gravity-compensation for one axis:
    - raw accel
    - filtered accel (accel_<axis>_filt)
    - gravity-compensated accel (accel_<axis>_lin)
    
    axis: 0, 1, or 2
    t_max: limit x-axis to [0, t_max] seconds (None = full length)
    """
    # Column names
    raw_col = f"accel_Accelerometer_{axis}"
    filt_col = f"accel_{axis}_filt"
    lin_col  = f"accel_{axis}_lin"

    if raw_col not in raw_df.columns:
        raise KeyError(f"Raw column '{raw_col}' not found in raw_df.")
    for c in (filt_col, lin_col, "time"):
        if c not in proc_df.columns:
            raise KeyError(f"Processed column '{c}' not found in proc_df.")

    # Use processed time axis (elapsed seconds)
    t = proc_df["time"].to_numpy()

    # Align lengths if needed (just in case)
    n = min(len(raw_df), len(proc_df))
    t = t[:n]
    raw = raw_df[raw_col].to_numpy()[:n]
    filt = proc_df[filt_col].to_numpy()[:n]
    lin = proc_df[lin_col].to_numpy()[:n]

    if t_max is not None:
        mask = t <= t_max
        t, raw, filt, lin = t[mask], raw[mask], filt[mask], lin[mask]

    plt.figure()
    plt.plot(t, raw, label="raw accel")
    plt.plot(t, filt, label="filtered accel")
    plt.plot(t, lin, label="linear accel (gravity-comp.)")
    plt.xlabel("time [s]")
    plt.ylabel("acceleration")
    plt.title(f"Accelerometer axis {axis}: gravity compensation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_gyro_drift_correction(
    raw_df: pd.DataFrame,
    proc_df: pd.DataFrame,
    axis: int = 0,
    t_max: float | None = None,
):
    """
    Plot gyroscope drift correction for one axis:
    - raw gyro
    - filtered gyro (gyro_<axis>_filt)
    - drift-corrected gyro (gyro_<axis>_corr)
    
    axis: 0, 1, or 2
    t_max: limit x-axis to [0, t_max] seconds (None = full length)
    """
    raw_col = f"gyro_Gyroscope_{axis}"
    filt_col = f"gyro_{axis}_filt"
    corr_col = f"gyro_{axis}_corr"

    if raw_col not in raw_df.columns:
        raise KeyError(f"Raw column '{raw_col}' not found in raw_df.")
    for c in (filt_col, corr_col, "time"):
        if c not in proc_df.columns:
            raise KeyError(f"Processed column '{c}' not found in proc_df.")

    t = proc_df["time"].to_numpy()

    n = min(len(raw_df), len(proc_df))
    t = t[:n]
    raw = raw_df[raw_col].to_numpy()[:n]
    filt = proc_df[filt_col].to_numpy()[:n]
    corr = proc_df[corr_col].to_numpy()[:n]

    if t_max is not None:
        mask = t <= t_max
        t, raw, filt, corr = t[mask], raw[mask], filt[mask], corr[mask]

    plt.figure()
    plt.plot(t, raw, label="raw gyro")
    plt.plot(t, filt, label="filtered gyro")
    plt.plot(t, corr, label="drift-corrected gyro")
    plt.xlabel("time [s]")
    plt.ylabel("angular velocity")
    plt.title(f"Gyroscope axis {axis}: drift correction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def load_raw_and_processed(raw_csv_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_csv_path = Path(raw_csv_path)
    proc_csv_path = raw_csv_path.with_name(raw_csv_path.stem + "_preprocessed.csv")

    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {raw_csv_path}")
    if not proc_csv_path.exists():
        raise FileNotFoundError(f"Processed CSV not found: {proc_csv_path}")

    raw_df = pd.read_csv(raw_csv_path)
    proc_df = pd.read_csv(proc_csv_path)
    return raw_df, proc_df


def plot_accel_and_gyro_magnitude(csv_relative_path: str):
    """
    Load a processed CSV and plot:
      1) acceleration linear magnitude
      2) gyroscope corrected magnitude

    csv_relative_path: path relative to your working directory
    """
    csv_path = Path(csv_relative_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_accel = ["accel_0_lin", "accel_1_lin", "accel_2_lin"]
    required_gyro  = ["gyro_0_corr", "gyro_1_corr", "gyro_2_corr"]

    for c in required_accel + required_gyro + ["time"]:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}' in CSV")

    time = df["time"].to_numpy()

    # ---- Magnitudes ----
    accel_mag = np.linalg.norm(df[required_accel].to_numpy(), axis=1)
    gyro_mag  = np.linalg.norm(df[required_gyro].to_numpy(), axis=1)

    # ---- Plot acceleration magnitude ----
    plt.figure()
    plt.plot(time, accel_mag)
    plt.xlabel("time [s]")
    plt.ylabel("accel linear magnitude")
    plt.title("Acceleration magnitude (gravity-compensated)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---- Plot gyroscope magnitude ----
    plt.figure()
    plt.plot(time, gyro_mag)
    plt.xlabel("time [s]")
    plt.ylabel("gyro corrected magnitude")
    plt.title("Gyroscope magnitude (drift-corrected)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    main()
