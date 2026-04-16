from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Iterable

import numpy as np
import pandas as pd


# =========================
# Utility math
# =========================

def _as_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)

def _trapz(y: np.ndarray, t: np.ndarray) -> float:
    # Robust trapezoid integration
    if len(y) < 2:
        return 0.0
    return float(np.trapezoid(y, t))

def _cumtrapz(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Cumulative trapezoid integral with same length as y.
    """
    y = _as_float_array(y)
    t = _as_float_array(t)
    n = len(y)
    out = np.zeros(n, dtype=float)
    if n < 2:
        return out
    dt = np.diff(t)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dt)
    return out

def _gradient(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Time-derivative dy/dt using numpy.gradient (handles nonuniform t).
    """
    y = _as_float_array(y)
    t = _as_float_array(t)
    if len(y) < 2:
        return np.zeros_like(y)
    return np.gradient(y, t)

def _local_maxima_count(x: np.ndarray, min_prominence: float = 0.0) -> int:
    """
    Counts local maxima in x using simple neighbor comparison.
    min_prominence is a simple amplitude threshold against neighbors.
    """
    x = _as_float_array(x)
    if len(x) < 3:
        return 0
    # strict local max: x[i-1] < x[i] > x[i+1]
    peaks = 0
    for i in range(1, len(x) - 1):
        if x[i] > x[i - 1] and x[i] > x[i + 1]:
            if min_prominence <= 0:
                peaks += 1
            else:
                # simple prominence vs immediate neighbors
                if (x[i] - max(x[i - 1], x[i + 1])) >= min_prominence:
                    peaks += 1
    return peaks


# =========================
# Core reach metrics
# =========================

@dataclass
class ReachMetrics:
    # Duration
    duration_s: float

    # Velocity / speed peaks
    peak_velocity_mag: float
    peak_speed: float  # same as peak_velocity_mag unless you treat them differently

    # Average velocity/speed
    avg_velocity_vec: Tuple[float, float, float]
    avg_velocity_mag: float
    avg_speed: float

    # Movement units
    movement_units: int

    # Timing
    time_to_peak_speed_s: float

    # Path length
    path_length: float

    # Jerk
    jerk_mean: float
    jerk_rms: float
    jerk_peak: float


def compute_reach_metrics(
    reach_df: pd.DataFrame,
    *,
    time_col: str = "time",
    accel_cols: Tuple[str, str, str] = ("accel_0_lin", "accel_1_lin", "accel_2_lin"),
    # Movement unit detection uses speed peaks by default
    movement_unit_signal: str = "speed",  # "speed" or "accel_mag"
    movement_unit_min_prominence: float = 0.0,
    # Optional: remove residual bias in accel within the reach to reduce integration drift
    detrend_accel_mean: bool = True,
) -> ReachMetrics:
    """
    Compute metrics for one reach segment.

    Notes:
    - Velocity is obtained by integrating linear acceleration over time.
      Without absolute initial velocity, we assume v(t0)=0 by construction.
      If your reach segments truly begin at rest, this is reasonable.
    - Displacement is obtained by integrating velocity (v) over time.
    - Path length = integral of speed over time.
    - "Movement units" is implemented as the number of peaks in the chosen signal
      (default: speed), where each peak corresponds to accel then decel.
    """

    for c in (time_col, *accel_cols):
        if c not in reach_df.columns:
            raise KeyError(f"Missing required column '{c}' in reach segment.")

    t = _as_float_array(reach_df[time_col].to_numpy())
    if len(t) < 2:
        # Degenerate reach
        return ReachMetrics(
            duration_s=0.0,
            peak_velocity_mag=0.0,
            peak_speed=0.0,
            avg_velocity_vec=(0.0, 0.0, 0.0),
            avg_velocity_mag=0.0,
            avg_speed=0.0,
            movement_units=0,
            time_to_peak_speed_s=0.0,
            path_length=0.0,
            jerk_mean=0.0,
            jerk_rms=0.0,
            jerk_peak=0.0,
        )

    # Duration
    duration = float(t[-1] - t[0])

    # Acceleration (linear)
    ax = _as_float_array(reach_df[accel_cols[0]].to_numpy())
    ay = _as_float_array(reach_df[accel_cols[1]].to_numpy())
    az = _as_float_array(reach_df[accel_cols[2]].to_numpy())

    if detrend_accel_mean:
        # Subtract mean within reach to reduce integration drift
        ax = ax - float(np.mean(ax))
        ay = ay - float(np.mean(ay))
        az = az - float(np.mean(az))

    # Integrate accel -> velocity (assuming v0=0)
    vx = _cumtrapz(ax, t)
    vy = _cumtrapz(ay, t)
    vz = _cumtrapz(az, t)

    vmag = np.linalg.norm(np.column_stack([vx, vy, vz]), axis=1)
    speed = vmag  # same thing here (speed = |v|)

    peak_speed = float(np.max(speed))
    peak_velocity_mag = peak_speed

    # Average velocity:
    # - vector average over the reach duration = displacement / duration
    dx = _trapz(vx, t)
    dy = _trapz(vy, t)
    dz = _trapz(vz, t)
    if duration > 0:
        avg_vx, avg_vy, avg_vz = dx / duration, dy / duration, dz / duration
    else:
        avg_vx, avg_vy, avg_vz = 0.0, 0.0, 0.0

    avg_velocity_mag = float(np.linalg.norm([avg_vx, avg_vy, avg_vz]))

    # Average speed = mean(|v|)
    avg_speed = float(np.mean(speed))

    # Time to peak speed
    peak_idx = int(np.argmax(speed))
    time_to_peak = float(t[peak_idx] - t[0])

    # Path length = integral of speed over time
    path_length = _trapz(speed, t)

    # Jerk (time-derivative of acceleration vector)
    jx = _gradient(ax, t)
    jy = _gradient(ay, t)
    jz = _gradient(az, t)
    jmag = np.linalg.norm(np.column_stack([jx, jy, jz]), axis=1)

    jerk_mean = float(np.mean(jmag))
    jerk_rms = float(np.sqrt(np.mean(jmag ** 2)))
    jerk_peak = float(np.max(jmag))

    # Movement units: count peaks in chosen signal
    if movement_unit_signal == "speed":
        mu_signal = speed
    elif movement_unit_signal == "accel_mag":
        mu_signal = np.linalg.norm(np.column_stack([ax, ay, az]), axis=1)
    else:
        raise ValueError("movement_unit_signal must be 'speed' or 'accel_mag'")

    movement_units = _local_maxima_count(mu_signal, min_prominence=movement_unit_min_prominence)

    return ReachMetrics(
        duration_s=duration,
        peak_velocity_mag=peak_velocity_mag,
        peak_speed=peak_speed,
        avg_velocity_vec=(float(avg_vx), float(avg_vy), float(avg_vz)),
        avg_velocity_mag=avg_velocity_mag,
        avg_speed=avg_speed,
        movement_units=movement_units,
        time_to_peak_speed_s=time_to_peak,
        path_length=path_length,
        jerk_mean=jerk_mean,
        jerk_rms=jerk_rms,
        jerk_peak=jerk_peak,
    )


# =========================
# Pickle processing helpers
# =========================

def load_reach_pickle(pkl_path: str | Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Expected pickle structure from your pipeline:
      {
        "<csv_stem>": {
           "reach00": DataFrame,
           "reach01": DataFrame,
           ...
        },
        ...
      }
    """
    import pickle
    p = Path(pkl_path)
    with open(p, "rb") as f:
        obj = pickle.load(f)
    return obj


def summarize_pickle_to_dataframe(
    pkl_path: str | Path,
    *,
    movement_unit_signal: str = "speed",
    movement_unit_min_prominence: float = 0.0,
    detrend_accel_mean: bool = True,
) -> pd.DataFrame:
    """
    Computes metrics for every reach in every csv_stem within a pickle.
    Returns a flat table (DataFrame) suitable for saving to CSV.
    """
    data = load_reach_pickle(pkl_path)

    rows: list[dict[str, Any]] = []
    for csv_stem, reach_dict in data.items():
        for reach_id, reach_df in reach_dict.items():
            start_time_s = None
            if "time" in reach_df.columns and not reach_df.empty:
                start_time_s = float(reach_df["time"].iloc[0])

            m = compute_reach_metrics(
                reach_df,
                movement_unit_signal=movement_unit_signal,
                movement_unit_min_prominence=movement_unit_min_prominence,
                detrend_accel_mean=detrend_accel_mean,
            )
            rows.append({
                "pickle_file": str(Path(pkl_path).name),
                "csv_stem": csv_stem,
                "reach_id": reach_id,
                "start_time_s": start_time_s,

                "duration_s": m.duration_s,

                "peak_velocity_mag": m.peak_velocity_mag,
                "peak_speed": m.peak_speed,

                "avg_velocity_x": m.avg_velocity_vec[0],
                "avg_velocity_y": m.avg_velocity_vec[1],
                "avg_velocity_z": m.avg_velocity_vec[2],
                "avg_velocity_mag": m.avg_velocity_mag,
                "avg_speed": m.avg_speed,

                "movement_units": m.movement_units,
                "time_to_peak_speed_s": m.time_to_peak_speed_s,

                "path_length": m.path_length,

                "jerk_mean": m.jerk_mean,
                "jerk_rms": m.jerk_rms,
                "jerk_peak": m.jerk_peak,
            })

    return pd.DataFrame(rows)


def write_summary_csv_for_pickle(
    pkl_path: str | Path,
    out_csv_path: str | Path,
    **kwargs,
) -> None:
    df = summarize_pickle_to_dataframe(pkl_path, **kwargs)
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    
