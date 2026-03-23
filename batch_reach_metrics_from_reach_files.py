from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

from reach_metrics import compute_reach_metrics


# --------------------------------------------
# Regex helpers
# --------------------------------------------

PARTICIPANT_RX = re.compile(r"Participant[_\s]+(\d+)", re.IGNORECASE)
SENSOR_RX = re.compile(r"sensor[-_](\d+)", re.IGNORECASE)


def extract_sensor_id(text: str) -> Optional[int]:
    m = SENSOR_RX.search(text)
    return int(m.group(1)) if m else None


# --------------------------------------------
# Normalize pickle structure
# --------------------------------------------

def normalize_pickle_to_sensor_map(obj: Dict[str, Any]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Normalize various pickle layouts to:

        sensor_map[sensor_key][reach_id] = reach_df
    """

    sensor_map: Dict[str, Dict[str, pd.DataFrame]] = {}

    for top_key, value in obj.items():

        # Case: top-level keys already include sensor-####
        sid = extract_sensor_id(str(top_key))

        if isinstance(value, dict):

            # Direct structure: { "sensor-####": { "reach00": df } }
            if sid is not None and all(hasattr(v, "columns") for v in value.values()):
                sensor_map[str(top_key)] = value
                continue

            # Structure: { "<csv_stem_with_sensor>": { "reach00": df } }
            if sid is not None:
                sensor_map.setdefault(f"sensor-{sid}", {})
                for reach_id, reach_df in value.items():
                    if hasattr(reach_df, "columns"):
                        sensor_map[f"sensor-{sid}"][f"{top_key}__{reach_id}"] = reach_df

    if not sensor_map:
        raise ValueError("Unrecognized pickle structure.")

    return sensor_map


# --------------------------------------------
# Process a single participant folder
# --------------------------------------------

def process_participant_folder(
    participant_dir: Path,
    output_root: Path,
    *,
    movement_unit_signal: str = "speed",
    movement_unit_min_prominence: float = 0.05,
    detrend_accel_mean: bool = True,
):

    print(f"\nProcessing {participant_dir.name}")

    pkl_files = sorted(participant_dir.rglob("*.pkl"))
    if not pkl_files:
        print("  No .pkl files found.")
        return

    for pkl_path in pkl_files:

        print(f"  Processing reach file: {pkl_path.name}")

        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)

        sensor_map = normalize_pickle_to_sensor_map(obj)

        reachfile_out_dir = output_root / participant_dir.name / pkl_path.stem
        reachfile_out_dir.mkdir(parents=True, exist_ok=True)

        for sensor_key, reach_dict in sensor_map.items():

            sid = extract_sensor_id(sensor_key)
            sid_name = str(sid) if sid is not None else "unknown"

            rows: List[Dict[str, Any]] = []

            for reach_id, reach_df in reach_dict.items():

                m = compute_reach_metrics(
                    reach_df,
                    movement_unit_signal=movement_unit_signal,
                    movement_unit_min_prominence=movement_unit_min_prominence,
                    detrend_accel_mean=detrend_accel_mean,
                )

                rows.append({
                    "participant": participant_dir.name,
                    "reach_file": pkl_path.name,
                    "sensor_id": sid,
                    "reach_id": reach_id,

                    "duration_s": m.duration_s,
                    "peak_speed": m.peak_speed,
                    "peak_velocity_mag": m.peak_velocity_mag,

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

            if rows:
                df = pd.DataFrame(rows).sort_values("reach_id")
                out_csv = reachfile_out_dir / f"reach_metrics_sensor-{sid_name}.csv"
                df.to_csv(out_csv, index=False)
                print(f"    Saved: {out_csv.name}")


# --------------------------------------------
# Process ALL participants
# --------------------------------------------

def process_all_participants(
    reach_files_root: str | Path = "Reach_Files",
    output_root: str | Path = "Reach_Metrics",
    exclude_participants: set[str] | None = None,
    **kwargs,
):
    reach_files_root = Path(reach_files_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not reach_files_root.exists():
        raise FileNotFoundError(f"Reach_Files folder not found: {reach_files_root.resolve()}")

    exclude_participants = exclude_participants or set()

    participant_dirs = sorted(
        [p for p in reach_files_root.iterdir()
         if p.is_dir() and "participant" in p.name.lower()]
    )

    if not participant_dirs:
        raise FileNotFoundError("No participant folders found in Reach_Files.")

    for pdir in participant_dirs:
        if pdir.name in exclude_participants:
            print(f"Skipping {pdir.name}")
            continue

        try:
            process_participant_folder(pdir, output_root, **kwargs)
        except Exception as e:
            print(f"[ERROR] Failed for {pdir.name}: {e}")


# --------------------------------------------
# Run batch
# --------------------------------------------

if __name__ == "__main__":
    process_all_participants(
        reach_files_root=r"Reach_Files",
        output_root=r"Reach_Metrics",
        exclude_participants=set(),  # e.g. {"Participant 2"}
        movement_unit_signal="speed",
        movement_unit_min_prominence=0.05,
        detrend_accel_mean=True,
    )





