from __future__ import annotations

import re
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd


# ======================================================
# Exclusion rules 
# ======================================================

@dataclass
class ExclusionRules:
    exclude_participant_nums: set[int] = field(default_factory=set)
    exclude_participant_folder_names: set[str] = field(default_factory=set)

    exclude_test_folder_names: set[str] = field(default_factory=set)
    exclude_test_substrings: List[str] = field(default_factory=list)
    exclude_test_regexes: List[re.Pattern] = field(default_factory=list)

    exclude_sensor_ids: set[int] = field(default_factory=set)
    exclude_file_substrings: List[str] = field(default_factory=list)
    exclude_file_regexes: List[re.Pattern] = field(default_factory=list)

    def test_folder_excluded(self, name: str) -> bool:
        if name in self.exclude_test_folder_names:
            return True
        if any(s in name for s in self.exclude_test_substrings):
            return True
        if any(rx.search(name) for rx in self.exclude_test_regexes):
            return True
        return False

    def file_excluded(self, filename: str, sensor_id: Optional[int]) -> bool:
        if sensor_id is not None and sensor_id in self.exclude_sensor_ids:
            return True
        if any(s in filename for s in self.exclude_file_substrings):
            return True
        if any(rx.search(filename) for rx in self.exclude_file_regexes):
            return True
        return False


# ======================================================
# File parsing helpers
# ======================================================

SENSOR_ID_RX = re.compile(r"_sensor-(\d+)_preprocessed\.csv$", re.IGNORECASE)

def parse_participant_number(folder_name: str) -> Optional[int]:
    m = re.search(r"Participant\s+(\d+)", folder_name, re.IGNORECASE)
    return int(m.group(1)) if m else None

def parse_sensor_id(filename: str) -> Optional[int]:
    m = SENSOR_ID_RX.search(filename)
    return int(m.group(1)) if m else None

def iter_csv_files(root: Path, exclusions: ExclusionRules):
    for participant_dir in root.iterdir():
        if not participant_dir.is_dir():
            continue

        pnum = parse_participant_number(participant_dir.name)
        if pnum is None:
            continue
        if pnum in exclusions.exclude_participant_nums:
            continue

        for test_dir in participant_dir.iterdir():
            if not test_dir.is_dir():
                continue
            if exclusions.test_folder_excluded(test_dir.name):
                continue

            for csv_path in test_dir.glob("*.csv"):
                sensor_id = parse_sensor_id(csv_path.name)
                if exclusions.file_excluded(csv_path.name, sensor_id):
                    continue

                yield pnum, test_dir.name, csv_path


# ======================================================
# Reach detection
# ======================================================

# def accel_linear_magnitude(df: pd.DataFrame) -> np.ndarray:
#     cols = ["accel_0_lin", "accel_1_lin", "accel_2_lin"]
#     if not all(c in df.columns for c in cols):
#         raise KeyError("Missing accel_*_lin columns")
#     a = df[cols].to_numpy(dtype=float)
#     return np.linalg.norm(a, axis=1)


def magnitude_from_cols(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = df[cols].to_numpy(dtype=float)
    return np.linalg.norm(x, axis=1)

def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    kernel = np.ones(win) / win
    return np.convolve(x, kernel, mode="same")

def find_reaches(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    # Which signals to use
    use_gyro: bool = True,
    # Thresholds (tune these!)
    accel_start_thresh: float = 1.2, #0.20,
    accel_end_thresh: float = 0.6, #0.08,
    gyro_start_thresh: float = 1.2, #0.50,
    gyro_end_thresh: float = 0.6, #0.20,
    # Smoothing
    smooth_window_s: float = 0.03,
    # End condition must hold for this long
    end_hold_s: float = 0.05,
    # Reach sanity checks
    min_duration_s: float = 0.12,
    require_peak: bool = True,
    peak_prominence: float = 0.02,
    # Merge reaches with tiny gaps (optional)
    merge_gap_s: float = 0.02,
) -> list[tuple[int, int]]:
    """
    Detect 'reaches' as burst-like events: rise -> peak -> return near zero.

    Returns list of (start_idx, end_idx) inclusive.
    Uses accel linear magnitude (and optionally gyro corrected magnitude).
    """

    if time_col not in df.columns:
        raise KeyError(f"Missing '{time_col}' column")

    # Required columns for accel_lin
    accel_cols = ["accel_0_lin", "accel_1_lin", "accel_2_lin"]
    if not all(c in df.columns for c in accel_cols):
        raise KeyError("Expected accel_0_lin/accel_1_lin/accel_2_lin in processed CSV.")

    t = df[time_col].to_numpy(dtype=float)
    if len(t) < 5:
        return []

    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        return []
    fs = 1.0 / dt

    a = magnitude_from_cols(df, accel_cols)

    if use_gyro:
        gyro_cols = ["gyro_0_corr", "gyro_1_corr", "gyro_2_corr"]
        if not all(c in df.columns for c in gyro_cols):
            raise KeyError("use_gyro=True but gyro_*_corr columns are missing.")
        g = magnitude_from_cols(df, gyro_cols)
    else:
        g = None

    # Smooth magnitude(s) a bit to reduce noise-triggering
    win = max(1, int(round(smooth_window_s * fs)))
    a_s = moving_average(a, win)
    if g is not None:
        g_s = moving_average(g, win)

    # Define "active" using hysteresis with separate start/end thresholds
    # Start if accel OR gyro exceeds start thresholds.
    # End when BOTH are below end thresholds (if gyro used), held for end_hold_s.
    def is_start(i: int) -> bool:
        if g is None:
            return a_s[i] >= accel_start_thresh
        return (a_s[i] >= accel_start_thresh) or (g_s[i] >= gyro_start_thresh)

    def is_end_condition(i: int) -> bool:
        if g is None:
            return a_s[i] <= accel_end_thresh
        return (a_s[i] <= accel_end_thresh) and (g_s[i] <= gyro_end_thresh)

    end_hold = max(1, int(round(end_hold_s * fs)))
    min_len = max(1, int(round(min_duration_s * fs)))
    merge_gap = max(0, int(round(merge_gap_s * fs)))

    reaches: list[tuple[int, int]] = []
    i = 0
    n = len(a_s)

    while i < n:
        # Find the next start
        while i < n and not is_start(i):
            i += 1
        if i >= n:
            break
        start = i

        # Now advance until we see end condition held for end_hold samples
        i += 1
        end = None
        hold = 0
        while i < n:
            if is_end_condition(i):
                hold += 1
                if hold >= end_hold:
                    end = i  # end idx inclusive (includes hold zone)
                    break
            else:
                hold = 0
            i += 1

        if end is None:
            end = n - 1  # reached end of file

        # Basic duration filter
        if end - start + 1 < min_len:
            i = end + 1
            continue

        # Optional: require a peak (rise then fall) with some prominence
        if require_peak:
            seg = a_s[start : end + 1]
            peak_idx = int(np.argmax(seg))
            peak_val = float(seg[peak_idx])
            start_val = float(seg[0])
            end_val = float(seg[-1])
            # peak should be above both ends by at least peak_prominence
            if not ((peak_val - start_val >= peak_prominence) and (peak_val - end_val >= peak_prominence)):
                i = end + 1
                continue

        reaches.append((start, end))
        i = end + 1

    # Optional merge if two reaches separated by a tiny gap
    if reaches and merge_gap > 0:
        merged = [reaches[0]]
        for s, e in reaches[1:]:
            ps, pe = merged[-1]
            if s - pe - 1 <= merge_gap:
                merged[-1] = (ps, e)
            else:
                merged.append((s, e))
        reaches = merged

    return reaches

# ======================================================
# MAIN LOGIC
# ======================================================

def extract_reach_pickles(
    processed_root: str | Path,
    output_root: str | Path = "Reach_Files",
    exclusions: ExclusionRules | None = None,
    threshold: float = 0.25,
):
    processed_root = Path(processed_root)
    output_root = Path(output_root)
    exclusions = exclusions or ExclusionRules()

    # Structure:
    # participant -> test -> { csv_stem -> { seg_id -> df } }
    grouped: Dict[int, Dict[str, Dict[str, Dict[str, pd.DataFrame]]]] = {}

    for pnum, test_name, csv_path in iter_csv_files(processed_root, exclusions):
        df = pd.read_csv(csv_path)
        #segments = find_activity_periods(df, threshold=threshold)
        #reaches = find_reaches(df, accel_start_thresh=2, accel_end_thresh=1, use_gyro=True)

        reaches = find_reaches(
            df,
            use_gyro=True,
            accel_start_thresh=1.2,
            accel_end_thresh=0.6,
            gyro_start_thresh=1.2,
            gyro_end_thresh=0.6,
            smooth_window_s=0.03,
            end_hold_s=0.05,
            min_duration_s=0.12,
            require_peak=False,
            peak_prominence=0.02,
            merge_gap_s=0.02,
        )


        if not reaches:
            continue

        csv_dict: Dict[str, pd.DataFrame] = {}
        for i, (s, e) in enumerate(reaches):
            reach_id = f"reach{i:02d}"
            csv_dict[reach_id] = df.iloc[s : e + 1].copy()

        grouped.setdefault(pnum, {}).setdefault(test_name, {})[csv_path.stem] = csv_dict

    # --------------------------------------------------
    # Save one pickle per test folder
    # --------------------------------------------------
    for pnum, tests in grouped.items():
        for test_name, test_dict in tests.items():
            out_dir = output_root / f"Participant {pnum}" 
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / f"{test_name}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(test_dict, f)

            print(f"Saved {out_path}")


# ======================================================
# Example usage
# ======================================================

if __name__ == "__main__":
    extract_reach_pickles(
        processed_root="Processed_CSVs",
        output_root="Reach_Files",
        exclusions=ExclusionRules(
            exclude_test_substrings=["calibration"],
            exclude_sensor_ids={99999},
        ),
        threshold=0.25,
    )


## Each pickle file contains:

# {
#   "<testfolder>_sensor-19392_preprocessed": {
#       "seg00": DataFrame,
#       "seg01": DataFrame,
#   },
#   "<testfolder>_sensor-20481_preprocessed": {
#       "seg00": DataFrame,
#   }
# }

## Example to access data
# import pickle

# with open("Activity_Files/Participant 3/2026-01-10_13-15-22/2026-01-10_13-15-22.pkl", "rb") as f:
#     data = pickle.load(f)

# df_segment = data["2026-01-10_13-15-22_sensor-19392_preprocessed"]["seg00"]