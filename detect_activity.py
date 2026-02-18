from __future__ import annotations

import re
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd


# ======================================================
# Exclusion rules (unchanged from before)
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
# Activity detection (same logic as before)
# ======================================================

def accel_linear_magnitude(df: pd.DataFrame) -> np.ndarray:
    cols = ["accel_0_lin", "accel_1_lin", "accel_2_lin"]
    if not all(c in df.columns for c in cols):
        raise KeyError("Missing accel_*_lin columns")
    a = df[cols].to_numpy(dtype=float)
    return np.linalg.norm(a, axis=1)


def find_activity_periods(
    df: pd.DataFrame,
    threshold: float = 0.25,
    min_duration_s: float = 1.0,
    merge_gap_s: float = 0.5,
) -> List[Tuple[int, int]]:
    t = df["time"].to_numpy()
    fs = 1.0 / np.median(np.diff(t))

    mag = accel_linear_magnitude(df)
    active = mag > threshold

    segments = []
    in_seg = False
    for i, val in enumerate(active):
        if val and not in_seg:
            start = i
            in_seg = True
        elif not val and in_seg:
            segments.append((start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((start, len(active) - 1))

    min_len = int(min_duration_s * fs)
    segments = [(s, e) for s, e in segments if (e - s + 1) >= min_len]

    merged = []
    max_gap = int(merge_gap_s * fs)
    for seg in segments:
        if not merged:
            merged.append(seg)
        else:
            ps, pe = merged[-1]
            s, e = seg
            if s - pe - 1 <= max_gap:
                merged[-1] = (ps, e)
            else:
                merged.append(seg)

    return merged


# ======================================================
# MAIN LOGIC
# ======================================================

def extract_activity_pickles(
    processed_root: str | Path,
    output_root: str | Path = "Activity_Files",
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
        segments = find_activity_periods(df, threshold=threshold)

        if not segments:
            continue

        csv_dict: Dict[str, pd.DataFrame] = {}
        for i, (s, e) in enumerate(segments):
            seg_id = f"seg{i:02d}"
            csv_dict[seg_id] = df.iloc[s : e + 1].copy()

        grouped.setdefault(pnum, {}).setdefault(test_name, {})[csv_path.stem] = csv_dict

    # --------------------------------------------------
    # Save one pickle per test folder
    # --------------------------------------------------
    for pnum, tests in grouped.items():
        for test_name, test_dict in tests.items():
            out_dir = output_root / f"Participant {pnum}" / test_name
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / f"{test_name}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(test_dict, f)

            print(f"Saved {out_path}")


# ======================================================
# Example usage
# ======================================================

if __name__ == "__main__":
    extract_activity_pickles(
        processed_root="Processed_CSVs",
        output_root="Activity_Files",
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
