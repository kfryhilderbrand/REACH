from __future__ import annotations

import re
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Any

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
    include_sensor_ids: set[int] = field(default_factory=set)
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
        if self.include_sensor_ids:
            if sensor_id is None or sensor_id not in self.include_sensor_ids:
                return True
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


def _start_mode(
    accel_value: float,
    gyro_value: Optional[float],
    *,
    accel_start_thresh: float,
    gyro_start_thresh: float,
) -> str:
    accel_started = accel_value >= accel_start_thresh
    gyro_started = gyro_value is not None and gyro_value >= gyro_start_thresh

    if accel_started and gyro_started:
        return "both"
    if accel_started:
        return "accel"
    if gyro_started:
        return "gyro"
    return "none"


@dataclass
class ReachDetectionParams:
    use_gyro: bool = True
    accel_start_thresh: float = 1.2
    accel_end_thresh: float = 0.6
    gyro_start_thresh: float = 1.2
    gyro_end_thresh: float = 0.6
    smooth_window_s: float = 0.03
    end_hold_s: float = 0.05
    min_duration_s: float = 0.12
    require_peak: bool = True
    peak_prominence: float = 0.02
    merge_gap_s: float = 0.02

    def as_kwargs(self) -> Dict[str, Any]:
        return {
            "use_gyro": self.use_gyro,
            "accel_start_thresh": self.accel_start_thresh,
            "accel_end_thresh": self.accel_end_thresh,
            "gyro_start_thresh": self.gyro_start_thresh,
            "gyro_end_thresh": self.gyro_end_thresh,
            "smooth_window_s": self.smooth_window_s,
            "end_hold_s": self.end_hold_s,
            "min_duration_s": self.min_duration_s,
            "require_peak": self.require_peak,
            "peak_prominence": self.peak_prominence,
            "merge_gap_s": self.merge_gap_s,
        }


DEFAULT_REACH_PARAMS = ReachDetectionParams()


@dataclass
class SensorThresholdSummary:
    participant_num: int
    sensor_id: int
    file_count: int
    accel_baseline: float
    accel_start_offset: float
    accel_end_offset: float
    gyro_baseline: float
    gyro_start_offset: float
    gyro_end_offset: float


def _safe_quantile(x: np.ndarray, q: float) -> float:
    if len(x) == 0:
        return float("nan")
    return float(np.quantile(x, q))


def _robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def _sensor_signal_stats(df: pd.DataFrame) -> Optional[dict[str, float]]:
    accel_cols = ["accel_0_lin", "accel_1_lin", "accel_2_lin"]
    gyro_cols = ["gyro_0_corr", "gyro_1_corr", "gyro_2_corr"]
    if not all(c in df.columns for c in accel_cols + gyro_cols):
        return None

    accel_mag = magnitude_from_cols(df, accel_cols)
    gyro_mag = magnitude_from_cols(df, gyro_cols)

    return {
        "accel_q50": _safe_quantile(accel_mag, 0.50),
        "accel_q75": _safe_quantile(accel_mag, 0.75),
        "accel_q90": _safe_quantile(accel_mag, 0.90),
        "accel_q99": _safe_quantile(accel_mag, 0.99),
        "gyro_q50": _safe_quantile(gyro_mag, 0.50),
        "gyro_q75": _safe_quantile(gyro_mag, 0.75),
        "gyro_q90": _safe_quantile(gyro_mag, 0.90),
        "gyro_q99": _safe_quantile(gyro_mag, 0.99),
    }


def _build_sensor_threshold_summary(
    sensor_id: int,
    stats_by_file: list[dict[str, float]],
) -> SensorThresholdSummary:
    accel_baseline = float(np.median([s["accel_q50"] for s in stats_by_file]))
    gyro_baseline = float(np.median([s["gyro_q50"] for s in stats_by_file]))

    accel_p75_offsets = np.asarray([s["accel_q75"] - s["accel_q50"] for s in stats_by_file], dtype=float)
    accel_p95_offsets = np.asarray([s["accel_q90"] - s["accel_q50"] for s in stats_by_file], dtype=float)
    gyro_p75_offsets = np.asarray([s["gyro_q75"] - s["gyro_q50"] for s in stats_by_file], dtype=float)
    gyro_p95_offsets = np.asarray([s["gyro_q90"] - s["gyro_q50"] for s in stats_by_file], dtype=float)

    accel_start_offset = max(0.08, float(np.quantile(accel_p95_offsets, 0.25)) * 0.55)
    accel_end_offset = max(0.03, float(np.quantile(accel_p75_offsets, 0.25)) * 0.70)
    gyro_start_offset = max(0.02, float(np.quantile(gyro_p95_offsets, 0.25)) * 0.55)
    gyro_end_offset = max(0.01, float(np.quantile(gyro_p75_offsets, 0.25)) * 0.70)

    if accel_end_offset >= accel_start_offset:
        accel_end_offset = max(0.03, 0.6 * accel_start_offset)
    if gyro_end_offset >= gyro_start_offset:
        gyro_end_offset = max(0.01, 0.6 * gyro_start_offset)

    return SensorThresholdSummary(
        participant_num=-1,
        sensor_id=sensor_id,
        file_count=len(stats_by_file),
        accel_baseline=accel_baseline,
        accel_start_offset=accel_start_offset,
        accel_end_offset=accel_end_offset,
        gyro_baseline=gyro_baseline,
        gyro_start_offset=gyro_start_offset,
        gyro_end_offset=gyro_end_offset,
    )


def learn_sensor_thresholds(
    processed_root: str | Path,
    exclusions: ExclusionRules | None = None,
    *,
    min_files_per_sensor: int = 2,
) -> dict[tuple[int, int], SensorThresholdSummary]:
    processed_root = Path(processed_root)
    exclusions = exclusions or ExclusionRules()

    stats_by_sensor: dict[tuple[int, int], list[dict[str, float]]] = {}

    for pnum, _test_name, csv_path in iter_csv_files(processed_root, exclusions):
        sensor_id = parse_sensor_id(csv_path.name)
        if sensor_id is None:
            continue
        df = pd.read_csv(csv_path)
        stats = _sensor_signal_stats(df)
        if stats is None:
            continue
        stats_by_sensor.setdefault((pnum, sensor_id), []).append(stats)

    learned: dict[tuple[int, int], SensorThresholdSummary] = {}
    for (pnum, sensor_id), sensor_stats in stats_by_sensor.items():
        if len(sensor_stats) < min_files_per_sensor:
            continue

        summary = _build_sensor_threshold_summary(sensor_id, sensor_stats)
        summary.participant_num = pnum
        learned[(pnum, sensor_id)] = summary

    return learned


def summarize_sensor_thresholds(thresholds: dict[tuple[int, int], SensorThresholdSummary]) -> pd.DataFrame:
    rows = []
    for key in sorted(thresholds):
        summary = thresholds[key]
        rows.append({
            "participant_num": summary.participant_num,
            "sensor_id": summary.sensor_id,
            "file_count": summary.file_count,
            "accel_baseline": summary.accel_baseline,
            "accel_start_offset": summary.accel_start_offset,
            "accel_end_offset": summary.accel_end_offset,
            "gyro_baseline": summary.gyro_baseline,
            "gyro_start_offset": summary.gyro_start_offset,
            "gyro_end_offset": summary.gyro_end_offset,
        })
    return pd.DataFrame(rows)


def params_for_file(
    df: pd.DataFrame,
    summary: Optional[SensorThresholdSummary],
) -> ReachDetectionParams:
    if summary is None:
        return DEFAULT_REACH_PARAMS

    stats = _sensor_signal_stats(df)
    if stats is None:
        return DEFAULT_REACH_PARAMS

    accel_start = stats["accel_q50"] + summary.accel_start_offset
    accel_end = stats["accel_q50"] + summary.accel_end_offset
    gyro_start = stats["gyro_q50"] + summary.gyro_start_offset
    gyro_end = stats["gyro_q50"] + summary.gyro_end_offset

    if accel_end >= accel_start:
        accel_end = stats["accel_q50"] + max(0.03, 0.6 * summary.accel_start_offset)
    if gyro_end >= gyro_start:
        gyro_end = stats["gyro_q50"] + max(0.01, 0.6 * summary.gyro_start_offset)

    return ReachDetectionParams(
        use_gyro=True,
        accel_start_thresh=float(accel_start),
        accel_end_thresh=float(accel_end),
        gyro_start_thresh=float(gyro_start),
        gyro_end_thresh=float(gyro_end),
        smooth_window_s=0.03,
        end_hold_s=0.05,
        min_duration_s=0.10,
        require_peak=False,
        peak_prominence=0.02,
        merge_gap_s=0.02,
    )


def adaptive_params_for_df(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    baseline_window_s: float = 3.0,
    start_sigma: float = 6.0,
    end_sigma: float = 3.0,
    peak_sigma: float = 4.0,
    min_accel_start: float = 0.08,
    min_accel_end: float = 0.03,
    min_gyro_start: float = 0.02,
    min_gyro_end: float = 0.01,
) -> ReachDetectionParams:
    """
    Estimate conservative per-file thresholds from the initial quiet baseline.

    Assumes the first ~3 seconds are mostly still, which matches the trial setup.
    """
    accel_cols = ["accel_0_lin", "accel_1_lin", "accel_2_lin"]
    gyro_cols = ["gyro_0_corr", "gyro_1_corr", "gyro_2_corr"]
    if not all(c in df.columns for c in accel_cols + gyro_cols + [time_col]):
        return DEFAULT_REACH_PARAMS

    t = df[time_col].to_numpy(dtype=float)
    if len(t) < 5:
        return DEFAULT_REACH_PARAMS

    accel_mag = magnitude_from_cols(df, accel_cols)
    gyro_mag = magnitude_from_cols(df, gyro_cols)

    baseline_mask = t <= (float(t[0]) + baseline_window_s)
    if int(np.sum(baseline_mask)) < 20:
        baseline_n = min(len(df), 200)
        baseline_mask = np.zeros(len(df), dtype=bool)
        baseline_mask[:baseline_n] = True

    a_base = accel_mag[baseline_mask]
    g_base = gyro_mag[baseline_mask]

    a_med = float(np.median(a_base))
    g_med = float(np.median(g_base))
    a_sigma = _robust_sigma(a_base)
    g_sigma = _robust_sigma(g_base)

    accel_start = max(min_accel_start, a_med + start_sigma * max(a_sigma, 1e-6))
    accel_end = max(min_accel_end, a_med + end_sigma * max(a_sigma, 1e-6))
    gyro_start = max(min_gyro_start, g_med + start_sigma * max(g_sigma, 1e-6))
    gyro_end = max(min_gyro_end, g_med + end_sigma * max(g_sigma, 1e-6))

    # Keep end thresholds meaningfully below start thresholds.
    accel_end = min(accel_end, accel_start * 0.7)
    gyro_end = min(gyro_end, gyro_start * 0.7)

    peak_prominence = max(
        0.02,
        peak_sigma * max(a_sigma, 1e-6),
    )

    return ReachDetectionParams(
        use_gyro=True,
        accel_start_thresh=float(accel_start),
        accel_end_thresh=float(accel_end),
        gyro_start_thresh=float(gyro_start),
        gyro_end_thresh=float(gyro_end),
        smooth_window_s=0.025,
        end_hold_s=0.035,
        min_duration_s=0.08,
        require_peak=True,
        peak_prominence=float(peak_prominence),
        merge_gap_s=0.06,
    )

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

    # Define "active" using hysteresis with separate start/end thresholds.
    # Start if accel OR gyro exceeds start thresholds.
    # End follows whichever signal actually triggered the reach; only dual-triggered
    # starts require both signals to settle.
    def is_start(i: int) -> bool:
        if g is None:
            return a_s[i] >= accel_start_thresh
        return (a_s[i] >= accel_start_thresh) or (g_s[i] >= gyro_start_thresh)

    def is_end_condition(i: int, start_mode: str) -> bool:
        accel_below = a_s[i] <= accel_end_thresh
        if g is None:
            return accel_below

        gyro_below = g_s[i] <= gyro_end_thresh
        if start_mode == "accel":
            return accel_below
        if start_mode == "gyro":
            return gyro_below
        return accel_below and gyro_below

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
        start_mode = _start_mode(
            a_s[start],
            g_s[start] if g is not None else None,
            accel_start_thresh=accel_start_thresh,
            gyro_start_thresh=gyro_start_thresh,
        )
        if start_mode == "none":
            i += 1
            continue

        # Now advance until we see end condition held for end_hold samples
        i += 1
        end = None
        hold = 0
        while i < n:
            if is_end_condition(i, start_mode):
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
    adaptive_thresholds: bool = False,
    verbose_thresholds: bool = True,
):
    processed_root = Path(processed_root)
    output_root = Path(output_root)
    exclusions = exclusions or ExclusionRules()
    sensor_thresholds = {}

    # Structure:
    # participant -> test -> { csv_stem -> { seg_id -> df } }
    grouped: Dict[int, Dict[str, Dict[str, Dict[str, pd.DataFrame]]]] = {}

    for pnum, test_name, csv_path in iter_csv_files(processed_root, exclusions):
        df = pd.read_csv(csv_path)
        sensor_id = parse_sensor_id(csv_path.name)
        #segments = find_activity_periods(df, threshold=threshold)
        #reaches = find_reaches(df, accel_start_thresh=2, accel_end_thresh=1, use_gyro=True)
        if adaptive_thresholds:
            reach_params = adaptive_params_for_df(df)
            if verbose_thresholds:
                print(
                    f"Adaptive thresholds for {csv_path.name}: "
                    f"accel_start={reach_params.accel_start_thresh:.3f}, "
                    f"accel_end={reach_params.accel_end_thresh:.3f}, "
                    f"gyro_start={reach_params.gyro_start_thresh:.3f}, "
                    f"gyro_end={reach_params.gyro_end_thresh:.3f}, "
                    f"peak_prom={reach_params.peak_prominence:.3f}"
                )
        else:
            summary = sensor_thresholds.get((pnum, sensor_id))
            reach_params = params_for_file(df, summary)

        reaches = find_reaches(df, **reach_params.as_kwargs())


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
