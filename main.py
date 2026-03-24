from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

from src.extract_data_multi_refactored import batch_extract_sensor_csvs
from src.preprocess_data_refactored import preprocess_participant_output_csvs
from src.detect_reaches import extract_reach_pickles, ExclusionRules
from src.batch_reach_metrics_from_reach_files import process_all_participants


DETECTOR_MODES = ("fixed", "adaptive_baseline")


def iter_participant_dirs(raw_data_root: Path) -> Iterable[Path]:
    """Yield participant directories under Raw_Data/*."""
    for p in sorted(raw_data_root.iterdir()):
        if p.is_dir():
            yield p


def parse_participant_number(name: str) -> Optional[int]:
    match = re.search(r"Participant\s+(\d+)", name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def list_participant_dirs(raw_data_root: Path) -> list[Path]:
    return list(iter_participant_dirs(raw_data_root))


def prompt_for_project_root(initial_dir: Optional[Path] = None) -> Path:
    """
    Ask the user to choose the project folder that contains Raw_Data.

    Tries a GUI folder picker first, then falls back to console input.
    """
    initial_dir = (initial_dir or Path.cwd()).resolve()

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(
            title="Select the project folder containing Raw_Data",
            initialdir=str(initial_dir),
        )
        root.destroy()

        if selected:
            return Path(selected)
    except Exception:
        pass

    user_input = input(
        f"Enter the project folder path that contains Raw_Data "
        f"[default: {initial_dir}]: "
    ).strip()
    return Path(user_input) if user_input else initial_dir


def resolve_project_root(project_root: Optional[str | Path] = None) -> Path:
    """
    Resolve and validate the project folder used by the pipeline.
    """
    selected_root = Path(project_root) if project_root is not None else prompt_for_project_root()
    selected_root = selected_root.expanduser().resolve()
    raw_data_root = selected_root / "Raw_Data"

    if not raw_data_root.exists():
        raise FileNotFoundError(
            f"Selected project folder does not contain Raw_Data: {raw_data_root}"
        )

    return selected_root


def _parse_participant_selection(selection: str, participant_dirs: list[Path]) -> list[Path]:
    selection = selection.strip()
    if not selection or selection.lower() == "all":
        return participant_dirs

    by_index = {str(i): p for i, p in enumerate(participant_dirs, start=1)}
    by_name = {p.name.lower(): p for p in participant_dirs}

    selected: list[Path] = []
    seen: set[Path] = set()

    for token in [part.strip() for part in selection.split(",") if part.strip()]:
        participant_dir = by_index.get(token) or by_name.get(token.lower())
        if participant_dir is None:
            raise ValueError(
                f"Unrecognized participant selection '{token}'. "
                f"Use 'all', numbers like 1,2, or exact folder names."
            )
        if participant_dir not in seen:
            selected.append(participant_dir)
            seen.add(participant_dir)

    if not selected:
        raise ValueError("No participants were selected.")

    return selected


def prompt_for_participants(raw_data_root: Path) -> list[Path]:
    """
    Ask the user whether to run all participants or a selected subset.
    """
    participant_dirs = list_participant_dirs(raw_data_root)
    if not participant_dirs:
        raise FileNotFoundError(f"No participant folders found in {raw_data_root}")

    participant_lines = "\n".join(
        f"{i}. {participant_dir.name}"
        for i, participant_dir in enumerate(participant_dirs, start=1)
    )
    prompt_text = (
        "Select participants to process.\n"
        "Enter 'all' for every participant, or a comma-separated list of numbers "
        "or folder names.\n\n"
        f"{participant_lines}"
    )

    try:
        import tkinter as tk
        from tkinter import simpledialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selection = simpledialog.askstring(
            "Select Participants",
            prompt_text,
            initialvalue="all",
            parent=root,
        )
        root.destroy()

        if selection is None:
            return participant_dirs

        return _parse_participant_selection(selection, participant_dirs)
    except Exception:
        print(prompt_text)
        selection = input("Participants to process [all]: ").strip() or "all"
        return _parse_participant_selection(selection, participant_dirs)


def resolve_detector_mode(detector_mode: Optional[str]) -> str:
    if detector_mode is not None:
        mode = detector_mode.strip().lower()
        if mode not in DETECTOR_MODES:
            raise ValueError(
                f"Unsupported detector mode '{detector_mode}'. "
                f"Choose from: {', '.join(DETECTOR_MODES)}"
            )
        return mode

    prompt_text = (
        "Select reach detector mode:\n"
        "1. fixed\n"
        "2. adaptive_baseline"
    )

    try:
        import tkinter as tk
        from tkinter import simpledialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selection = simpledialog.askstring(
            "Reach Detector Mode",
            prompt_text,
            initialvalue="fixed",
            parent=root,
        )
        root.destroy()
        mode = (selection or "fixed").strip().lower()
    except Exception:
        print(prompt_text)
        mode = (input("Detector mode [fixed]: ").strip() or "fixed").lower()

    if mode not in DETECTOR_MODES:
        raise ValueError(
            f"Unsupported detector mode '{mode}'. "
            f"Choose from: {', '.join(DETECTOR_MODES)}"
        )
    return mode


def parse_sensor_id_list(sensor_text: Optional[str]) -> Optional[set[int]]:
    if sensor_text is None:
        return None

    tokens = [part.strip() for part in sensor_text.split(",") if part.strip()]
    if not tokens:
        return set()

    try:
        return {int(token) for token in tokens}
    except ValueError as exc:
        raise ValueError(
            "Sensor lists must be comma-separated integers like '17738,21263'."
        ) from exc


def prompt_for_sensor_filters() -> tuple[Optional[set[int]], Optional[set[int]]]:
    """
    Ask the user which sensors to include or exclude for reach detection.

    The user can leave both prompts blank to use all sensors.
    """
    include_prompt = (
        "Sensor include list:\n"
        "Leave blank to include all sensors.\n"
        "Or enter comma-separated sensor IDs like:\n"
        "17738,21263"
    )
    exclude_prompt = (
        "Sensor exclude list:\n"
        "Leave blank to exclude none.\n"
        "Or enter comma-separated sensor IDs like:\n"
        "17794,21146"
    )

    try:
        import tkinter as tk
        from tkinter import simpledialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        include_text = simpledialog.askstring(
            "Include Sensors",
            include_prompt,
            initialvalue="",
            parent=root,
        )
        exclude_text = simpledialog.askstring(
            "Exclude Sensors",
            exclude_prompt,
            initialvalue="",
            parent=root,
        )
        root.destroy()
    except Exception:
        print(include_prompt)
        include_text = input("Sensors to include [all]: ").strip()
        print(exclude_prompt)
        exclude_text = input("Sensors to exclude [none]: ").strip()

    include_sensor_ids = parse_sensor_id_list(include_text) if include_text else None
    exclude_sensor_ids = parse_sensor_id_list(exclude_text) if exclude_text else None

    if include_sensor_ids and exclude_sensor_ids:
        overlap = include_sensor_ids & exclude_sensor_ids
        if overlap:
            raise ValueError(
                "The same sensor ID cannot be both included and excluded: "
                + ", ".join(str(sid) for sid in sorted(overlap))
            )

    return include_sensor_ids, exclude_sensor_ids


def run_pipeline(
    project_root: str | Path,
    *,
    selected_participants: Optional[list[str]] = None,
    detector_mode: str = "fixed",
    include_sensor_ids: Optional[set[int]] = None,
    exclude_sensor_ids: Optional[set[int]] = None,
    overwrite_extracted_csvs: bool = True,
    assume_time_is_epoch_us: bool = True,
):
    project_root = Path(project_root).resolve()
    raw_data_root = project_root / "Raw_Data"
    output_csv_root = project_root / "Output_CSVs"
    processed_csv_root = project_root / "Processed_CSVs"
    reach_files_root = project_root / "Reach_Files"
    reach_metrics_root = project_root / "Reach_Metrics"

    if not raw_data_root.exists():
        raise FileNotFoundError(f"Raw data root not found: {raw_data_root}")

    print(f"Using project folder: {project_root}")

    all_participant_dirs = list_participant_dirs(raw_data_root)
    selected_set = (
        set(selected_participants)
        if selected_participants is not None
        else {participant_dir.name for participant_dir in all_participant_dirs}
    )
    participant_dirs = [
        participant_dir
        for participant_dir in all_participant_dirs
        if participant_dir.name in selected_set
    ]

    if not participant_dirs:
        raise FileNotFoundError("No selected participant folders were found in Raw_Data.")

    excluded_participant_names = {
        participant_dir.name
        for participant_dir in all_participant_dirs
        if participant_dir.name not in selected_set
    }
    excluded_participant_nums = {
        pnum
        for pnum in (
            parse_participant_number(participant_name)
            for participant_name in excluded_participant_names
        )
        if pnum is not None
    }

    print("Participants to process: " + ", ".join(p.name for p in participant_dirs))
    print(f"Reach detector mode: {detector_mode}")
    if include_sensor_ids:
        print("Included sensors: " + ", ".join(str(sid) for sid in sorted(include_sensor_ids)))
    if exclude_sensor_ids:
        print("Excluded sensors: " + ", ".join(str(sid) for sid in sorted(exclude_sensor_ids)))

    # ------------------------------------------------------------
    # 1) Extract sensor CSVs from .h5 files (per participant)
    # ------------------------------------------------------------
    for participant_dir in participant_dirs:
        participant_name = participant_dir.name
        raw_h5_dir = participant_dir / "rawData"
        if not raw_h5_dir.exists():
            print(f"[SKIP] {participant_name}: missing {raw_h5_dir}")
            continue

        out_csv_dir = output_csv_root / participant_name
        out_csv_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Extracting .h5 -> CSVs for {participant_name} ===")
        batch_extract_sensor_csvs(
            input_path=raw_h5_dir,
            output_dir=out_csv_dir,
            overwrite=overwrite_extracted_csvs,
        )

        # --------------------------------------------------------
        # 2) Preprocess extracted CSVs (filters, epoch->elapsed, etc.)
        # --------------------------------------------------------
        print(f"\n=== Preprocessing CSVs for {participant_name} ===")
        preprocess_participant_output_csvs(
            participant_output_csv_root=out_csv_dir,
            participant_processed_root=processed_csv_root / participant_name,
            assume_time_is_epoch_us=assume_time_is_epoch_us,
        )

    # ------------------------------------------------------------
    # 3) Detect reaches -> Reach_Files/*.pkl
    #    (detect_reaches.py already loops through Processed_CSVs)
    # ------------------------------------------------------------
    print("\n=== Detecting reaches for ALL participants ===")
    extract_reach_pickles(
        processed_root=processed_csv_root,
        output_root=reach_files_root,
        exclusions=ExclusionRules(
            exclude_participant_nums=excluded_participant_nums,
            exclude_test_substrings=["calibration"],
            include_sensor_ids=include_sensor_ids or set(),
            exclude_sensor_ids=(exclude_sensor_ids or set()) | {99999},
        ),
        threshold=0.25,
        adaptive_thresholds=(detector_mode == "adaptive_baseline"),
        verbose_thresholds=(detector_mode == "adaptive_baseline"),
    )

    # ------------------------------------------------------------
    # 4) Compute metrics from reach files -> Reach_Metrics/*.csv
    #    (batch_reach_metrics_from_reach_files.py already loops Reach_Files)
    # ------------------------------------------------------------
    print("\n=== Computing reach metrics for ALL participants ===")
    process_all_participants(
        reach_files_root=reach_files_root,
        output_root=reach_metrics_root,
        exclude_participants=excluded_participant_names,
        movement_unit_signal="speed",
        movement_unit_min_prominence=0.05,
        detrend_accel_mean=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the HDF5 processing pipeline from a selected project folder."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Path to the project folder containing Raw_Data.",
    )
    parser.add_argument(
        "--participants",
        type=str,
        default=None,
        help="Participants to process: 'all' or a comma-separated list of numbers or folder names.",
    )
    parser.add_argument(
        "--detector-mode",
        type=str,
        default=None,
        help="Reach detector mode: 'fixed' or 'adaptive_baseline'.",
    )
    parser.add_argument(
        "--sensor-include",
        type=str,
        default=None,
        help="Comma-separated sensor IDs to include, for example '17738,21263'.",
    )
    parser.add_argument(
        "--sensor-exclude",
        type=str,
        default=None,
        help="Comma-separated sensor IDs to exclude, for example '17738,21263'.",
    )
    args = parser.parse_args()

    project_root = resolve_project_root(args.project_root)
    raw_data_root = project_root / "Raw_Data"
    detector_mode = resolve_detector_mode(args.detector_mode)
    if args.sensor_include is None and args.sensor_exclude is None:
        include_sensor_ids, exclude_sensor_ids = prompt_for_sensor_filters()
    else:
        include_sensor_ids = parse_sensor_id_list(args.sensor_include)
        exclude_sensor_ids = parse_sensor_id_list(args.sensor_exclude)
        if include_sensor_ids and exclude_sensor_ids:
            overlap = include_sensor_ids & exclude_sensor_ids
            if overlap:
                raise ValueError(
                    "The same sensor ID cannot be both included and excluded: "
                    + ", ".join(str(sid) for sid in sorted(overlap))
                )
    selected_participant_dirs = (
        _parse_participant_selection(args.participants, list_participant_dirs(raw_data_root))
        if args.participants is not None
        else prompt_for_participants(raw_data_root)
    )
    run_pipeline(
        project_root=project_root,
        selected_participants=[p.name for p in selected_participant_dirs],
        detector_mode=detector_mode,
        include_sensor_ids=include_sensor_ids,
        exclude_sensor_ids=exclude_sensor_ids,
        overwrite_extracted_csvs=True,
        assume_time_is_epoch_us=True,
    )


if __name__ == "__main__":
    main()
