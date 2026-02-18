from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from extract_data_multi_refactored import batch_extract_sensor_csvs
from preprocess_data_refactored import preprocess_participant_output_csvs
from detect_reaches import extract_reach_pickles, ExclusionRules
from batch_reach_metrics_from_reach_files import process_all_participants


def iter_participant_dirs(raw_data_root: Path) -> Iterable[Path]:
    """Yield participant directories under Raw_Data/*."""
    for p in sorted(raw_data_root.iterdir()):
        if p.is_dir():
            yield p


def run_pipeline(
    raw_data_root: str | Path = "Raw_Data",
    output_csv_root: str | Path = "Output_CSVs",
    processed_csv_root: str | Path = "Processed_CSVs",
    reach_files_root: str | Path = "Reach_Files",
    reach_metrics_root: str | Path = "Reach_Metrics",
    *,
    overwrite_extracted_csvs: bool = True,
    assume_time_is_epoch_us: bool = True,
):
    raw_data_root = Path(raw_data_root)
    output_csv_root = Path(output_csv_root)
    processed_csv_root = Path(processed_csv_root)
    reach_files_root = Path(reach_files_root)
    reach_metrics_root = Path(reach_metrics_root)

    if not raw_data_root.exists():
        raise FileNotFoundError(f"Raw data root not found: {raw_data_root.resolve()}")

    # ------------------------------------------------------------
    # 1) Extract sensor CSVs from .h5 files (per participant)
    # ------------------------------------------------------------
    for participant_dir in iter_participant_dirs(raw_data_root):
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
            exclude_test_substrings=["calibration"],
            exclude_sensor_ids={99999},
        ),
        threshold=0.25,
    )

    # ------------------------------------------------------------
    # 4) Compute metrics from reach files -> Reach_Metrics/*.csv
    #    (batch_reach_metrics_from_reach_files.py already loops Reach_Files)
    # ------------------------------------------------------------
    print("\n=== Computing reach metrics for ALL participants ===")
    process_all_participants(
        reach_files_root=reach_files_root,
        output_root=reach_metrics_root,
        exclude_participants=set(),
        movement_unit_signal="speed",
        movement_unit_min_prominence=0.05,
        detrend_accel_mean=True,
    )


if __name__ == "__main__":
    run_pipeline()
