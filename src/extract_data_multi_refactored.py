"""
Refactor note:
- Original script had a hard-coded Windows path in main() (Participant 1) and called
  batch_extract_sensor_csvs() directly.
- The core extraction functions are unchanged; we only add a small helper you can import
  from main.py so you can run per-participant without editing this file again.
"""

import h5py
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union


def run_for_participant(raw_data_dir: Union[str, Path],
                        output_csv_participant_dir: Union[str, Path],
                        *,
                        overwrite: bool = True) -> None:
    """
    Convenience wrapper used by main.py.
    raw_data_dir: Raw_Data/<participant>/rawData
    output_csv_participant_dir: Output_CSVs/<participant>
    """
    batch_extract_sensor_csvs(
        input_path=raw_data_dir,
        output_dir=output_csv_participant_dir,
        overwrite=overwrite,
    )


def main():
    # Keep this as a small example, but main.py is the recommended entrypoint now.
    batch_extract_sensor_csvs(
        input_path="Raw_Data/Participant 1/rawData",
        output_dir="Output_CSVs/Participant 1",
        overwrite=True,
    )


# ==============================
# Utility: find .h5 files
# ==============================
def collect_h5_files(path: Union[str, Path]) -> List[Path]:
    """Return a list of .h5/.hdf5 files from a path (file or directory)."""
    path = Path(path)
    if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}:
        return [path]
    elif path.is_dir():
        return sorted(path.glob("*.h5")) + sorted(path.glob("*.hdf5"))
    else:
        raise FileNotFoundError(f"No .h5 files found at {path}")


def print_h5_structure(h5_path: Union[str, Path]) -> None:
    """Print the groups and datasets inside an HDF5 file (for debugging)."""
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        print(f"Structure of {h5_path}:")
        def _visitor(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"[GROUP]   {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"[DATASET] {name} -> shape={obj.shape}, dtype={obj.dtype}")
        f.visititems(_visitor)


def is_sensor_group(name: str, obj: h5py.Group) -> bool:
    """
    Decide if a group is a 'sensor group'.
    Default logic: group directly under /Sensors whose last path component is all
    digits (e.g. /Sensors/19392).
    """
    parts = name.strip("/").split("/")
    return len(parts) == 2 and parts[0] == "Sensors" and parts[1].isdigit()


def find_sensor_groups(f: h5py.File) -> List[h5py.Group]:
    """Find all groups in the file that look like sensor groups."""
    sensor_groups = []

    def _visitor(name, obj):
        if isinstance(obj, h5py.Group) and is_sensor_group(name, obj):
            sensor_groups.append(obj)

    f.visititems(_visitor)
    return sensor_groups


def categorize_datasets(
    sensor_group: h5py.Group,
    time_keywords=("time", "timestamp"),
    accel_keywords=("acc", "accelerometer"),
    gyro_keywords=("gyro", "gyroscope"),
) -> Dict[str, List[h5py.Dataset]]:
    """
    Within a sensor group, find datasets for time, accelerometer, and gyroscope
    using simple substring matching on dataset names.
    """
    time_ds = []
    accel_ds = []
    gyro_ds = []

    def _visitor(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        if "/" in name:
            return
        full_name = obj.name.lower()

        if any(k in full_name for k in time_keywords):
            time_ds.append(obj)
        if any(k in full_name for k in accel_keywords):
            accel_ds.append(obj)
        if any(k in full_name for k in gyro_keywords):
            gyro_ds.append(obj)

    sensor_group.visititems(_visitor)

    return {
        "time": time_ds,
        "accel": accel_ds,
        "gyro": gyro_ds,
    }


def choose_time_dataset(datasets: List[h5py.Dataset]) -> h5py.Dataset:
    """Choose a time dataset (first one by default, error if none)."""
    if not datasets:
        raise ValueError("No time datasets found for this sensor.")
    if len(datasets) > 1:
        print(f"Warning: multiple time datasets found, using {datasets[0].name}")
    return datasets[0]


def dataset_to_columns(ds: h5py.Dataset, prefix: Optional[str] = None) -> Dict[str, pd.Series]:
    """
    Convert a dataset into one or more columns:
    - 1D: single column
    - 2D: one column per second dimension (e.g. x, y, z -> col_0, col_1, col_2)
    """
    data = ds[()]  # load into memory
    base_name = ds.name.split("/")[-1]
    if prefix:
        base_name = f"{prefix}_{base_name}"

    columns = {}

    if data.ndim == 1:
        columns[base_name] = pd.Series(data)
    elif data.ndim == 2:
        n_rows, n_cols = data.shape
        for i in range(n_cols):
            col_name = f"{base_name}_{i}"
            columns[col_name] = pd.Series(data[:, i])
    else:
        raise ValueError(f"Dataset {ds.name} has unsupported ndim={data.ndim}")

    return columns


def add_orientation_columns(
    df: pd.DataFrame,
    h5_file: h5py.File,
    sensor_id: str,
) -> None:
    """
    Add orientation quaternion columns from /Processed/<sensor_id>/Orientation when
    available and aligned to the sensor time series.
    """
    orientation_path = f"Processed/{sensor_id}/Orientation"
    if orientation_path not in h5_file:
        return

    orientation_ds = h5_file[orientation_path]
    orientation = orientation_ds[()]

    if orientation.ndim != 2 or orientation.shape[1] != 4:
        print(
            f"Skipping {orientation_ds.name}: expected shape (n, 4), "
            f"got {orientation.shape}"
        )
        return

    if orientation.shape[0] != len(df):
        print(
            f"Skipping {orientation_ds.name}: length {orientation.shape[0]} "
            f"!= time length {len(df)}"
        )
        return

    for i in range(4):
        df[f"orientation_{i}"] = orientation[:, i]


def build_sensor_dataframe(
    sensor_group: h5py.Group,
    time_keywords=("time", "timestamp"),
    accel_keywords=("acc", "accelerometer"),
    gyro_keywords=("gyro", "gyroscope"),
) -> pd.DataFrame:
    categorized = categorize_datasets(
        sensor_group,
        time_keywords=time_keywords,
        accel_keywords=accel_keywords,
        gyro_keywords=gyro_keywords,
    )

    time_ds = choose_time_dataset(categorized["time"])
    time_data = time_ds[()]
    df = pd.DataFrame({"time": time_data})

    def add_datasets(ds_list: List[h5py.Dataset], prefix: str):
        for ds in ds_list:
            data = ds[()]
            if data.shape[0] != len(df):
                print(
                    f"Skipping {ds.name}: length {data.shape[0]} "
                    f"!= time length {len(df)}"
                )
                continue
            cols = dataset_to_columns(ds, prefix=prefix)
            for col_name, series in cols.items():
                if len(series) != len(df):
                    print(
                        f"Skipping column {col_name} from {ds.name}: "
                        f"length {len(series)} != time length {len(df)}"
                    )
                    continue
                df[col_name] = series.values

    add_datasets(categorized["accel"], prefix="accel")
    add_datasets(categorized["gyro"], prefix="gyro")

    return df


def extract_sensor_csvs_from_h5(
    h5_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    time_keywords=("time", "timestamp"),
    accel_keywords=("acc", "accelerometer"),
    gyro_keywords=("gyro", "gyroscope"),
):
    h5_path = Path(h5_path)
    if output_dir is None:
        output_dir = h5_path.parent
    output_dir = Path(output_dir) / h5_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        sensor_groups = find_sensor_groups(f)

        if not sensor_groups:
            print(f"No sensor groups found in {h5_path}")
            return

        for sensor_group in sensor_groups:
            sensor_id = sensor_group.name.strip("/").split("/")[-1]
            try:
                df = build_sensor_dataframe(
                    sensor_group,
                    time_keywords=time_keywords,
                    accel_keywords=accel_keywords,
                    gyro_keywords=gyro_keywords,
                )
            except ValueError as e:
                print(f"Skipping sensor {sensor_id} in {h5_path.name}: {e}")
                continue

            add_orientation_columns(df, f, sensor_id)

            out_name = f"{h5_path.stem}_sensor-{sensor_id}.csv"
            out_path = output_dir / out_name

            if out_path.exists() and not overwrite:
                print(f"Skipping {out_name}: already exists (use overwrite=True).")
                continue

            df.to_csv(out_path, index=False)
            print(f"Wrote {out_path}")


def batch_extract_sensor_csvs(
    input_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    time_keywords=("time", "timestamp"),
    accel_keywords=("acc", "accelerometer"),
    gyro_keywords=("gyro", "gyroscope"),
):
    files = collect_h5_files(input_path)
    if not files:
        print(f"No .h5 files found under {input_path}")
        return

    for h5_file in files:
        print(f"Processing {h5_file} ...")
        extract_sensor_csvs_from_h5(
            h5_file,
            output_dir=output_dir,
            overwrite=overwrite,
            time_keywords=time_keywords,
            accel_keywords=accel_keywords,
            gyro_keywords=gyro_keywords,
        )


if __name__ == "__main__":
    main()
