# HDF5 Processing

This repository takes wearable sensor data stored in `.h5` files and turns it into easier-to-use CSV files and reach summaries.

In simple terms, it does this:

1. Reads raw sensor recordings from `Raw_Data`
2. Exports the raw accelerometer and gyroscope data for each sensor to CSV files
3. Preprocesses those CSV files so they are easier to analyze
4. Detects reaches from the processed motion data
5. Calculates reach metrics and saves them to summary CSV files

## What This Repo Is For

This project is designed for data collected from multiple sensors during movement trials.

The pipeline helps you go from:

- raw `.h5` sensor recordings

to:

- per-sensor raw CSV files
- per-sensor processed CSV files
- detected reach segments
- reach metric summary files

You do not need to manually edit the code each time you run it. The main script can prompt you for the project folder and which participants you want to process.

## Typical Folder Setup

Your project folder should contain a `Raw_Data` folder.

Example:

```text
Project/
  Raw_Data/
    Participant 1/
      rawData/
        20251017-090642_Free_Form.h5
        20251017-090827_Free_Form.h5
      SubjectMetadata.xml
      Free Form_trials.csv
    Participant 2/
      rawData/
        ...
```

The pipeline will create the output folders for you inside that same project folder.

## What The User Will Be Prompted To Provide

When you run the main pipeline, the program may prompt you for:

1. The project folder
   This is the folder that contains `Raw_Data`.

2. Which participants to process
   You can choose:
   - `all`
   - a few participants by number, like `1,2`
   - exact folder names, like `Participant 1,Participant 2`

3. The reach detector mode
   Current choices are:
   - `fixed`
   - `adaptive_baseline`

4. Which sensors to include
   You can:
   - leave this blank to include all sensors
   - enter a comma-separated list such as `17738,21263`

5. Which sensors to exclude
   You can:
   - leave this blank to exclude no sensors
   - enter a comma-separated list such as `17794,21146`

If you prefer, you can also provide these from the command line instead of answering prompts.

## Main Script

The main entry point is:

- [main.py](/c:/Users/katel/Desktop/HDF5%20Processing/main.py)

To run it:

```powershell
python main.py
```

You can also run it with options:

```powershell
python main.py --project-root "C:\path\to\Project"
```

```powershell
python main.py --project-root "C:\path\to\Project" --participants "all" --detector-mode fixed
```

```powershell
python main.py --project-root "C:\path\to\Project" --participants "Participant 1" --sensor-exclude "17794,21146"
```

## What Files Are Generated

The pipeline creates several folders inside your selected project folder.

### 1. `Output_CSVs`

This contains raw sensor data exported from the `.h5` files.

Example:

```text
Output_CSVs/
  Participant 1/
    20251017-090642_Free_Form/
      20251017-090642_Free_Form_sensor-17738.csv
      20251017-090642_Free_Form_sensor-17794.csv
```

These files usually contain:

- `time`
- accelerometer columns
- gyroscope columns
- orientation columns when available

### 2. `Processed_CSVs`

This contains cleaned and preprocessed sensor data.

Example:

```text
Processed_CSVs/
  Participant 1/
    20251017-090642_Free_Form/
      20251017-090642_Free_Form_sensor-17738_preprocessed.csv
```

These files usually contain:

- `time`
- filtered acceleration
- gravity-compensated linear acceleration
- filtered gyroscope
- corrected gyroscope

### 3. `Reach_Files`

This contains detected reach segments saved as `.pkl` files.

Example:

```text
Reach_Files/
  Participant 1/
    20251017-090642_Free_Form.pkl
```

These files are intermediate analysis files used by the pipeline.

### 4. `Reach_Metrics`

This contains the final summary CSV files for detected reaches.

Example:

```text
Reach_Metrics/
  Participant 1/
    20251017-090642_Free_Form/
      reach_metrics_sensor-17738.csv
      reach_metrics_sensor-21146.csv
```

These files contain one row per detected reach and include measurements such as:

- reach duration
- peak speed
- average speed
- path length
- movement units
- jerk metrics

## What Each Script Does

- [main.py](/c:/Users/katel/Desktop/HDF5%20Processing/main.py)
  Runs the full pipeline and prompts the user for input.

- [extract_data_multi_refactored.py](/c:/Users/katel/Desktop/HDF5%20Processing/extract_data_multi_refactored.py)
  Reads `.h5` files and exports raw per-sensor CSV files.

- [preprocess_data_refactored.py](/c:/Users/katel/Desktop/HDF5%20Processing/preprocess_data_refactored.py)
  Converts raw sensor CSV files into processed CSV files for analysis.

- [detect_reaches.py](/c:/Users/katel/Desktop/HDF5%20Processing/detect_reaches.py)
  Finds candidate reaches in the processed motion data.

- [reach_metrics.py](/c:/Users/katel/Desktop/HDF5%20Processing/reach_metrics.py)
  Computes measurements for each detected reach.

- [batch_reach_metrics_from_reach_files.py](/c:/Users/katel/Desktop/HDF5%20Processing/batch_reach_metrics_from_reach_files.py)
  Turns detected reach files into summary CSVs.

- [plot_random_raw_vs_processed.py](/c:/Users/katel/Desktop/HDF5%20Processing/plot_random_raw_vs_processed.py)
  Randomly selects a sensor file and plots raw vs processed signals for visual checking.

## Sensor Filtering

You can choose to include or exclude specific sensor IDs during reach detection.

This can now be done in two ways:

1. Through prompts when the pipeline starts
   The program will ask:
   - which sensors to include
   - which sensors to exclude

2. Through command-line options

Examples:

```powershell
python main.py --project-root "C:\path\to\Project" --sensor-include "17738,21263"
```

```powershell
python main.py --project-root "C:\path\to\Project" --sensor-exclude "17794,21146"
```

This affects which sensors are included in:

- `Reach_Files`
- `Reach_Metrics`

## Detector Modes

The pipeline currently supports two reach detector modes:

- `fixed`
  Uses the standard fixed thresholds

- `adaptive_baseline`
  Estimates thresholds from the beginning of each file

If you are unsure which to use, start with:

```text
fixed
```

## Notes For Non-Programmers

- You do not need to create the output folders yourself.
- You do not need to rename the generated files.
- If you rerun the pipeline, existing output files may be overwritten.
- If something looks wrong, a good first check is to compare:
  - `Output_CSVs`
  - `Processed_CSVs`
  - `Reach_Metrics`

## Quick Start

If you only remember one step, use this:

```powershell
python main.py
```

Then:

1. Select the project folder containing `Raw_Data`
2. Choose the participants
3. Choose the detector mode
4. Choose which sensors to include
5. Choose which sensors to exclude
6. Wait for the pipeline to finish

When it is done, look in:

- `Output_CSVs`
- `Processed_CSVs`
- `Reach_Files`
- `Reach_Metrics`
