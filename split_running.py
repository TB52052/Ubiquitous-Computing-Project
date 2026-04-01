from __future__ import annotations

"""
Split a single long running session into multiple shorter sessions.

Why this script exists
----------------------
In activity-recognition or fall-detection projects, some classes may have very
few sessions even when one of the recordings is extremely long. This can create
practical problems such as:
- poor class balance at the *session* level,
- labels being excluded by `min_sessions_per_label` filters,
- less stable grouped cross-validation,
- one unusually long recording dominating the dataset.

This utility addresses that situation by taking one long running recording,
aligning the available sensor streams to a common valid time range, and slicing
that range into several new session folders.

Expected file formats
---------------------
- accel.txt / gyro.txt : timestamp, x, y, z
- pressure.txt         : timestamp, value

The script keeps the original numeric values unchanged. It only:
1. cleans malformed rows,
2. aligns sensor streams by overlapping timestamp range,
3. splits the overlap into equal time segments,
4. writes each segment as a new session.
"""

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# USER CONFIGURATION
# -----------------------------------------------------------------------------
# Edit the following paths and parameters directly, then run the script.
# This design keeps the file convenient for one-off dataset repair operations.
# =============================================================================

# Path to the accelerometer file of the long running session to be split.
ACCEL_FILE = Path(
    r"D:\UBAfinal\organized_data\non_fall\running\complete\355982084243690-2026-03-21_11-52-21-Running-test1\accel.txt"
)

# Path to the gyroscope file of the same session.
GYRO_FILE = Path(
    r"D:\UBAfinal\organized_data\non_fall\running\complete\355982084243690-2026-03-21_11-52-21-Running-test1\gyro.txt"
)

# Optional barometer/pressure file. Set to None if pressure is not available.
PRESSURE_FILE: Optional[Path] = Path(
    r"D:\UBAfinal\organized_data\non_fall\running\complete\355982084243690-2026-03-21_11-52-21-Running-test1\pressure.txt"
)

# Root directory where the new split sessions will be written.
OUTPUT_DIR = Path(r"D:\UBAfinal\organized_data_split")

# Base session name used when naming the generated parts.
SESSION_NAME = "355982084243690-2026-03-21_11-52-21-Running-test1"

# Number of output sessions to create from the original long session.
NUM_PARTS = 3

# Optional trimming from the aligned common range.
# These values are in seconds and are useful if the beginning or end contains
# setup noise, delays, or partial movement that should be discarded.
TRIM_START_SEC = 0.0
TRIM_END_SEC = 0.0

# Safety guard: reject splits that would make each part too short to be useful.
MIN_PART_SEC = 30.0

# =============================================================================
# INTERNAL HELPERS
# =============================================================================

# Recognises common sensor suffixes so a clean base session name can be inferred
# from a file path when a manual SESSION_NAME is not provided.
SENSOR_SUFFIX_RE = re.compile(r"([_-]?)(accel|gyro|pressure)(\.txt)?$", re.IGNORECASE)


def infer_base_session_name(path: Path) -> str:
    """
    Infer a base session name from a sensor filename.

    Example
    -------
    ".../session_01_accel.txt" -> "session_01"

    This helper is mainly a fallback for convenience. In normal use, the script
    can rely on the explicit `SESSION_NAME` constant defined above.
    """
    stem = path.stem
    stem = SENSOR_SUFFIX_RE.sub("", stem)
    stem = stem.rstrip("-_")
    return stem or "running_session"


def safe_name(name: str) -> str:
    """
    Convert an arbitrary session name into a filesystem-safe folder name.

    This removes characters that are invalid on common operating systems and
    normalises whitespace to underscores.
    """
    name = re.sub(r'[<>:"/\\|?*]+', "_", name)
    name = re.sub(r"\s+", "_", name).strip("._ ")
    return name or "session"


def read_sensor(path: Path, expected_cols: int) -> pd.DataFrame:
    """
    Read, validate, and clean a sensor text file.

    Parameters
    ----------
    path:
        File path to the raw sensor text file.
    expected_cols:
        Expected minimum number of columns.
        - 4 for 3-axis sensors: timestamp, x, y, z
        - 2 for scalar sensors: timestamp, value

    Returns
    -------
    pandas.DataFrame
        A cleaned dataframe sorted by timestamp and deduplicated on timestamp.

    Cleaning steps
    --------------
    1. Read raw CSV-like content without headers.
    2. Keep only the expected columns.
    3. Assign semantic column names.
    4. Convert all values to numeric.
    5. Drop invalid rows.
    6. Sort by timestamp.
    7. Remove duplicate timestamps.

    The goal is to make later alignment and slicing deterministic.
    """
    df = pd.read_csv(path, header=None)
    if df.shape[1] < expected_cols:
        raise ValueError(f"{path} has {df.shape[1]} columns, expected at least {expected_cols}")

    # Interpret the file according to the expected sensor structure.
    if expected_cols == 4:
        df = df.iloc[:, :4].copy()
        df.columns = ["timestamp", "x", "y", "z"]
    elif expected_cols == 2:
        df = df.iloc[:, :2].copy()
        df.columns = ["timestamp", "value"]
    else:
        raise ValueError("expected_cols must be 4 or 2")

    # Remove obvious empty rows before numeric conversion.
    df = df.dropna().copy()

    # Convert all columns to numeric so malformed text becomes NaN and can be
    # filtered out cleanly.
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only valid rows and enforce chronological order.
    df = df.dropna().sort_values("timestamp")

    # Duplicate timestamps can break range slicing and resampling logic later,
    # so keep the first occurrence only.
    df = df.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"{path} contains no valid rows after cleaning")
    return df


def common_time_range(
    accel: pd.DataFrame,
    gyro: pd.DataFrame,
    pressure: Optional[pd.DataFrame],
    trim_start_sec: float,
    trim_end_sec: float,
) -> Tuple[float, float]:
    """
    Compute the shared valid timestamp interval across all provided sensors.

    The script only uses the time span in which *all required sensors* have data.
    This avoids creating split sessions where one sensor starts later or ends
    earlier than the others.

    The optional trimming arguments are applied after alignment so the user can
    discard unwanted setup or ending sections.
    """
    # The latest sensor start time defines the earliest safe shared timestamp.
    start_ts = max(float(accel["timestamp"].min()), float(gyro["timestamp"].min()))

    # The earliest sensor end time defines the latest safe shared timestamp.
    end_ts = min(float(accel["timestamp"].max()), float(gyro["timestamp"].max()))

    # If pressure is present, it must also overlap with the other sensors.
    if pressure is not None:
        start_ts = max(start_ts, float(pressure["timestamp"].min()))
        end_ts = min(end_ts, float(pressure["timestamp"].max()))

    # Convert trim values from seconds to milliseconds because the sensor
    # timestamps are assumed to be millisecond-based.
    start_ts += trim_start_sec * 1000.0
    end_ts -= trim_end_sec * 1000.0

    if end_ts <= start_ts:
        raise ValueError(
            "No usable overlapping time range after alignment and trimming. "
            "Check timestamps or reduce TRIM_START_SEC / TRIM_END_SEC."
        )
    return start_ts, end_ts


def slice_df(df: pd.DataFrame, start_ts: float, end_ts: float, is_last: bool) -> pd.DataFrame:
    """
    Slice one sensor dataframe to a timestamp interval.

    The final segment includes the right boundary (`<= end_ts`) so the full
    aligned range is covered without dropping the last sample. Earlier segments
    use `< end_ts` to avoid duplicating boundary rows between adjacent parts.
    """
    if is_last:
        mask = (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
    else:
        mask = (df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)
    return df.loc[mask].copy().reset_index(drop=True)


def save_sensor(df: pd.DataFrame, out_path: Path) -> None:
    """
    Save a cleaned/sliced sensor dataframe back to disk without headers.

    The output format intentionally mirrors the original text-file layout so the
    generated sessions remain compatible with the rest of the project pipeline.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, header=False, index=False)


def build_output_session_dir(output_dir: Path, session_name: str, part_idx: int, total_parts: int) -> Path:
    """
    Construct the destination folder for one generated session part.

    The folder structure matches the existing organised dataset convention:
    non_fall / running / complete / <session_name_partXXofYY>
    """
    safe_session = safe_name(f"{session_name}_part{part_idx:02d}of{total_parts:02d}")
    return output_dir / "non_fall" / "running" / "complete" / safe_session


def main() -> None:
    """
    Split the configured running session into multiple new sessions.

    End-to-end workflow
    -------------------
    1. Validate user configuration.
    2. Load accel / gyro / optional pressure files.
    3. Clean each file independently.
    4. Compute their overlapping valid time range.
    5. Divide that range into equal segments.
    6. Slice each sensor stream into those segments.
    7. Save each segment as a new session directory.
    8. Write a manifest summarising the generated sessions.
    """
    if NUM_PARTS < 2:
        raise ValueError("NUM_PARTS must be at least 2")

    # Resolve to absolute paths so printed paths and file operations are stable.
    accel_file = ACCEL_FILE.resolve()
    gyro_file = GYRO_FILE.resolve()
    pressure_file = PRESSURE_FILE.resolve() if PRESSURE_FILE else None
    output_dir = OUTPUT_DIR.resolve()

    # Fail early if any required input file is missing.
    if not accel_file.exists():
        raise FileNotFoundError(f"ACCEL_FILE does not exist: {accel_file}")
    if not gyro_file.exists():
        raise FileNotFoundError(f"GYRO_FILE does not exist: {gyro_file}")
    if pressure_file is not None and not pressure_file.exists():
        raise FileNotFoundError(f"PRESSURE_FILE does not exist: {pressure_file}")

    # Use the explicit session name if provided; otherwise infer one.
    session_name = SESSION_NAME.strip() or infer_base_session_name(accel_file)

    # Read and clean all available sensor streams.
    accel = read_sensor(accel_file, expected_cols=4)
    gyro = read_sensor(gyro_file, expected_cols=4)
    pressure = read_sensor(pressure_file, expected_cols=2) if pressure_file else None

    # Keep only the time interval shared by every required sensor.
    start_ts, end_ts = common_time_range(
        accel=accel,
        gyro=gyro,
        pressure=pressure,
        trim_start_sec=TRIM_START_SEC,
        trim_end_sec=TRIM_END_SEC,
    )

    total_duration_sec = (end_ts - start_ts) / 1000.0
    part_duration_sec = total_duration_sec / NUM_PARTS

    # Protect against over-splitting into segments that would be too short to be
    # meaningful for feature extraction or model training.
    if part_duration_sec < MIN_PART_SEC:
        raise ValueError(
            f"Each part would be only {part_duration_sec:.2f}s, which is smaller than MIN_PART_SEC={MIN_PART_SEC}."
        )

    # Create equally spaced boundaries over the valid common range.
    boundaries = np.linspace(start_ts, end_ts, NUM_PARTS + 1)
    manifest_rows = []

    for i in range(NUM_PARTS):
        part_start = float(boundaries[i])
        part_end = float(boundaries[i + 1])
        is_last = i == NUM_PARTS - 1

        # Slice each sensor stream using the same time boundaries.
        accel_part = slice_df(accel, part_start, part_end, is_last=is_last)
        gyro_part = slice_df(gyro, part_start, part_end, is_last=is_last)
        pressure_part = slice_df(pressure, part_start, part_end, is_last=is_last) if pressure is not None else None

        # Empty output would indicate a broken split or unexpected timestamp
        # layout, so stop immediately rather than writing incomplete sessions.
        if accel_part.empty:
            raise RuntimeError(f"Accel part {i + 1} is empty after slicing.")
        if gyro_part.empty:
            raise RuntimeError(f"Gyro part {i + 1} is empty after slicing.")
        if pressure is not None and pressure_part is not None and pressure_part.empty:
            raise RuntimeError(f"Pressure part {i + 1} is empty after slicing.")

        session_dir = build_output_session_dir(output_dir, session_name, i + 1, NUM_PARTS)

        # Preserve the expected sensor filenames so the downstream pipeline can
        # consume the generated sessions without any extra configuration.
        save_sensor(accel_part, session_dir / "accel.txt")
        save_sensor(gyro_part, session_dir / "gyro.txt")
        if pressure_part is not None:
            save_sensor(pressure_part, session_dir / "pressure.txt")

        # Record metadata for auditing and reproducibility.
        manifest_rows.append(
            {
                "part_index": i + 1,
                "session_dir": str(session_dir),
                "start_ts": part_start,
                "end_ts": part_end,
                "duration_sec": (part_end - part_start) / 1000.0,
                "accel_rows": len(accel_part),
                "gyro_rows": len(gyro_part),
                "pressure_rows": len(pressure_part) if pressure_part is not None else 0,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "running_split_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    # Human-readable summary for quick verification.
    print("Done.")
    print(f"Session name     : {session_name}")
    print(f"Total duration   : {total_duration_sec:.2f} sec")
    print(f"Number of parts  : {NUM_PARTS}")
    print(f"Part duration    : {part_duration_sec:.2f} sec")
    print(f"Output root      : {output_dir}")
    print(f"Manifest written : {manifest_path}")


if __name__ == "__main__":
    main()
