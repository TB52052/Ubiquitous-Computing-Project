from __future__ import annotations

"""
Rebuilt shared core module for the fall-detection project.

This file is designed to replace the missing original `features_model.py` that
was expected by `multiclass_analysis.py` and the binary evaluation/export
script. It provides a compatible API:

- build_feature_table(data_dir, output_dir)
- get_feature_columns(df)
- get_models()
- configuration globals such as RESAMPLE_HZ / WINDOW_SECONDS / OVERLAP

Important note
--------------
This is a reconstructed core implementation based on the structure of the
uploaded project scripts and the available legacy activity-recognition code.
It is compatible with the current pipeline layout, but it cannot guarantee
bit-for-bit reproduction of the lost original results.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


# -----------------------------------------------------------------------------
# Runtime configuration expected by the evaluation scripts
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
RESAMPLE_HZ = 32.0
WINDOW_SECONDS = 4.0
OVERLAP = 0.5
MIN_WINDOW_ROWS = int(RESAMPLE_HZ * WINDOW_SECONDS)
USE_PRESSURE = True
TRIM_SECONDS = 0.0


# -----------------------------------------------------------------------------
# Session representation
# -----------------------------------------------------------------------------
@dataclass
class SessionRecord:
    session_dir: Path
    session_id: str
    raw_label: str
    binary_group: str
    accel_path: Path
    gyro_path: Path
    pressure_path: Optional[Path]


# -----------------------------------------------------------------------------
# Feature definition
# -----------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "acc_mag_mean", "acc_mag_std", "acc_mag_min", "acc_mag_max", "acc_mag_range", "acc_mag_iqr", "acc_mag_energy",
    "gyro_mag_mean", "gyro_mag_std", "gyro_mag_min", "gyro_mag_max", "gyro_mag_range", "gyro_mag_iqr", "gyro_mag_energy",
    "acc_x_mean", "acc_x_std", "acc_y_mean", "acc_y_std", "acc_z_mean", "acc_z_std",
    "gyro_x_mean", "gyro_x_std", "gyro_y_mean", "gyro_y_std", "gyro_z_mean", "gyro_z_std",
    "impact_peak_acc", "impact_peak_gyro",
    "post_acc_mean", "post_acc_var", "post_gyro_mean", "post_gyro_var",
    "pressure_mean", "pressure_std", "pressure_range", "pressure_slope", "has_pressure",
    "acc_fft_band_1", "acc_fft_band_2", "acc_fft_band_3", "acc_fft_band_4",
    "gyro_fft_band_1", "gyro_fft_band_2", "gyro_fft_band_3", "gyro_fft_band_4",
]


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def normalise_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_iqr(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    q75, q25 = np.percentile(values, [75, 25])
    return float(q75 - q25)


def calc_slope(y: np.ndarray, x: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    x_centered = x - x.mean()
    denom = float(np.sum(x_centered ** 2))
    if denom == 0.0:
        return 0.0
    return float(np.sum(x_centered * (y - y.mean())) / denom)


def band_powers(signal: np.ndarray, n_bands: int = 4) -> List[float]:
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return [0.0] * n_bands
    centered = signal - signal.mean()
    spec = np.abs(np.fft.rfft(centered)) ** 2
    if spec.size <= 1:
        return [0.0] * n_bands
    spec = spec[1:]
    bins = np.array_split(spec, n_bands)
    return [float(np.sum(b)) if len(b) else 0.0 for b in bins]


# -----------------------------------------------------------------------------
# Dataset discovery
# -----------------------------------------------------------------------------
def infer_binary_group(path_parts: Sequence[str], raw_label: str) -> str:
    lowered = [normalise_label(p) for p in path_parts]
    if "fall" in lowered:
        return "fall"
    if "non_fall" in lowered or "nonfall" in lowered:
        return "non_fall"

    raw = normalise_label(raw_label)
    if raw == "fall" or raw.startswith("fall_"):
        return "fall"
    return "non_fall"


def infer_raw_label(data_dir: Path, session_dir: Path) -> str:
    try:
        rel_parts = list(session_dir.relative_to(data_dir).parts)
    except ValueError:
        rel_parts = list(session_dir.parts)

    # Drop the last part because it is the session identifier itself.
    parent_parts = [normalise_label(p) for p in rel_parts[:-1]]
    structural = {
        "complete", "incomplete", "session", "sessions", "train", "test",
        "valid", "validation", "data", "organized_data"
    }
    category_names = {"fall", "non_fall", "nonfall"}
    candidates = [p for p in parent_parts if p not in structural and p not in category_names]

    if candidates:
        return candidates[-1]
    if parent_parts:
        return parent_parts[-1]
    return normalise_label(session_dir.name)


def discover_sessions(data_dir: Path) -> Tuple[List[SessionRecord], pd.DataFrame]:
    records: List[SessionRecord] = []
    unknown_rows: List[Dict[str, str]] = []

    accel_files = sorted(data_dir.rglob("accel.txt"))
    if not accel_files:
        raise FileNotFoundError(f"No accel.txt files were found under {data_dir}")

    for accel_path in accel_files:
        session_dir = accel_path.parent
        gyro_path = session_dir / "gyro.txt"
        pressure_path = session_dir / "pressure.txt"

        if not gyro_path.exists():
            unknown_rows.append({"path": str(session_dir), "reason": "missing_gyro"})
            continue

        raw_label = infer_raw_label(data_dir, session_dir)
        try:
            rel_parts = list(session_dir.relative_to(data_dir).parts)
        except ValueError:
            rel_parts = list(session_dir.parts)
        binary_group = infer_binary_group(rel_parts, raw_label)

        records.append(
            SessionRecord(
                session_dir=session_dir,
                session_id=session_dir.name,
                raw_label=raw_label,
                binary_group=binary_group,
                accel_path=accel_path,
                gyro_path=gyro_path,
                pressure_path=pressure_path if pressure_path.exists() and USE_PRESSURE else None,
            )
        )

    unknown_df = pd.DataFrame(unknown_rows, columns=["path", "reason"])
    return records, unknown_df


# -----------------------------------------------------------------------------
# Sensor reading and resampling
# -----------------------------------------------------------------------------
def read_sensor(path: Path, expected_cols: int) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if df.shape[1] < expected_cols:
        raise ValueError(f"{path} has {df.shape[1]} columns, expected at least {expected_cols}")

    if expected_cols == 4:
        df = df.iloc[:, :4].copy()
        df.columns = ["timestamp", "x", "y", "z"]
    elif expected_cols == 2:
        df = df.iloc[:, :2].copy()
        df.columns = ["timestamp", "value"]
    else:
        raise ValueError("expected_cols must be 4 or 2")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values("timestamp")
    df = df.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{path} contains no valid numeric rows after cleaning")
    return df


def resample_sensor(df: pd.DataFrame, grid: np.ndarray, expected_cols: int) -> pd.DataFrame:
    out = pd.DataFrame({"timestamp": grid})
    if expected_cols == 4:
        for col in ["x", "y", "z"]:
            out[col] = np.interp(grid, df["timestamp"].to_numpy(), df[col].to_numpy())
    else:
        out["value"] = np.interp(grid, df["timestamp"].to_numpy(), df["value"].to_numpy())
    return out


def align_and_resample_session(record: SessionRecord) -> pd.DataFrame:
    accel = read_sensor(record.accel_path, expected_cols=4)
    gyro = read_sensor(record.gyro_path, expected_cols=4)
    pressure = read_sensor(record.pressure_path, expected_cols=2) if record.pressure_path else None

    start_ts = max(float(accel["timestamp"].min()), float(gyro["timestamp"].min()))
    end_ts = min(float(accel["timestamp"].max()), float(gyro["timestamp"].max()))
    if pressure is not None:
        start_ts = max(start_ts, float(pressure["timestamp"].min()))
        end_ts = min(end_ts, float(pressure["timestamp"].max()))

    start_ts += TRIM_SECONDS * 1000.0
    end_ts -= TRIM_SECONDS * 1000.0
    if end_ts <= start_ts:
        raise ValueError("No overlapping time range after alignment")

    step_ms = 1000.0 / float(RESAMPLE_HZ)
    grid = np.arange(start_ts, end_ts, step_ms)
    if grid.size < 8:
        raise ValueError("Aligned session is too short after resampling")

    accel_r = resample_sensor(accel, grid, expected_cols=4)
    gyro_r = resample_sensor(gyro, grid, expected_cols=4)

    if pressure is not None:
        pressure_r = resample_sensor(pressure, grid, expected_cols=2)
        pressure_values = pressure_r["value"].to_numpy(dtype=float)
        has_pressure = 1.0
    else:
        pressure_values = np.zeros_like(grid, dtype=float)
        has_pressure = 0.0

    return pd.DataFrame(
        {
            "timestamp": grid,
            "acc_x": accel_r["x"].to_numpy(dtype=float),
            "acc_y": accel_r["y"].to_numpy(dtype=float),
            "acc_z": accel_r["z"].to_numpy(dtype=float),
            "gyro_x": gyro_r["x"].to_numpy(dtype=float),
            "gyro_y": gyro_r["y"].to_numpy(dtype=float),
            "gyro_z": gyro_r["z"].to_numpy(dtype=float),
            "pressure": pressure_values,
            "has_pressure": has_pressure,
        }
    )


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------
def extract_window_features(window: pd.DataFrame) -> Dict[str, float]:
    acc = window[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=float)
    gyro = window[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype=float)
    pressure = window["pressure"].to_numpy(dtype=float)
    ts = window["timestamp"].to_numpy(dtype=float) / 1000.0

    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    post_start = int(0.75 * len(window))
    post_acc = acc_mag[post_start:] if post_start < len(acc_mag) else acc_mag
    post_gyro = gyro_mag[post_start:] if post_start < len(gyro_mag) else gyro_mag

    values: Dict[str, float] = {
        "acc_mag_mean": float(np.mean(acc_mag)),
        "acc_mag_std": float(np.std(acc_mag)),
        "acc_mag_min": float(np.min(acc_mag)),
        "acc_mag_max": float(np.max(acc_mag)),
        "acc_mag_range": float(np.max(acc_mag) - np.min(acc_mag)),
        "acc_mag_iqr": safe_iqr(acc_mag),
        "acc_mag_energy": float(np.mean(acc_mag ** 2)),
        "gyro_mag_mean": float(np.mean(gyro_mag)),
        "gyro_mag_std": float(np.std(gyro_mag)),
        "gyro_mag_min": float(np.min(gyro_mag)),
        "gyro_mag_max": float(np.max(gyro_mag)),
        "gyro_mag_range": float(np.max(gyro_mag) - np.min(gyro_mag)),
        "gyro_mag_iqr": safe_iqr(gyro_mag),
        "gyro_mag_energy": float(np.mean(gyro_mag ** 2)),
        "acc_x_mean": float(np.mean(acc[:, 0])),
        "acc_x_std": float(np.std(acc[:, 0])),
        "acc_y_mean": float(np.mean(acc[:, 1])),
        "acc_y_std": float(np.std(acc[:, 1])),
        "acc_z_mean": float(np.mean(acc[:, 2])),
        "acc_z_std": float(np.std(acc[:, 2])),
        "gyro_x_mean": float(np.mean(gyro[:, 0])),
        "gyro_x_std": float(np.std(gyro[:, 0])),
        "gyro_y_mean": float(np.mean(gyro[:, 1])),
        "gyro_y_std": float(np.std(gyro[:, 1])),
        "gyro_z_mean": float(np.mean(gyro[:, 2])),
        "gyro_z_std": float(np.std(gyro[:, 2])),
        "impact_peak_acc": float(np.max(acc_mag)),
        "impact_peak_gyro": float(np.max(gyro_mag)),
        "post_acc_mean": float(np.mean(post_acc)),
        "post_acc_var": float(np.var(post_acc)),
        "post_gyro_mean": float(np.mean(post_gyro)),
        "post_gyro_var": float(np.var(post_gyro)),
        "pressure_mean": float(np.mean(pressure)),
        "pressure_std": float(np.std(pressure)),
        "pressure_range": float(np.max(pressure) - np.min(pressure)),
        "pressure_slope": calc_slope(pressure, ts),
        "has_pressure": float(window["has_pressure"].iloc[0]),
    }

    acc_bands = band_powers(acc_mag, n_bands=4)
    gyro_bands = band_powers(gyro_mag, n_bands=4)
    for i, val in enumerate(acc_bands, start=1):
        values[f"acc_fft_band_{i}"] = float(val)
    for i, val in enumerate(gyro_bands, start=1):
        values[f"gyro_fft_band_{i}"] = float(val)
    return values


# -----------------------------------------------------------------------------
# Public API expected by the rest of the project
# -----------------------------------------------------------------------------
def build_feature_table(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    """
    Build a window-level feature table from the organised dataset.

    Side effects
    ------------
    Writes two optional diagnostic files to `output_dir`:
    - session_summary.csv
    - unknown_sessions_from_model.csv
    """
    session_records, unknown_df = discover_sessions(data_dir)
    rows: List[Dict[str, object]] = []
    session_summary_rows: List[Dict[str, object]] = []
    skip_rows: List[Dict[str, str]] = []

    window_rows = int(MIN_WINDOW_ROWS)
    if window_rows < 8:
        raise ValueError("MIN_WINDOW_ROWS must be at least 8")

    step_rows = max(1, int(round(window_rows * (1.0 - OVERLAP))))

    for rec in session_records:
        try:
            aligned = align_and_resample_session(rec)
        except Exception as exc:  # noqa: BLE001
            skip_rows.append({"path": str(rec.session_dir), "reason": f"preprocess_failed:{exc}"})
            continue

        n_windows = 0
        for start in range(0, len(aligned) - window_rows + 1, step_rows):
            window = aligned.iloc[start : start + window_rows].reset_index(drop=True)
            feats = extract_window_features(window)
            feats.update(
                {
                    "session_id": rec.session_id,
                    "raw_label": rec.raw_label,
                    "binary_group": rec.binary_group,
                    "binary_label": 1 if rec.binary_group == "fall" else 0,
                    "window_index": n_windows,
                }
            )
            rows.append(feats)
            n_windows += 1

        if n_windows == 0:
            skip_rows.append({"path": str(rec.session_dir), "reason": "too_short_for_windowing"})
            continue

        session_summary_rows.append(
            {
                "session_id": rec.session_id,
                "raw_label": rec.raw_label,
                "binary_group": rec.binary_group,
                "n_windows": n_windows,
                "has_pressure": 1 if rec.pressure_path else 0,
            }
        )

    feat_df = pd.DataFrame(rows)
    if feat_df.empty:
        feat_df = pd.DataFrame(columns=FEATURE_COLUMNS + ["session_id", "raw_label", "binary_group", "binary_label", "window_index"])

    session_summary = pd.DataFrame(session_summary_rows)
    if session_summary.empty:
        session_summary = pd.DataFrame(columns=["session_id", "raw_label", "binary_group", "n_windows", "has_pressure"])

    skipped_df = pd.concat(
        [unknown_df, pd.DataFrame(skip_rows, columns=["path", "reason"])],
        ignore_index=True,
    )

    ensure_output_dir(output_dir)
    session_summary.to_csv(output_dir / "session_summary.csv", index=False)
    skipped_df.to_csv(output_dir / "unknown_sessions_from_model.csv", index=False)
    return feat_df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in FEATURE_COLUMNS if c in df.columns]


def get_models() -> Dict[str, Pipeline]:
    """
    Binary candidate models expected by the binary evaluation script.
    The key `logreg` is required by `fit_full_logreg(...)`.
    """
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=600, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "linear_svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        ]),
        "tree": Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=10, min_samples_leaf=4, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "rf": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=350, max_depth=14, min_samples_leaf=3,
                                            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "extra_trees": Pipeline([
            ("clf", ExtraTreesClassifier(n_estimators=350, max_depth=14, min_samples_leaf=3,
                                          class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "hist_gb": Pipeline([
            ("clf", HistGradientBoostingClassifier(max_depth=8, learning_rate=0.08, max_iter=250,
                                                    random_state=RANDOM_STATE)),
        ]),
    }


__all__ = [
    "RANDOM_STATE",
    "RESAMPLE_HZ",
    "WINDOW_SECONDS",
    "OVERLAP",
    "MIN_WINDOW_ROWS",
    "USE_PRESSURE",
    "TRIM_SECONDS",
    "FEATURE_COLUMNS",
    "build_feature_table",
    "get_feature_columns",
    "get_models",
]
