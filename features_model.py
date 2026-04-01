from __future__ import annotations

"""
Binary fall-detection evaluation and export script.

What this file does
-------------------
This script evaluates binary fall detection on top of the project's shared
feature-extraction pipeline. It focuses on the question:

    "Given a window of sensor-derived features, is this window a fall or not?"

The script performs several jobs:
1. Build a feature table from the organised dataset.
2. Summarise the data that entered the pipeline.
3. Compare candidate binary classifiers.
4. Save plots and optional debug CSV files.
5. Train a final logistic-regression model on the full dataset.
6. Export model parameters for Android-side inference.
7. Apply an additional fusion rule that combines model probability with
   interpretable impact/stillness thresholds.

Architectural note
------------------
This file imports `features_model as core`, which means it relies on a shared
implementation module that contains lower-level functions such as:
- build_feature_table(...)
- get_models()
- get_feature_columns(...)
- configuration constants like RESAMPLE_HZ / WINDOW_SECONDS / OVERLAP

In other words, this script is the evaluation/orchestration layer, not the
lowest-level feature-engineering implementation itself.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

import features_model as core


# Default project locations.
DEFAULT_DATA_DIR = Path("./organized_data")
DEFAULT_OUTPUT_DIR = Path("./model_output")


def ensure_output_dir(output_dir: Path) -> None:
    """
    Ensure the output directory exists before writing any artefacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)


def save_confusion_matrix_png(cm: np.ndarray, labels: List[str], title: str, out_path: Path) -> None:
    """
    Save a confusion matrix as a PNG image.

    Parameters
    ----------
    cm:
        2D confusion matrix array.
    labels:
        Human-readable class labels in the same order used to compute `cm`.
    title:
        Plot title.
    out_path:
        Destination PNG path.

    Why save a figure?
    ------------------
    Confusion matrices are one of the most interpretable ways to inspect model
    behaviour. For fall detection, they immediately show the trade-off between
    false alarms and missed falls.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    # Annotate each cell with its integer count. The text colour is adapted to
    # the background intensity for readability.
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison_png(df: pd.DataFrame, out_path: Path) -> None:
    """
    Save a grouped bar chart comparing binary classifier performance.

    The selected metrics focus on fall detection quality rather than only raw
    accuracy, because in imbalanced safety tasks a high accuracy alone can be
    misleading.
    """
    metrics = ["accuracy", "precision_fall", "recall_fall", "f1_fall"]
    x = np.arange(len(df))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, df[metric].values, width=width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"].tolist())
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Binary fall-detection model comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_session_counts_png(session_summary: pd.DataFrame, out_path: Path) -> None:
    """
    Plot the number of sessions per raw activity label.

    Even though this is a binary fall-detection script, the raw activity labels
    still matter because they reveal which underlying activities contribute to
    the `fall` and `non_fall` categories.
    """
    counts = session_summary["raw_label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.index, counts.values)
    ax.set_title("Session counts by activity label")
    ax.set_xlabel("Activity label")
    ax.set_ylabel("Number of sessions")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def configure_core(args: argparse.Namespace) -> None:
    """
    Forward user-selected preprocessing parameters into the shared core module.

    This keeps binary and multiclass experiments aligned: if you change the
    resampling rate, window length, overlap, or pressure usage here, the same
    settings can be reused across the project.
    """
    core.RESAMPLE_HZ = float(args.resample_hz)
    core.WINDOW_SECONDS = float(args.window_seconds)
    core.OVERLAP = float(args.overlap)
    core.MIN_WINDOW_ROWS = int(core.RESAMPLE_HZ * core.WINDOW_SECONDS)
    core.USE_PRESSURE = not args.no_pressure


def build_feature_table_in_temp(
    data_dir: Path,
    save_debug_csv: bool,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build the window-level feature table in a temporary workspace.

    Outputs
    -------
    feat_df:
        Main dataframe used for model training and evaluation.
    session_summary:
        Per-session summary emitted by the core pipeline.
    unknown_df:
        Rows describing skipped or unrecognised sessions/files.

    Why a temporary directory?
    --------------------------
    The shared pipeline may create intermediate diagnostic files. Using a
    temporary directory keeps the permanent output folder clean unless the user
    explicitly asks to retain debug CSVs.
    """
    with tempfile.TemporaryDirectory(prefix="fall_model_tmp_") as tmp:
        tmp_dir = Path(tmp)

        # This is the central preprocessing + feature extraction call.
        feat_df = core.build_feature_table(data_dir, tmp_dir)

        session_summary_path = tmp_dir / "session_summary.csv"
        unknown_path = tmp_dir / "unknown_sessions_from_model.csv"

        session_summary = pd.read_csv(session_summary_path) if session_summary_path.exists() else pd.DataFrame()
        unknown_df = pd.read_csv(unknown_path) if unknown_path.exists() else pd.DataFrame(columns=["path", "reason"])

        if save_debug_csv:
            feat_df.to_csv(output_dir / "feature_table.csv", index=False)
            if not session_summary.empty:
                session_summary.to_csv(output_dir / "session_summary.csv", index=False)
            if not unknown_df.empty:
                unknown_df.to_csv(output_dir / "unknown_sessions_from_model.csv", index=False)

    return feat_df, session_summary, unknown_df


def print_data_summary(session_summary: pd.DataFrame, unknown_df: pd.DataFrame) -> None:
    """
    Print a compact summary of the data that entered the binary pipeline.

    This helps confirm that the organised dataset is being interpreted as
    expected before any model metrics are trusted.
    """
    print("\n===== Data Summary =====")
    if session_summary.empty:
        print("No valid sessions entered the feature table.")
        return

    print(f"Sessions used: {len(session_summary)}")
    print(f"Total windows: {int(session_summary['n_windows'].sum())}")
    print(f"Average windows per session: {session_summary['n_windows'].mean():.2f}")
    print("\nSession counts by label:")
    print(session_summary["raw_label"].value_counts().sort_index().to_string())

    if not unknown_df.empty:
        print("\nSkipped / unknown file summary:")
        print(unknown_df["reason"].value_counts().to_string())


def get_cv_splits(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    """
    Choose an appropriate cross-validation strategy.

    Strategy
    --------
    - Prefer GroupKFold when there are enough independent sessions. This is the
      safer option because it prevents windows from the same session appearing
      in both train and test folds.
    - Fall back to StratifiedKFold only when there are too few unique sessions
      for a meaningful grouped split.

    Returns
    -------
    (splits, name)
        `splits` is a precomputed list of train/test indices.
        `name` is a short human-readable description used in console output.
    """
    unique_groups = groups.nunique()
    if unique_groups >= 4:
        cv = GroupKFold(n_splits=min(5, unique_groups))
        return list(cv.split(X, y, groups=groups)), f"GroupKFold(n_splits={min(5, unique_groups)})"

    # If grouping is not feasible, preserve class balance as well as possible.
    minority = int(y.value_counts().min())
    n_splits = min(3, max(2, minority))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=core.RANDOM_STATE)
    return list(cv.split(X, y)), f"StratifiedKFold(n_splits={n_splits})"


def evaluate_models(
    df: pd.DataFrame,
    output_dir: Path,
    save_debug_csv: bool,
    save_predictions_csv: bool,
) -> pd.DataFrame:
    """
    Compare all candidate binary classifiers on the feature table.

    Evaluation unit
    ---------------
    The rows of `df` are window-level samples, but grouping by `session_id`
    keeps evaluation more realistic by respecting the original recording
    boundaries.

    Returns
    -------
    pandas.DataFrame
        Model comparison table sorted by fall-class F1 score.
    """
    X = df[core.get_feature_columns(df)]
    y = df["binary_label"].astype(int)
    groups = df["session_id"].astype(str)
    models = core.get_models()

    splits, cv_name = get_cv_splits(X, y, groups)
    print(f"\nCross-validation: {cv_name}")

    results = []
    labels = ["non_fall", "fall"]

    for name, model in models.items():
        y_true_all: List[int] = []
        y_pred_all: List[int] = []
        y_prob_all: List[float] = []

        for train_idx, test_idx in splits:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # clone(...) guarantees that each fold starts with a fresh estimator
            # instead of reusing state from the previous fold.
            m = clone(model)
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)

            # Use calibrated probabilities when available; otherwise fall back to
            # hard predictions cast to float so downstream code still works.
            y_prob = m.predict_proba(X_test)[:, 1] if hasattr(m, "predict_proba") else y_pred.astype(float)

            y_true_all.extend(y_test.tolist())
            y_pred_all.extend(y_pred.tolist())
            y_prob_all.extend(y_prob.tolist())

        cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
        save_confusion_matrix_png(cm, labels, f"{name} confusion matrix", output_dir / f"{name}_confusion_matrix.png")

        row = {
            "model": name,
            "accuracy": accuracy_score(y_true_all, y_pred_all),
            "precision_fall": precision_score(y_true_all, y_pred_all, zero_division=0),
            "recall_fall": recall_score(y_true_all, y_pred_all, zero_division=0),
            "f1_fall": f1_score(y_true_all, y_pred_all, zero_division=0),
            "n_eval_windows": len(y_true_all),
        }
        results.append(row)

        if save_predictions_csv:
            pd.DataFrame({"y_true": y_true_all, "y_pred": y_pred_all, "y_prob": y_prob_all}).to_csv(
                output_dir / f"{name}_predictions.csv", index=False
            )

        if save_debug_csv:
            pd.DataFrame(cm, index=["true_nonfall", "true_fall"], columns=["pred_nonfall", "pred_fall"]).to_csv(
                output_dir / f"{name}_confusion_matrix.csv"
            )

    # F1 on the fall class is a sensible ranking metric because this is the
    # safety-critical class the system ultimately cares about most.
    results_df = pd.DataFrame(results).sort_values("f1_fall", ascending=False).reset_index(drop=True)
    save_model_comparison_png(results_df, output_dir / "binary_model_comparison.png")
    if save_debug_csv:
        results_df.to_csv(output_dir / "model_comparison.csv", index=False)

    print("\n===== Binary Model Comparison =====")
    print(results_df.to_string(index=False))
    return results_df


def fit_full_logreg(df: pd.DataFrame):
    """
    Train the project's deployment-oriented logistic regression on all data.

    This is separate from cross-validation because the final exported model
    should be fitted using the full available training set.
    """
    model = core.get_models()["logreg"]
    X = df[core.get_feature_columns(df)]
    y = df["binary_label"].astype(int)
    model.fit(X, y)
    return model


def save_model_artifacts(model, feature_columns: List[str], output_dir: Path) -> None:
    """
    Save the final trained model and, when possible, an Android-friendly JSON.

    Outputs
    -------
    final_logreg.joblib
        Serialized scikit-learn pipeline for Python-side reuse.
    logreg_android_export.json
        Lightweight export of scaler parameters and logistic-regression weights
        so the model can be reimplemented on Android without needing Python.
    """
    # Save the complete Python-side model pipeline.
    joblib.dump(model, output_dir / "final_logreg.joblib")

    scaler = model.named_steps.get("scaler")
    clf = model.named_steps.get("clf")

    # Only export a JSON weight file when the pipeline structure matches the
    # expected "scaler + linear classifier with coefficients" format.
    if scaler is None or clf is None or not hasattr(clf, "coef_"):
        return

    export = {
        "model_type": "logistic_regression",
        "feature_order": feature_columns,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
    }
    (output_dir / "logreg_android_export.json").write_text(json.dumps(export, indent=2), encoding="utf-8")


def run_fusion_rule(
    df: pd.DataFrame,
    output_dir: Path,
    save_debug_csv: bool,
    save_predictions_csv: bool,
) -> Dict[str, float]:
    """
    Apply a hybrid decision rule that combines ML probability with heuristics.

    Rationale
    ---------
    In real fall detection, a classifier alone can produce false positives for
    high-motion non-fall activities. This fusion rule adds domain knowledge:

    - impact_gate: require evidence of a strong acceleration or gyro peak
    - stillness_gate: require evidence of reduced movement afterwards
    - y_prob >= 0.5: require the learned model to also support the decision

    This mirrors the common intuition that a true fall often looks like:
        sudden impact + abnormal motion pattern + post-impact stillness
    """
    model = fit_full_logreg(df)
    feature_cols = core.get_feature_columns(df)
    fused = df.copy()

    # Convert each window into a fall probability using the trained logreg model.
    fused["y_prob"] = model.predict_proba(fused[feature_cols])[:, 1]

    # Simple threshold-based impact gate.
    fused["impact_gate"] = (
        (fused["impact_peak_acc"] >= 15.0) | (fused["impact_peak_gyro"] >= 4.0)
    ).astype(int)

    # Simple threshold-based post-event stillness gate.
    fused["stillness_gate"] = (
        (fused["post_acc_var"] <= 8.0) | (fused["post_gyro_var"] <= 8.0)
    ).astype(int)

    # Final fused decision. All three conditions must agree.
    fused["fusion_pred"] = (
        (fused["impact_gate"] == 1) & (fused["y_prob"] >= 0.5) & (fused["stillness_gate"] == 1)
    ).astype(int)

    cm = confusion_matrix(fused["binary_label"], fused["fusion_pred"], labels=[0, 1])
    save_confusion_matrix_png(cm, ["non_fall", "fall"], "fusion confusion matrix", output_dir / "fusion_confusion_matrix.png")

    metrics = {
        "model": "fusion_rule",
        "accuracy": accuracy_score(fused["binary_label"], fused["fusion_pred"]),
        "precision_fall": precision_score(fused["binary_label"], fused["fusion_pred"], zero_division=0),
        "recall_fall": recall_score(fused["binary_label"], fused["fusion_pred"], zero_division=0),
        "f1_fall": f1_score(fused["binary_label"], fused["fusion_pred"], zero_division=0),
        "n_eval_windows": len(fused),
    }

    if save_predictions_csv:
        fused[[
            "session_id",
            "raw_label",
            "binary_label",
            "y_prob",
            "impact_gate",
            "stillness_gate",
            "fusion_pred",
        ]].to_csv(output_dir / "fusion_predictions.csv", index=False)

    if save_debug_csv:
        pd.DataFrame(cm, index=["true_nonfall", "true_fall"], columns=["pred_nonfall", "pred_fall"]).to_csv(
            output_dir / "fusion_confusion_matrix.csv"
        )

    print("\n===== Fusion Rule Metrics =====")
    print(pd.DataFrame([metrics]).to_string(index=False))
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for binary fall-detection analysis.
    """
    parser = argparse.ArgumentParser(
        description="Optimized binary fall-detection pipeline: keep PNG figures, print key stats to console."
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    # Preprocessing parameters shared with the core feature pipeline.
    parser.add_argument("--window_seconds", type=float, default=core.WINDOW_SECONDS)
    parser.add_argument("--overlap", type=float, default=core.OVERLAP)
    parser.add_argument("--resample_hz", type=float, default=core.RESAMPLE_HZ)
    parser.add_argument("--no_pressure", action="store_true")

    # Optional exports. By default the script keeps the output folder lightweight.
    parser.add_argument("--save_debug_csv", action="store_true", help="Also save feature/session CSV debug files.")
    parser.add_argument("--save_model_artifacts", action="store_true", help="Also save joblib and Android JSON export.")
    parser.add_argument("--save_predictions_csv", action="store_true", help="Also save per-window prediction CSV files.")
    return parser


def main() -> None:
    """
    Run the complete binary fall-detection workflow.

    Execution order
    ---------------
    1. Parse arguments.
    2. Validate paths.
    3. Configure shared preprocessing settings.
    4. Build features.
    5. Print input summary and session distribution.
    6. Evaluate candidate models.
    7. Evaluate the fusion rule.
    8. Optionally export the final deployment model.
    9. Print a compact output summary.
    """
    parser = build_argparser()
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    ensure_output_dir(output_dir)
    configure_core(args)

    feat_df, session_summary, unknown_df = build_feature_table_in_temp(
        data_dir=data_dir,
        save_debug_csv=args.save_debug_csv,
        output_dir=output_dir,
    )

    print_data_summary(session_summary, unknown_df)
    if not session_summary.empty:
        save_session_counts_png(session_summary, output_dir / "session_counts.png")

    results_df = evaluate_models(
        df=feat_df,
        output_dir=output_dir,
        save_debug_csv=args.save_debug_csv,
        save_predictions_csv=args.save_predictions_csv,
    )

    fusion_metrics = run_fusion_rule(
        df=feat_df,
        output_dir=output_dir,
        save_debug_csv=args.save_debug_csv,
        save_predictions_csv=args.save_predictions_csv,
    )

    if args.save_model_artifacts:
        final_model = fit_full_logreg(feat_df)
        save_model_artifacts(final_model, core.get_feature_columns(feat_df), output_dir)

    print("\nDone.")
    print(f"Data dir   : {data_dir}")
    print(f"Output dir : {output_dir}")
    print("\nSaved PNG files:")
    for path in sorted(output_dir.glob("*.png")):
        print(f" - {path.name}")

    if not args.save_model_artifacts:
        print("\nModel artifacts were not saved. Add --save_model_artifacts if you later need joblib/JSON exports.")
    if not args.save_debug_csv and not args.save_predictions_csv:
        print("Only PNG figures were kept in the output directory.")


if __name__ == "__main__":
    main()
