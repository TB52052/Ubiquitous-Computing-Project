from __future__ import annotations

"""
Multiclass activity-recognition evaluation script.

Purpose
-------
This script evaluates how well the extracted window-level sensor features can
separate multiple activity classes such as walking, running, sitting, different
fall types, and other labelled actions.

Important architectural idea
----------------------------
This file does *not* rebuild the full feature-engineering stack itself. It uses
`features_model` as a shared core module so that:
- feature extraction stays consistent across binary and multiclass tasks,
- resampling/windowing settings are defined in one place,
- changes to the preprocessing pipeline automatically affect both analyses.

High-level pipeline
-------------------
1. Configure the shared core preprocessing settings.
2. Build a window-level feature table from the organised dataset.
3. Summarise the dataset and count sessions per label.
4. Exclude labels with too few sessions if requested.
5. Evaluate several classical ML models with GroupKFold by session.
6. Save comparison figures and confusion matrices.
"""

import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import features_model as core


# Reproducibility constant used by models that support seeded randomness.
RANDOM_STATE = 42

# Sensible default project paths.
DEFAULT_DATA_DIR = Path("./organized_data")
DEFAULT_OUTPUT_DIR = Path("./model_output")


def ensure_output_dir(output_dir: Path) -> None:
    """
    Create the output directory if it does not already exist.

    This helper keeps all file-writing functions simple because they can assume
    the destination root is already available.
    """
    output_dir.mkdir(parents=True, exist_ok=True)


def configure_core(args: argparse.Namespace) -> None:
    """
    Push command-line preprocessing settings into the shared core module.

    Why mutate `core` globals?
    --------------------------
    The project uses `features_model` as a central implementation of feature
    extraction. This script reuses that implementation by updating its runtime
    configuration before calling `core.build_feature_table(...)`.
    """
    core.RESAMPLE_HZ = float(args.resample_hz)
    core.WINDOW_SECONDS = float(args.window_seconds)
    core.OVERLAP = float(args.overlap)

    # MIN_WINDOW_ROWS is derived from sampling rate and window length. It
    # defines the minimum number of resampled rows required for a valid window.
    core.MIN_WINDOW_ROWS = int(core.RESAMPLE_HZ * core.WINDOW_SECONDS)

    # Pressure can be disabled if some datasets do not contain that sensor.
    core.USE_PRESSURE = not args.no_pressure


def build_feature_table_in_temp(
    data_dir: Path,
    save_debug_csv: bool,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build the feature table using a temporary workspace.

    Returns
    -------
    feat_df:
        Window-level feature table used for model evaluation.
    session_summary:
        Per-session summary exported by the shared core pipeline.
    unknown_df:
        Records of skipped or unrecognised sessions/files.

    Why use a temporary directory?
    ------------------------------
    The core pipeline may need scratch outputs such as summaries or diagnostic
    CSVs. Keeping those in a temporary folder makes the default workflow clean.
    Only the user-requested debug CSVs are copied into the final output folder.
    """
    with tempfile.TemporaryDirectory(prefix="multiclass_tmp_") as tmp:
        tmp_dir = Path(tmp)

        # Shared preprocessing and feature extraction happen here.
        feat_df = core.build_feature_table(data_dir, tmp_dir)

        session_summary_path = tmp_dir / "session_summary.csv"
        unknown_path = tmp_dir / "unknown_sessions_from_model.csv"

        session_summary = pd.read_csv(session_summary_path) if session_summary_path.exists() else pd.DataFrame()
        unknown_df = pd.read_csv(unknown_path) if unknown_path.exists() else pd.DataFrame(columns=["path", "reason"])

        # Persist optional debug artefacts only when explicitly requested.
        if save_debug_csv:
            feat_df.to_csv(output_dir / "feature_table.csv", index=False)
            if not session_summary.empty:
                session_summary.to_csv(output_dir / "session_summary.csv", index=False)
            if not unknown_df.empty:
                unknown_df.to_csv(output_dir / "unknown_sessions_from_model.csv", index=False)

    return feat_df, session_summary, unknown_df


def get_multiclass_models() -> Dict[str, Pipeline]:
    """
    Define the candidate models for multiclass activity recognition.

    Why compare several models?
    ---------------------------
    Different classical models behave differently on handcrafted sensor
    features. For example:
    - linear models are fast and interpretable,
    - tree ensembles often capture non-linear boundaries better,
    - KNN can work surprisingly well on compact, well-scaled features.

    Each model is wrapped in a `Pipeline` where needed so scaling is always
    applied consistently before the classifier.
    """
    return {
        "logreg_multinomial": Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=600,
                    multi_class="multinomial",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
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
            (
                "clf",
                DecisionTreeClassifier(
                    max_depth=10,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
        "rf": Pipeline([
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=350,
                    max_depth=14,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]),
        "extra_trees": Pipeline([
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=350,
                    max_depth=14,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]),
        "hist_gb": Pipeline([
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=8,
                    learning_rate=0.08,
                    max_iter=250,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
    }


def save_confusion_matrix_png(cm: np.ndarray, labels: List[str], title: str, out_path: Path) -> None:
    """
    Save a readable confusion-matrix figure for a multiclass classifier.

    The figure size is made dynamic so that label names remain readable even as
    the number of classes grows.
    """
    n = len(labels)
    fig_w = max(8, 0.9 * n + 4)
    fig_h = max(6, 0.8 * n + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel("Predicted label", fontsize=14)
    ax.set_ylabel("True label", fontsize=14)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)

    # Use white text on dark cells and black text on light cells for legibility.
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison_png(df: pd.DataFrame, out_path: Path) -> None:
    """
    Save a grouped bar chart comparing multiclass evaluation metrics per model.

    Metrics used
    ------------
    - accuracy: overall correctness
    - macro_precision / macro_recall / macro_f1: class-balanced view
    - weighted_f1: class-frequency-weighted view
    """
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    x = np.arange(len(df))
    width = 0.16

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 2) * width, df[metric].values, width=width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"].tolist(), fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Multiclass activity model comparison", fontsize=17, pad=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_session_counts_png(session_counts: pd.Series, out_path: Path) -> None:
    """
    Visualise how many *sessions* exist for each activity label.

    Session counts matter more than raw window counts when grouped
    cross-validation is used, because sessions are the unit of independence.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(session_counts.index, session_counts.values)
    ax.set_title("Session counts by activity label", fontsize=17, pad=10)
    ax.set_xlabel("Activity label", fontsize=13)
    ax.set_ylabel("Number of sessions", fontsize=13)
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def print_input_summary(session_summary: pd.DataFrame, unknown_df: pd.DataFrame) -> None:
    """
    Print a concise summary of what entered the multiclass pipeline.

    This is useful for spotting issues such as:
    - labels that are missing entirely,
    - sessions skipped during preprocessing,
    - unexpectedly low window counts.
    """
    print("\n===== Data Summary =====")
    if session_summary.empty:
        print("No valid sessions entered the feature table.")
        return

    print(f"Sessions used: {len(session_summary)}")
    print(f"Total windows: {int(session_summary['n_windows'].sum())}")
    print("\nAll discovered labels in feature table:")
    print(session_summary["raw_label"].value_counts().sort_index().to_string())

    if not unknown_df.empty:
        print("\nSkipped / unknown file summary:")
        print(unknown_df["reason"].value_counts().to_string())


def evaluate_multiclass(
    feat_df: pd.DataFrame,
    output_dir: Path,
    min_sessions_per_label: int,
    n_splits: int,
    save_debug_csv: bool,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Evaluate multiple multiclass models using grouped cross-validation.

    Key design decision
    -------------------
    GroupKFold uses `session_id` as the grouping variable. This prevents windows
    from the same original recording session appearing in both training and test
    folds, which would otherwise inflate evaluation results.

    Returns
    -------
    results_df:
        One row per model with aggregate metrics.
    excluded_labels:
        Labels removed because they had too few sessions.
    label_order:
        Sorted class order used when building confusion matrices.
    """
    # Count sessions per raw activity label, not windows. This gives a more
    # honest view of how much independent data exists for each class.
    session_label_df = feat_df[["session_id", "raw_label"]].drop_duplicates()
    session_counts = session_label_df["raw_label"].value_counts().sort_index()
    save_session_counts_png(session_counts, output_dir / "multiclass_session_counts.png")

    print("\n===== Session Counts by Label =====")
    print(session_counts.to_string())

    # Labels with too few sessions can make grouped CV unstable or impossible.
    eligible_labels = session_counts[session_counts >= min_sessions_per_label].index.tolist()
    excluded_labels = session_counts[session_counts < min_sessions_per_label].index.tolist()

    filtered = feat_df[feat_df["raw_label"].isin(eligible_labels)].copy()
    if filtered.empty:
        raise RuntimeError("No labels left after min_sessions_per_label filtering.")

    # Build the design matrix and target vector.
    feature_cols = core.get_feature_columns(filtered)
    X = filtered[feature_cols]
    y = filtered["raw_label"].astype(str)
    groups = filtered["session_id"].astype(str)

    # The number of folds cannot exceed the number of unique sessions.
    unique_groups = groups.nunique()
    effective_splits = min(n_splits, unique_groups)
    if effective_splits < 2:
        raise RuntimeError("Not enough grouped sessions for cross-validation.")

    print(f"\nCross-validation: GroupKFold(n_splits={effective_splits})")
    print(f"Eligible labels: {eligible_labels}")
    if excluded_labels:
        print(f"Excluded labels: {excluded_labels}")

    models = get_multiclass_models()
    cv = GroupKFold(n_splits=effective_splits)
    label_order = sorted(eligible_labels)
    results = []

    # Evaluate each candidate model independently.
    for name, model in models.items():
        y_true_all: List[str] = []
        y_pred_all: List[str] = []

        for train_idx, test_idx in cv.split(X, y, groups=groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # clone(...) ensures each fold starts from a fresh, unfitted model.
            m = clone(model)
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)

            y_true_all.extend(y_test.tolist())
            y_pred_all.extend(y_pred.tolist())

        # Aggregate predictions across all folds to compute final metrics.
        cm = confusion_matrix(y_true_all, y_pred_all, labels=label_order)
        save_confusion_matrix_png(
            cm=cm,
            labels=label_order,
            title=f"{name} multiclass confusion matrix",
            out_path=output_dir / f"{name}_multiclass_confusion_matrix.png",
        )

        row = {
            "model": name,
            "accuracy": accuracy_score(y_true_all, y_pred_all),
            "macro_precision": precision_score(y_true_all, y_pred_all, average="macro", zero_division=0),
            "macro_recall": recall_score(y_true_all, y_pred_all, average="macro", zero_division=0),
            "macro_f1": f1_score(y_true_all, y_pred_all, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0),
            "n_eval_windows": len(y_true_all),
            "n_labels": len(label_order),
        }
        results.append(row)

        # Optional CSV export keeps machine-readable versions of the results.
        if save_debug_csv:
            pd.DataFrame(cm, index=label_order, columns=label_order).to_csv(
                output_dir / f"{name}_multiclass_confusion_matrix.csv"
            )

    # Rank models by macro F1 because it treats classes more evenly.
    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    save_model_comparison_png(results_df, output_dir / "multiclass_model_comparison.png")

    if save_debug_csv:
        results_df.to_csv(output_dir / "multiclass_model_comparison.csv", index=False)
        pd.DataFrame({"eligible_labels": pd.Series(label_order)}).to_csv(
            output_dir / "multiclass_labels_used.csv", index=False
        )
        if excluded_labels:
            pd.DataFrame({"excluded_labels": pd.Series(excluded_labels)}).to_csv(
                output_dir / "multiclass_labels_excluded.csv", index=False
            )

    print("\n===== Multiclass Model Comparison =====")
    print(results_df.to_string(index=False))
    return results_df, excluded_labels, label_order


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for multiclass analysis.

    The defaults are intentionally aligned with the shared project pipeline so
    the script can usually run without extra flags.
    """
    parser = argparse.ArgumentParser(
        description="Optimized multiclass activity analysis: keep PNG figures, print key stats to console."
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    # Label-filtering control. Useful when some activities have too few sessions.
    parser.add_argument("--min_sessions_per_label", type=int, default=1)

    # Maximum requested number of grouped CV folds.
    parser.add_argument("--n_splits", type=int, default=5)

    # Preprocessing parameters forwarded into the shared core module.
    parser.add_argument("--window_seconds", type=float, default=core.WINDOW_SECONDS)
    parser.add_argument("--overlap", type=float, default=core.OVERLAP)
    parser.add_argument("--resample_hz", type=float, default=core.RESAMPLE_HZ)
    parser.add_argument("--no_pressure", action="store_true")

    # When enabled, CSV diagnostics are also saved in addition to PNG figures.
    parser.add_argument("--save_debug_csv", action="store_true", help="Also save CSV debug outputs.")
    return parser.parse_args()


def main() -> None:
    """
    Run the full multiclass evaluation workflow.

    This function wires together the full script and prints a compact run
    summary suitable for terminal use.
    """
    args = parse_args()
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

    print_input_summary(session_summary, unknown_df)

    # We keep the returned values available so the script can be extended later
    # (for example, to save the best model or generate extra reports).
    results_df, excluded, used = evaluate_multiclass(
        feat_df=feat_df,
        output_dir=output_dir,
        min_sessions_per_label=args.min_sessions_per_label,
        n_splits=args.n_splits,
        save_debug_csv=args.save_debug_csv,
    )

    print("\nDone.")
    print(f"Data dir   : {data_dir}")
    print(f"Output dir : {output_dir}")
    print("\nSaved PNG files:")
    for path in sorted(output_dir.glob("*.png")):
        print(f" - {path.name}")
    if not args.save_debug_csv:
        print("Only PNG figures were kept in the output directory.")


if __name__ == "__main__":
    main()
