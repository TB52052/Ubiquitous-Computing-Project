from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import m2cgen as m2c
import pandas as pd
from sklearn.base import clone

import multiclass_analysis as mc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ExtraTrees from your ORIGINAL multiclass pipeline."
    )
    parser.add_argument("--data_dir", type=Path, default=Path("./organized_data"))
    parser.add_argument("--output_dir", type=Path, default=Path("./model_output_export"))
    parser.add_argument("--min_sessions_per_label", type=int, default=1)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--window_seconds", type=float, default=float(getattr(mc.core, "WINDOW_SECONDS", 4.0)))
    parser.add_argument("--overlap", type=float, default=float(getattr(mc.core, "OVERLAP", 0.5)))
    parser.add_argument("--resample_hz", type=float, default=float(getattr(mc.core, "RESAMPLE_HZ", 32.0)))
    parser.add_argument("--no_pressure", action="store_true")
    parser.add_argument("--save_debug_csv", action="store_true")
    parser.add_argument("--joblib_name", default="fall_model.pkl")
    parser.add_argument("--java_name", default="FallDetectionModel.java")
    return parser.parse_args()


def build_args_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        resample_hz=args.resample_hz,
        window_seconds=args.window_seconds,
        overlap=args.overlap,
        no_pressure=args.no_pressure,
    )


def export_java_with_metadata(model, feature_cols: list[str], class_labels: list[str], java_path: Path) -> None:
    clf = model.named_steps["clf"]
    java_code = m2c.export_to_java(clf, class_name="FallDetectionModel")

    labels_java = ", ".join([f'"{x}"' for x in class_labels])
    features_java = ", ".join([f'"{x}"' for x in feature_cols])

    insert_block = f'''
    public static final String[] CLASS_LABELS = new String[]{{{labels_java}}};
    public static final String[] FEATURE_ORDER = new String[]{{{features_java}}};
'''

    marker = "public class FallDetectionModel {"
    if marker in java_code:
        java_code = java_code.replace(marker, marker + insert_block, 1)

    java_path.write_text(java_code, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    # Use your ORIGINAL preprocessing pipeline exactly as multiclass_analysis does.
    mc.configure_core(build_args_namespace(args))
    feat_df, session_summary, unknown_df = mc.build_feature_table_in_temp(
        data_dir=data_dir,
        save_debug_csv=args.save_debug_csv,
        output_dir=output_dir,
    )

    mc.print_input_summary(session_summary, unknown_df)

    # Reproduce the same label filtering logic as the original script.
    session_label_df = feat_df[["session_id", "raw_label"]].drop_duplicates()
    session_counts = session_label_df["raw_label"].value_counts().sort_index()
    eligible_labels = session_counts[session_counts >= args.min_sessions_per_label].index.tolist()
    filtered = feat_df[feat_df["raw_label"].isin(eligible_labels)].copy()
    if filtered.empty:
        raise RuntimeError("No labels left after filtering.")

    feature_cols = mc.core.get_feature_columns(filtered)
    X = filtered[feature_cols]
    y = filtered["raw_label"].astype(str)
    class_labels = sorted(y.unique().tolist())

    # Train ONLY extra_trees using the original model definition.
    model = clone(mc.get_multiclass_models()["extra_trees"])
    model.fit(X, y)

    joblib_path = output_dir / args.joblib_name
    java_path = output_dir / args.java_name
    joblib.dump(model, joblib_path)
    export_java_with_metadata(model, feature_cols, class_labels, java_path)

    # Also save the same evaluation figures using the original evaluation code.
    mc.evaluate_multiclass(
        feat_df=feat_df,
        output_dir=output_dir,
        min_sessions_per_label=args.min_sessions_per_label,
        n_splits=args.n_splits,
        save_debug_csv=args.save_debug_csv,
    )

    pd.DataFrame({"feature_name": feature_cols}).to_csv(output_dir / "feature_order.csv", index=False)
    pd.DataFrame({"class_label": class_labels}).to_csv(output_dir / "class_labels.csv", index=False)

    print("\nExport done.")
    print(f"Joblib: {joblib_path}")
    print(f"Java  : {java_path}")


if __name__ == "__main__":
    main()
