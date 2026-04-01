from __future__ import annotations

"""
Command-line entry point for dataset organisation.

This script is intentionally very small: it does not implement the heavy lifting
itself. Instead, it delegates the real work to another module imported as
`core`. The expected responsibilities of that lower-level module are:

1. Discover raw sensor sessions inside the input directory.
2. Decide which files belong to the same recording session.
3. Determine whether a session is complete or incomplete.
4. Copy or move sessions into a standard output folder structure.
5. Optionally write CSV manifests for later inspection.

Why keep this file thin?
------------------------
A thin CLI wrapper is easier to maintain because:
- the reusable logic lives in one place (`core`),
- the command-line interface stays clean and readable,
- future GUI / notebook / batch use can call the same core functions.

Typical usage
-------------
python organize_data.py --input_dir ./falling_data --output_dir ./organized_data --write_manifest

Important note
--------------
The import below assumes there is a separate implementation module available as
`organize_data` and that this file is being used as the command-line wrapper in
that project structure.
"""

import argparse
from pathlib import Path

import organize_data as core


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    """
    Build and return the command-line parser for the organiser script.

    Returns
    -------
    argparse.ArgumentParser
        A configured parser describing the available command-line options.

    Design intent
    -------------
    The script exposes only operational parameters here:
    - where to read raw data from,
    - where to write organised data to,
    - whether to copy or move files,
    - whether incomplete sessions should be excluded,
    - whether CSV manifests should be written.

    The actual interpretation of a "session" is deliberately left to the core
    module, because that is project-specific domain logic.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Optimized organizer: discover sensor sessions, organise them into a "
            "clean dataset layout, print a summary to the console, and optionally "
            "write CSV manifest files."
        )
    )

    # Root folder that contains the raw recordings collected from the phone.
    parser.add_argument("--input_dir", type=Path, default=Path("./falling_data"))

    # Target directory that will receive the normalised dataset structure.
    parser.add_argument("--output_dir", type=Path, default=Path("./organized_data"))

    # "copy" is safer during experimentation; "move" is cleaner once the
    # organisation rules are known to be correct.
    parser.add_argument("--mode", choices=["copy", "move"], default="copy")

    # When enabled, existing destination files may be replaced.
    parser.add_argument("--overwrite", action="store_true")

    # When enabled, only sessions that satisfy the core completeness rules will
    # be exported into the organised dataset.
    parser.add_argument("--complete_only", action="store_true")

    # Manifest files are optional because they are useful for auditing, but not
    # always needed during day-to-day runs.
    parser.add_argument(
        "--write_manifest",
        action="store_true",
        help="Also write manifest.csv and unknown_files.csv for bookkeeping.",
    )
    return parser


# -----------------------------------------------------------------------------
# Main execution flow
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Execute the dataset-organisation workflow.

    High-level flow
    ---------------
    1. Parse command-line arguments.
    2. Resolve input/output paths to absolute paths.
    3. Verify that the input directory exists.
    4. Ask the core module to discover sessions.
    5. Ask the core module to organise those sessions.
    6. Print a human-readable summary.
    7. Optionally write CSV manifest files.

    This function performs only orchestration. It intentionally avoids embedding
    dataset-specific logic so that the behaviour stays easy to reason about.
    """
    parser = build_argparser()
    args = parser.parse_args()

    # Resolve paths early so all downstream logging and file operations use
    # canonical absolute paths rather than mixed relative paths.
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    # Fail fast with a clear message if the raw-data root folder is missing.
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Delegate raw-file inspection and session grouping to the core module.
    # Expected return values:
    # - sessions: recognised recording sessions
    # - unknown_files: files that could not be classified or grouped reliably
    sessions, unknown_files = core.discover_sessions(input_dir)

    # Delegate the actual reorganisation step. The core module is expected to
    # create the output folder structure and copy/move files accordingly.
    core.organize_sessions(
        sessions=sessions,
        output_dir=output_dir,
        mode=args.mode,
        overwrite=args.overwrite,
        complete_only=args.complete_only,
    )

    # Print a console summary so the user can immediately see what was found
    # and whether any files were ignored or marked unknown.
    core.print_summary(sessions, unknown_files)
    print(f"\nOrganized output written to: {output_dir}")

    # Optionally export CSV bookkeeping files. These are useful when auditing
    # the dataset, debugging missed sessions, or preparing appendices/results.
    if args.write_manifest:
        core.write_manifest(
            sessions=sessions,
            unknown_files=unknown_files,
            manifest_path=output_dir / "manifest.csv",
        )
        print(f"Manifest written to: {output_dir / 'manifest.csv'}")
    else:
        print(
            "Manifest files were not written. "
            "Add --write_manifest if you later need CSV bookkeeping."
        )


# Standard Python script entry point.
if __name__ == "__main__":
    main()
