#!/usr/bin/env python3
"""Run the complete persisted batch independently for several gap tolerances."""

from __future__ import annotations

import argparse
import gc
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "analysis.yaml"
DEFAULT_TOLERANCES = (0.0, 15.0, 30.0, 60.0, 90.0, 120.0)


def _tolerance_label(tolerance: float) -> str:
    """Return a filesystem-safe, unambiguous tolerance label."""

    if tolerance < 0:
        raise ValueError("Gap tolerances cannot be negative.")
    if tolerance.is_integer():
        return f"{int(tolerance):03d}s"
    return f"{tolerance:g}s".replace(".", "p")


def _run_name(prefix: str, tolerance: float) -> str:
    return f"{prefix}_{_tolerance_label(tolerance)}_gssc_yasa_sleepfm_cpu_v1"


def _run_one(config_path: Path, tolerance: float, prefix: str) -> None:
    """Execute the same extraction and QC workflow as notebook 01."""

    import pandas as pd

    from dmt_hypnodensities import (
        assemble_outputs,
        finalize_run,
        load_config,
        prepare_run,
        run_batch,
        save_table,
    )

    base_config = load_config(config_path)
    config = replace(base_config, gap_tolerance_seconds=tolerance)
    if set(config.stagers) != {"gssc", "yasa", "sleepfm"}:
        raise ValueError("The sweep requires exactly GSSC, YASA and SleepFM.")
    if config.compute_device != "cpu":
        raise ValueError("The sweep configuration must use compute.device: cpu.")
    if config.file_selection_policy != "prefer_d5":
        raise ValueError("The sweep requires file_selection_policy: prefer_d5.")

    name = _run_name(prefix, tolerance)
    run = prepare_run(config, name, reuse_existing=True)
    print(f"\n[{name}] Starting complete batch in {run.root}", flush=True)

    batch_summary = run_batch(run.config, reuse_completed=True)
    finalize_run(run, batch_summary)

    tables = assemble_outputs(run.recordings, strict=True)
    staging_qc_summary = (
        tables.staging_qc.groupby(["stager", "status"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    block_qc_summary = (
        tables.blocks.groupby(["condition"], dropna=False)
        .agg(
            recordings=("recording_id", "nunique"),
            blocks=("block_id", "nunique"),
            epochs=("n_epochs", "sum"),
        )
        .reset_index()
    )
    cardinality = {
        "gap_tolerance_seconds": tolerance,
        "selected_recordings": len(tables.file_selection),
        "successful_recordings": int(tables.batch_summary["status"].eq("ok").sum()),
        "blocks": len(tables.blocks),
        "feature_rows": len(tables.features),
        "hypnodensity_rows": len(tables.hypnodensities),
        "spectrum_rows": len(tables.spectra),
    }
    save_table(staging_qc_summary, run.tables / "staging_qc_summary.csv")
    save_table(block_qc_summary, run.tables / "block_qc_summary.csv")
    save_table(pd.DataFrame([cardinality]), run.tables / "run_cardinality.csv")
    print(f"[{name}] Completed: {cardinality}", flush=True)

    # This also helps when _run_one is invoked directly in tests or interactively.
    # In the normal sweep the whole child process exits immediately afterwards,
    # which is what guarantees release of native and joblib-worker memory.
    del tables, batch_summary, staging_qc_summary, block_qc_summary, run, config, base_config
    gc.collect()


def _run_sweep(
    config_path: Path,
    tolerances: tuple[float, ...],
    prefix: str,
) -> None:
    """Launch one fresh Python process per tolerance, sequentially."""

    for index, tolerance in enumerate(tolerances, start=1):
        name = _run_name(prefix, tolerance)
        print(
            f"\n=== Tolerance {index}/{len(tolerances)}: {tolerance:g} s ({name}) ===",
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config",
                str(config_path),
                "--run-prefix",
                prefix,
                "--single-tolerance",
                str(tolerance),
            ],
            cwd=REPOSITORY_ROOT,
            check=True,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the notebook-01 batch independently for each gap tolerance. "
            "Every tolerance has its own immutable output directory."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Base analysis YAML (default: configs/analysis.yaml).",
    )
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=DEFAULT_TOLERANCES,
        help="Gap tolerances in seconds for the sweep.",
    )
    parser.add_argument(
        "--run-prefix",
        default="gap_sensitivity",
        help="Prefix for the independent run directories.",
    )
    parser.add_argument(
        "--single-tolerance",
        type=float,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    if args.single_tolerance is not None:
        _run_one(config_path, float(args.single_tolerance), args.run_prefix)
        return
    tolerances = tuple(float(value) for value in args.tolerances)
    if len(set(tolerances)) != len(tolerances):
        raise ValueError("Duplicate gap tolerances would target the same analysis twice.")
    for tolerance in tolerances:
        _tolerance_label(tolerance)
    _run_sweep(config_path, tolerances, args.run_prefix)


if __name__ == "__main__":
    main()
