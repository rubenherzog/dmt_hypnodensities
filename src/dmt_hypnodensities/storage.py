"""Persist tabular outputs; raw signals and epoch tensors are never written."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .pipeline import RecordingResult


def _write_parquet_atomic(table, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_parquet(temporary, index=False)
    temporary.replace(path)


def _write_csv_atomic(table, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def save_table(table: pd.DataFrame, path: Path | str) -> Path:
    """Atomically save an analysis table as CSV or Parquet based on its suffix."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".csv":
        _write_csv_atomic(table, destination)
    elif destination.suffix == ".parquet":
        _write_parquet_atomic(table, destination)
    else:
        raise ValueError("Analysis tables must use a .csv or .parquet suffix.")
    return destination


def save_recording_result(
    result: RecordingResult,
    output_directory: Path | str,
    name: str,
) -> dict[str, Path]:
    """Save features, hypnodensities, spectra and compact QC tables."""

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    written = {}

    feature_path = output / f"{name}_features.parquet"
    _write_parquet_atomic(result.features, feature_path)
    written["features"] = feature_path

    if not result.hypnodensities.empty:
        staging_path = output / f"{name}_hypnodensities.parquet"
        _write_parquet_atomic(result.hypnodensities, staging_path)
        written["hypnodensities"] = staging_path

    if not result.spectra.empty:
        spectrum_path = output / f"{name}_spectra.parquet"
        _write_parquet_atomic(result.spectra, spectrum_path)
        written["spectra"] = spectrum_path

    blocks_path = output / f"{name}_blocks.csv"
    _write_csv_atomic(result.blocks, blocks_path)
    written["blocks"] = blocks_path

    if not result.staging_qc.empty:
        staging_qc_path = output / f"{name}_staging_qc.csv"
        _write_csv_atomic(result.staging_qc, staging_qc_path)
        written["staging_qc"] = staging_qc_path
    return written
