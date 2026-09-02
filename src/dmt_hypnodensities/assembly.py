"""Assemble persisted per-recording outputs into analysis-ready tables."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class AnalysisTables:
    """Validated study-level tables; no signals or epoch tensors are included."""

    features: pd.DataFrame
    hypnodensities: pd.DataFrame
    spectra: pd.DataFrame
    blocks: pd.DataFrame
    staging_qc: pd.DataFrame
    batch_summary: pd.DataFrame
    file_selection: pd.DataFrame


EPOCH_METADATA_COLUMNS = frozenset(
    {
        "recording_id",
        "subject",
        "session",
        "condition",
        "block_id",
        "epoch",
        "continuous_run_id",
        "epoch_in_run",
        "break_before",
        "gap_before_seconds",
        "electrode",
        "el_10-20",
        "absolute_start",
        "absolute_end",
        "label_time",
        "experimental_label",
    }
)


def _read_optional(path: Path, parquet: bool = False) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path) if parquet else pd.read_csv(path)


def _recording_names(output: Path, names: Sequence[str] | None) -> tuple[str, ...]:
    if names is not None:
        return tuple(dict.fromkeys(Path(name).stem for name in names))
    summary = _read_optional(output / "batch_summary.csv")
    if summary.empty or not {"recording", "status"}.issubset(summary):
        raise FileNotFoundError(
            "Automatic assembly requires batch_summary.csv; alternatively pass recording_names."
        )
    return tuple(Path(name).stem for name in summary.loc[summary["status"].eq("ok"), "recording"])


def _concatenate(tables: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [table for table in tables if not table.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def _validate_unique(table: pd.DataFrame, keys: Sequence[str], label: str) -> None:
    if table.empty:
        return
    missing = [key for key in keys if key not in table]
    if missing:
        raise ValueError(f"{label} is missing key columns: {missing}")
    duplicated = table.duplicated(list(keys), keep=False)
    if duplicated.any():
        raise ValueError(f"{label} contains {int(duplicated.sum())} rows with duplicate keys.")


def assemble_outputs(
    output_directory: Path | str,
    recording_names: Sequence[str] | None = None,
    strict: bool = True,
) -> AnalysisTables:
    """Load canonical outputs and validate their cardinalities and block references."""

    output = Path(output_directory).expanduser().resolve()
    names = _recording_names(output, recording_names)
    feature_tables = []
    hypnodensity_tables = []
    spectrum_tables = []
    block_tables = []
    qc_tables = []
    for name in names:
        feature_path = output / f"{name}_features.parquet"
        block_path = output / f"{name}_blocks.csv"
        if strict and (not feature_path.is_file() or not block_path.is_file()):
            raise FileNotFoundError(f"Incomplete canonical outputs for {name!r}.")
        feature_tables.append(_read_optional(feature_path, parquet=True))
        hypnodensity_tables.append(
            _read_optional(output / f"{name}_hypnodensities.parquet", parquet=True)
        )
        spectrum_tables.append(_read_optional(output / f"{name}_spectra.parquet", parquet=True))
        block_tables.append(_read_optional(block_path))
        qc_tables.append(_read_optional(output / f"{name}_staging_qc.csv"))

    tables = AnalysisTables(
        features=_concatenate(feature_tables),
        hypnodensities=_concatenate(hypnodensity_tables),
        spectra=_concatenate(spectrum_tables),
        blocks=_concatenate(block_tables),
        staging_qc=_concatenate(qc_tables),
        batch_summary=_read_optional(output / "batch_summary.csv"),
        file_selection=_read_optional(output / "file_selection.csv"),
    )
    _validate_unique(
        tables.features,
        ("recording_id", "block_id", "epoch", "electrode"),
        "features",
    )
    _validate_unique(
        tables.hypnodensities,
        ("recording_id", "block_id", "epoch", "stager", "channel_set"),
        "hypnodensities",
    )
    _validate_unique(tables.blocks, ("recording_id", "block_id"), "blocks")
    _validate_unique(
        tables.spectra,
        ("recording_id", "block_id", "epoch", "electrode", "frequency_hz"),
        "spectra",
    )
    if not tables.hypnodensities.empty and not tables.blocks.empty:
        known_blocks = set(zip(tables.blocks["recording_id"], tables.blocks["block_id"]))
        staging_blocks = set(
            zip(
                tables.hypnodensities["recording_id"],
                tables.hypnodensities["block_id"],
            )
        )
        unknown = staging_blocks - known_blocks
        if unknown:
            raise ValueError(f"Hypnodensities reference unknown blocks: {sorted(unknown)[:5]}")
    return tables


def join_epoch_features_hypnodensities(
    features: pd.DataFrame,
    hypnodensities: pd.DataFrame,
    strict: bool = True,
) -> pd.DataFrame:
    """Attach monoelectrode features to aligned staging rows without duplication.

    Multichannel SleepFM rows have no unique feature electrode and are intentionally
    excluded. They remain available in the canonical hypnodensity table for analyses
    with an explicitly defined multichannel feature aggregation.
    """

    keys = ("recording_id", "block_id", "epoch", "electrode")
    _validate_unique(features, keys, "features")
    required = (*keys, "stager", "channel_set")
    missing = [key for key in required if key not in hypnodensities]
    if missing:
        raise ValueError(f"hypnodensities is missing key columns: {missing}")
    mono = hypnodensities.loc[hypnodensities["electrode"].notna()].copy()
    _validate_unique(mono, (*keys, "stager", "channel_set"), "monoelectrode hypnodensities")
    feature_columns = [*keys, *(column for column in features if column not in hypnodensities)]
    joined = mono.merge(
        features[feature_columns],
        on=list(keys),
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    unmatched = joined["_merge"].ne("both")
    if strict and unmatched.any():
        example = joined.loc[unmatched, list(keys)].head().to_dict("records")
        raise ValueError(
            f"{int(unmatched.sum())} staging rows have no aligned feature row; examples: {example}"
        )
    return joined.drop(columns="_merge")


def feature_value_columns(features: pd.DataFrame) -> tuple[str, ...]:
    """Return numeric extracted-feature columns, excluding epoch metadata."""

    return tuple(
        column
        for column in features.select_dtypes(include="number").columns
        if column not in EPOCH_METADATA_COLUMNS
    )
