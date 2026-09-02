"""Small orchestration layer shared by scripts and notebooks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .channels import load_electrode_mapping, select_analysis_electrodes
from .config import AnalysisConfig
from .epochs import build_epoch_blocks
from .features import extract_block_features
from .io import RecordingData, load_fieldtrip_recording, read_fieldtrip_channels
from .sleepfm import LocalSleepFMPredictor
from .staging import SleepFMPredictor, stage_block


@dataclass(frozen=True)
class RecordingResult:
    """All non-signal outputs generated for one recording."""

    features: pd.DataFrame
    hypnodensities: pd.DataFrame
    spectra: pd.DataFrame
    blocks: pd.DataFrame
    staging_qc: pd.DataFrame


def process_recording(
    path: Path | str,
    config: AnalysisConfig,
    channels: Sequence[str] | None = None,
    analyses: Sequence[str] | None = None,
    stagers: Sequence[str] | None = None,
    n_jobs: int | None = None,
    sleepfm_predictor: SleepFMPredictor | None = None,
    sleepfm_channel_sets: Sequence[Sequence[str]] | None = None,
    show_progress: bool = False,
) -> RecordingResult:
    """Load once, construct blocks once, then run staging and feature analyses."""

    electrode_to_1020: Mapping[str, str] = {}
    if channels is None:
        if config.electrode_mapping_path is None:
            raise ValueError(
                "Automatic channel selection requires data.electrode_mapping in config."
            )
        mapping = load_electrode_mapping(config.electrode_mapping_path)
        selection = select_analysis_electrodes(mapping, read_fieldtrip_channels(path))
        channels = selection.channels
        electrode_to_1020 = selection.electrode_to_1020
    elif config.electrode_mapping_path is not None:
        mapping = load_electrode_mapping(config.electrode_mapping_path)
        electrode_to_1020 = dict(zip(mapping["EGI-257 Label"], mapping["10-20 Label"]))

    recording = load_fieldtrip_recording(
        path,
        selected_channels=channels,
        expected_sampling_frequency=config.sampling_frequency,
    )
    blocks = build_epoch_blocks(
        recording,
        gap_tolerance_seconds=config.gap_tolerance_seconds,
        epoch_duration_seconds=config.epoch_duration_seconds,
        label_time=config.label_time,
    )

    requested = tuple(config.feature_analyses if analyses is None else analyses)
    requested_stagers = tuple(config.stagers if stagers is None else stagers)
    if (
        "sleepfm" in requested_stagers
        and sleepfm_predictor is None
        and config.sleepfm_repository is not None
    ):
        sleepfm_predictor = LocalSleepFMPredictor(
            config.sleepfm_repository,
            device=config.compute_device,
        )
    requested_sleepfm_sets = (
        config.sleepfm_channel_sets if sleepfm_channel_sets is None else sleepfm_channel_sets
    )
    workers = config.n_jobs if n_jobs is None else n_jobs
    feature_tables = []
    hypnodensity_tables = []
    spectrum_tables = []
    staging_qc_tables = []
    block_rows = []
    block_iterator = tqdm(
        blocks,
        desc=f"{recording.recording_id} · blocks",
        unit="block",
        leave=False,
        disable=not show_progress,
    )
    for block in block_iterator:
        staging = stage_block(
            block,
            requested_stagers,
            n_jobs=workers,
            min_epochs=dict(config.stager_min_epochs),
            sleepfm_predictor=sleepfm_predictor,
            sleepfm_channel_sets=requested_sleepfm_sets,
            device=config.compute_device,
        )
        if not staging.hypnodensities.empty:
            hypnodensity_tables.append(staging.hypnodensities)
        if not staging.qc.empty:
            staging_qc_tables.append(staging.qc)

        yasa_features = staging.yasa_features if "yasa" in requested_stagers else None
        if yasa_features is not None:
            statuses = set(staging.qc.loc[staging.qc["stager"] == "yasa", "status"])
            yasa_features.attrs["status"] = statuses.pop() if len(statuses) == 1 else "partial"
        result = extract_block_features(
            block,
            requested,
            n_jobs=workers,
            precomputed_yasa_features=yasa_features,
        )
        feature_tables.append(result.features)
        if not result.spectra.empty:
            spectrum_tables.append(result.spectra)
        block_rows.append(
            {
                "recording_id": block.recording_id,
                "block_id": block.block_id,
                "start_time": block.start_time,
                "end_time": block.end_time,
                "elapsed_duration_seconds": block.elapsed_duration_seconds,
                "signal_duration_seconds": block.signal_duration_seconds,
                "n_trials": len(block.trial_indices),
                "n_epoch_runs": len(block.continuous_runs),
                "max_context_gap_seconds": (
                    max(block.inter_trial_gaps_seconds)
                    if block.inter_trial_gaps_seconds
                    else 0.0
                ),
                "n_epochs": block.n_epochs,
                "discarded_tail_samples": block.discarded_samples,
                **result.qc,
            }
        )

    result = RecordingResult(
        features=pd.concat(feature_tables, ignore_index=True) if feature_tables else pd.DataFrame(),
        hypnodensities=pd.concat(hypnodensity_tables, ignore_index=True)
        if hypnodensity_tables
        else pd.DataFrame(),
        spectra=pd.concat(spectrum_tables, ignore_index=True)
        if spectrum_tables
        else pd.DataFrame(),
        blocks=pd.DataFrame(block_rows),
        staging_qc=pd.concat(staging_qc_tables, ignore_index=True)
        if staging_qc_tables
        else pd.DataFrame(),
    )
    result = _attach_recording_metadata(result, recording)
    return _attach_electrode_labels(result, electrode_to_1020)


def _attach_recording_metadata(
    result: RecordingResult,
    recording: RecordingData,
) -> RecordingResult:
    """Add study identifiers to every output table without duplicating signal data."""

    def labelled(table: pd.DataFrame) -> pd.DataFrame:
        if table.empty:
            return table
        output = table.copy()
        position = output.columns.get_loc("recording_id") + 1
        for name, value in reversed(
            (
                ("subject", recording.subject),
                ("session", recording.session),
                ("condition", recording.condition),
            )
        ):
            if name not in output:
                output.insert(position, name, value)
        return output

    return RecordingResult(
        features=labelled(result.features),
        hypnodensities=labelled(result.hypnodensities),
        spectra=labelled(result.spectra),
        blocks=labelled(result.blocks),
        staging_qc=labelled(result.staging_qc),
    )


def _attach_electrode_labels(
    result: RecordingResult,
    electrode_to_1020: Mapping[str, str],
) -> RecordingResult:
    """Attach the notebook-compatible standard label without replacing EGI identity."""

    def labelled(table: pd.DataFrame) -> pd.DataFrame:
        if table.empty or not electrode_to_1020:
            return table
        output = table.copy()
        if "electrode" in output:
            position = output.columns.get_loc("electrode") + 1
            output.insert(position, "el_10-20", output["electrode"].map(electrode_to_1020))
        if "channel_set" in output:
            output["channel_set_10-20"] = output["channel_set"].map(
                lambda value: "+".join(
                    electrode_to_1020.get(item, item) for item in str(value).split("+")
                )
            )
        return output

    return RecordingResult(
        features=labelled(result.features),
        hypnodensities=labelled(result.hypnodensities),
        spectra=labelled(result.spectra),
        blocks=result.blocks,
        staging_qc=labelled(result.staging_qc),
    )
