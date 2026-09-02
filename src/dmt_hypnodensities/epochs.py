"""Build artifact-safe 30-second epochs and optional context groups."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .io import RecordingData, TrialData


@dataclass(frozen=True)
class EpochMetadata:
    """Time, experimental label and continuity boundary for one epoch."""

    epoch_in_block: int
    epoch_in_run: int
    continuous_run_id: str
    absolute_start: float
    absolute_end: float
    label_time: float
    experimental_label: str
    break_before: bool
    gap_before_seconds: float


@dataclass(frozen=True)
class ContinuousRun:
    """Strictly sample-continuous signal containing only complete epochs."""

    run_id: str
    trial_indices: tuple[int, ...]
    channels: tuple[str, ...]
    sampling_frequency: float
    start_time: float
    end_time: float
    signal: np.ndarray
    sample_times: np.ndarray
    epochs: np.ndarray
    discarded_samples: int

    @property
    def n_epochs(self) -> int:
        return int(self.epochs.shape[0])


@dataclass(frozen=True)
class EpochBlock:
    """Context group of artifact-safe epochs from one or more continuous runs."""

    recording_id: str
    block_id: str
    trial_indices: tuple[int, ...]
    channels: tuple[str, ...]
    sampling_frequency: float
    start_time: float
    end_time: float
    elapsed_duration_seconds: float
    signal_duration_seconds: float
    inter_trial_gaps_seconds: tuple[float, ...]
    signal: np.ndarray
    sample_times: np.ndarray
    epochs: np.ndarray
    epoch_metadata: tuple[EpochMetadata, ...]
    discarded_samples: int
    continuous_runs: tuple[ContinuousRun, ...]

    @property
    def n_epochs(self) -> int:
        return int(self.epochs.shape[0])


def _nonempty_trials(trials: Sequence[TrialData]) -> tuple[TrialData, ...]:
    return tuple(
        sorted(
            (trial for trial in trials if trial.times.size and trial.data.shape[0]),
            key=lambda trial: float(trial.times[0]),
        )
    )


def _strict_parts(
    trials: Sequence[TrialData],
    sampling_frequency: float,
) -> tuple[tuple[np.ndarray, np.ndarray, tuple[int, ...]], ...]:
    """Split on every missing sample and merge only truly adjacent trial pieces."""

    expected_step = 1.0 / sampling_frequency
    tolerance = max(expected_step * 0.1, 1e-7)
    pieces = []
    for trial in _nonempty_trials(trials):
        differences = np.diff(trial.times)
        breaks = np.flatnonzero(~np.isclose(differences, expected_step, rtol=0, atol=tolerance))
        boundaries = np.concatenate(([0], breaks + 1, [len(trial.times)]))
        for start, stop in itertools.pairwise(boundaries):
            if stop > start:
                pieces.append((trial.data[start:stop], trial.times[start:stop], (trial.index,)))

    runs: list[tuple[np.ndarray, np.ndarray, tuple[int, ...]]] = []
    for data, times, indices in pieces:
        if runs:
            previous_data, previous_times, previous_indices = runs[-1]
            step = float(times[0] - previous_times[-1])
            if np.isclose(step, expected_step, rtol=0, atol=tolerance):
                runs[-1] = (
                    np.concatenate((previous_data, data), axis=0),
                    np.concatenate((previous_times, times)),
                    tuple(dict.fromkeys((*previous_indices, *indices))),
                )
                continue
        runs.append((data, times, indices))
    return tuple(runs)


def _epoch_run(
    recording: RecordingData,
    run_number: int,
    data: np.ndarray,
    times: np.ndarray,
    trial_indices: tuple[int, ...],
    epoch_duration_seconds: float,
) -> ContinuousRun | None:
    samples_per_epoch_float = epoch_duration_seconds * recording.sampling_frequency
    samples_per_epoch = round(samples_per_epoch_float)
    if not np.isclose(samples_per_epoch, samples_per_epoch_float, rtol=0, atol=1e-6):
        raise ValueError("epoch_duration_seconds * sampling_frequency must be an integer.")
    n_epochs = len(times) // samples_per_epoch
    if n_epochs == 0:
        return None
    usable_samples = n_epochs * samples_per_epoch
    signal = data[:usable_samples]
    sample_times = times[:usable_samples]
    epochs = signal.reshape(n_epochs, samples_per_epoch, signal.shape[1]).transpose(0, 2, 1)
    return ContinuousRun(
        run_id=f"{recording.recording_id}__R{run_number:04d}",
        trial_indices=trial_indices,
        channels=recording.channels,
        sampling_frequency=recording.sampling_frequency,
        start_time=float(sample_times[0]),
        end_time=float(sample_times[-1]),
        signal=signal,
        sample_times=sample_times,
        epochs=epochs,
        discarded_samples=len(times) - usable_samples,
    )


def _effective_gap(previous: ContinuousRun, current: ContinuousRun) -> float:
    sample_step = 1.0 / previous.sampling_frequency
    return max(0.0, float(current.sample_times[0] - previous.sample_times[-1] - sample_step))


def _group_runs(
    runs: Sequence[ContinuousRun],
    gap_tolerance_seconds: float,
) -> tuple[tuple[ContinuousRun, ...], ...]:
    if not runs:
        return ()
    groups = []
    current = [runs[0]]
    for run in runs[1:]:
        if _effective_gap(current[-1], run) <= gap_tolerance_seconds:
            current.append(run)
        else:
            groups.append(tuple(current))
            current = [run]
    groups.append(tuple(current))
    return tuple(groups)


def _build_context_block(
    recording: RecordingData,
    block_number: int,
    runs: Sequence[ContinuousRun],
    label_time: Callable[[float], str],
) -> EpochBlock:
    epochs = np.concatenate([run.epochs for run in runs], axis=0)
    signal = np.concatenate([run.signal for run in runs], axis=0)
    sample_times = np.concatenate([run.sample_times for run in runs])
    metadata = []
    epoch_in_block = 0
    previous_run = None
    for run in runs:
        gap = np.nan if previous_run is None else _effective_gap(previous_run, run)
        for epoch_in_run in range(run.n_epochs):
            samples_per_epoch = run.epochs.shape[-1]
            start = epoch_in_run * samples_per_epoch
            stop = start + samples_per_epoch
            epoch_times = run.sample_times[start:stop]
            representative_time = float(np.median(epoch_times))
            metadata.append(
                EpochMetadata(
                    epoch_in_block=epoch_in_block,
                    epoch_in_run=epoch_in_run,
                    continuous_run_id=run.run_id,
                    absolute_start=float(epoch_times[0]),
                    absolute_end=float(epoch_times[-1]),
                    label_time=representative_time,
                    experimental_label=label_time(representative_time),
                    break_before=previous_run is not None and epoch_in_run == 0,
                    gap_before_seconds=gap if epoch_in_run == 0 else 0.0,
                )
            )
            epoch_in_block += 1
        previous_run = run
    gaps = tuple(
        _effective_gap(previous, current) for previous, current in itertools.pairwise(runs)
    )
    trial_indices = tuple(dict.fromkeys(index for run in runs for index in run.trial_indices))
    return EpochBlock(
        recording_id=recording.recording_id,
        block_id=f"{recording.recording_id}__B{block_number:03d}",
        trial_indices=trial_indices,
        channels=recording.channels,
        sampling_frequency=recording.sampling_frequency,
        start_time=float(runs[0].sample_times[0]),
        end_time=float(runs[-1].sample_times[-1]),
        elapsed_duration_seconds=float(runs[-1].sample_times[-1] - runs[0].sample_times[0]),
        signal_duration_seconds=epochs.shape[0] * epochs.shape[-1] / recording.sampling_frequency,
        inter_trial_gaps_seconds=gaps,
        signal=signal,
        sample_times=sample_times,
        epochs=epochs,
        epoch_metadata=tuple(metadata),
        discarded_samples=sum(run.discarded_samples for run in runs),
        continuous_runs=tuple(runs),
    )


def build_epoch_blocks(
    recording: RecordingData,
    gap_tolerance_seconds: float,
    epoch_duration_seconds: float,
    label_time: Callable[[float], str],
) -> tuple[EpochBlock, ...]:
    """Build clean epochs first, then group them only for temporal context."""

    if gap_tolerance_seconds < 0:
        raise ValueError("gap_tolerance_seconds cannot be negative.")
    if epoch_duration_seconds <= 0:
        raise ValueError("epoch_duration_seconds must be positive.")
    parts = _strict_parts(recording.trials, recording.sampling_frequency)
    runs = []
    for run_number, (data, times, trial_indices) in enumerate(parts):
        run = _epoch_run(
            recording,
            run_number,
            data,
            times,
            trial_indices,
            epoch_duration_seconds,
        )
        if run is not None:
            runs.append(run)
    return tuple(
        _build_context_block(recording, block_number, group, label_time)
        for block_number, group in enumerate(_group_runs(runs, gap_tolerance_seconds))
    )
