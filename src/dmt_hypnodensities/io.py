"""Read the external FieldTrip MATLAB/HDF5 recordings."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

_RECORDING_PATTERN = re.compile(r"(?P<subject>P\d+)(?P<session>D\d+)")


@dataclass(frozen=True)
class TrialData:
    """One cleaned FieldTrip trial with samples in ``time x channel`` order."""

    index: int
    times: np.ndarray
    data: np.ndarray


@dataclass(frozen=True)
class RecordingData:
    """Selected channels and trials from one source recording."""

    path: Path
    recording_id: str
    subject: str
    session: str
    condition: str
    sampling_frequency: float
    channels: tuple[str, ...]
    trials: tuple[TrialData, ...]


def _decode_matlab_label(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset)
    if values.dtype.kind in {"u", "i"}:
        raw = values.astype("<u2", copy=False).tobytes()
        return raw.decode("utf-16-le").rstrip("\x00")
    return values.tobytes().decode("utf-16-le").rstrip("\x00")


def _parse_recording_name(path: Path) -> tuple[str, str, str]:
    match = _RECORDING_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Cannot parse subject/session from {path.name!r}.")
    subject = match.group("subject")
    session = match.group("session")
    condition = "placebo" if session == "D1" else "DMT"
    return subject, session, condition


def _select_channel_indices(
    all_channels: Sequence[str], selected_channels: Sequence[str] | None
) -> tuple[tuple[str, ...], np.ndarray]:
    if selected_channels is None:
        channels = tuple(all_channels)
    else:
        channels = tuple(selected_channels)
        duplicates = sorted({channel for channel in channels if channels.count(channel) > 1})
        if duplicates:
            raise ValueError(f"Duplicate requested channels: {duplicates}")

    channel_to_index = {channel: index for index, channel in enumerate(all_channels)}
    missing = [channel for channel in channels if channel not in channel_to_index]
    if missing:
        raise ValueError(f"Requested channels are absent from the recording: {missing}")

    indices = np.asarray([channel_to_index[channel] for channel in channels], dtype=int)
    return channels, indices


def _read_data_reference(
    handle: h5py.File,
    reference: h5py.Reference,
    channel_indices: np.ndarray,
    n_all_channels: int,
    dtype: np.dtype,
) -> np.ndarray:
    dataset = handle[reference]
    if dataset.ndim != 2:
        raise ValueError(f"Expected 2-D trial data, found shape {dataset.shape}.")

    # h5py requires increasing fancy indices. Read in physical order, then restore
    # the exact order requested by the caller (e.g. ["E45", "E9"]).
    physical_order = np.argsort(channel_indices)
    sorted_indices = channel_indices[physical_order]
    requested_order = np.argsort(physical_order)

    if dataset.shape[1] == n_all_channels:
        data = np.asarray(dataset[:, sorted_indices], dtype=dtype)[:, requested_order]
    elif dataset.shape[0] == n_all_channels:
        data = np.asarray(dataset[sorted_indices, :], dtype=dtype)[requested_order, :].T
    else:
        raise ValueError(
            f"Neither trial dimension matches the {n_all_channels} channel labels: "
            f"shape={dataset.shape}."
        )
    return data


def _infer_sampling_frequency(trials: Sequence[TrialData]) -> float:
    steps = []
    for trial in trials:
        if trial.times.size < 2:
            continue
        differences = np.diff(trial.times)
        valid = differences[np.isfinite(differences) & (differences > 0)]
        if valid.size:
            steps.append(float(np.median(valid)))
    if not steps:
        raise ValueError("Cannot infer sampling frequency from the trial timestamps.")
    return 1.0 / float(np.median(np.asarray(steps)))


def _load_fieldtrip_recording(
    path: Path | str,
    selected_channels: Sequence[str] | None = None,
    expected_sampling_frequency: float | None = None,
    dtype: np.dtype = np.float32,
) -> RecordingData:
    recording_path = Path(path).expanduser().resolve()
    subject, session, condition = _parse_recording_name(recording_path)

    trials = []
    with h5py.File(recording_path, "r") as handle:
        group_name = "data_avref_inner" if "data_avref_inner" in handle else "data_avref"
        if group_name not in handle:
            raise KeyError("Neither 'data_avref_inner' nor 'data_avref' exists in the MAT file.")
        group = handle[group_name]

        label_references = np.asarray(group["label"][0]).reshape(-1)
        all_channels = tuple(
            _decode_matlab_label(handle[reference]) for reference in label_references
        )
        channels, channel_indices = _select_channel_indices(all_channels, selected_channels)

        trial_rows = group["trial"]
        time_rows = group["time"]
        if trial_rows.shape[0] != time_rows.shape[0]:
            raise ValueError("FieldTrip trial/time reference counts do not match.")

        for trial_index, (trial_row, time_row) in enumerate(zip(trial_rows, time_rows)):
            data_parts = []
            time_parts = []
            for data_reference, time_reference in zip(
                np.asarray(trial_row).reshape(-1), np.asarray(time_row).reshape(-1)
            ):
                data_parts.append(
                    _read_data_reference(
                        handle,
                        data_reference,
                        channel_indices,
                        len(all_channels),
                        np.dtype(dtype),
                    )
                )
                time_parts.append(np.asarray(handle[time_reference]).reshape(-1).astype(float))

            data = np.concatenate(data_parts, axis=0)
            times = np.concatenate(time_parts)
            if data.shape[0] != times.size:
                raise ValueError(
                    f"Trial {trial_index} has {data.shape[0]} samples but {times.size} timestamps."
                )
            trials.append(TrialData(index=trial_index, times=times, data=data))

    sampling_frequency = _infer_sampling_frequency(trials)
    if expected_sampling_frequency is not None and not np.isclose(
        sampling_frequency, expected_sampling_frequency, rtol=0, atol=1e-3
    ):
        raise ValueError(
            f"Inferred sampling frequency {sampling_frequency:.6f} Hz does not match "
            f"expected {expected_sampling_frequency:.6f} Hz."
        )

    return RecordingData(
        path=recording_path,
        recording_id=recording_path.stem,
        subject=subject,
        session=session,
        condition=condition,
        sampling_frequency=sampling_frequency,
        channels=channels,
        trials=tuple(trials),
    )


def load_fieldtrip_recording(
    path: Path | str,
    selected_channels: Sequence[str] | None = None,
    expected_sampling_frequency: float | None = None,
    dtype: np.dtype = np.float32,
) -> RecordingData:
    """Load FieldTrip data with channel names mapped to their true source indices."""

    return _load_fieldtrip_recording(
        path,
        selected_channels=selected_channels,
        expected_sampling_frequency=expected_sampling_frequency,
        dtype=dtype,
    )


def read_fieldtrip_channels(path: Path | str) -> tuple[str, ...]:
    """Read only channel labels, without loading any EEG trial arrays."""

    recording_path = Path(path).expanduser().resolve()
    with h5py.File(recording_path, "r") as handle:
        group_name = "data_avref_inner" if "data_avref_inner" in handle else "data_avref"
        if group_name not in handle:
            raise KeyError("Neither 'data_avref_inner' nor 'data_avref' exists in the MAT file.")
        references = np.asarray(handle[group_name]["label"][0]).reshape(-1)
        return tuple(_decode_matlab_label(handle[reference]) for reference in references)
