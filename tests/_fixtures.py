"""Shared synthetic fixtures for pipeline tests."""

from pathlib import Path

import numpy as np

from dmt_hypnodensities.epochs import build_epoch_blocks
from dmt_hypnodensities.io import RecordingData, TrialData


def make_block():
    sampling_frequency = 100.0
    samples = int(10 * 30 * sampling_frequency)
    times = np.arange(samples) / sampling_frequency
    signal = np.column_stack(
        (
            20e-6 * np.sin(2 * np.pi * 2 * times),
            20e-6 * np.sin(2 * np.pi * 10 * times),
        )
    )
    recording = RecordingData(
        path=Path("DMTCI_P07D1_inner.mat"),
        recording_id="DMTCI_P07D1_inner",
        subject="P07",
        session="D1",
        condition="placebo",
        sampling_frequency=sampling_frequency,
        channels=("E6", "E8"),
        trials=(TrialData(index=0, times=times, data=signal),),
    )
    return build_epoch_blocks(recording, 70.0, 30.0, lambda _: "before")[0]
