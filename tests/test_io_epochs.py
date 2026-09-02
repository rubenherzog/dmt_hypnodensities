"""Synthetic regression tests for FieldTrip reading and block construction."""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from dmt_hypnodensities.config import AnalysisConfig, ExperimentalWindow
from dmt_hypnodensities.epochs import build_epoch_blocks
from dmt_hypnodensities.io import RecordingData, TrialData, load_fieldtrip_recording

SFREQ = 500.0
EPOCH_SAMPLES = int(30 * SFREQ)


def _utf16_values(label: str) -> np.ndarray:
    return np.frombuffer(label.encode("utf-16-le"), dtype="<u2").reshape(-1, 1)


def _make_fieldtrip_fixture(path: Path) -> None:
    channels = ["E1", "E9", "E45"]
    starts = [0.0, 99.0, 200.0]  # Inter-trial gaps are ~69 s and ~71 s.

    with h5py.File(path, "w") as handle:
        group = handle.create_group("data_avref_inner")

        label_references = np.empty((1, len(channels)), dtype=h5py.ref_dtype)
        for index, channel in enumerate(channels):
            label_dataset = handle.create_dataset(f"label_{index}", data=_utf16_values(channel))
            label_references[0, index] = label_dataset.ref
        group.create_dataset("label", data=label_references)

        trial_references = np.empty((len(starts), 1), dtype=h5py.ref_dtype)
        time_references = np.empty((len(starts), 1), dtype=h5py.ref_dtype)
        for trial_index, start in enumerate(starts):
            times = start + np.arange(EPOCH_SAMPLES, dtype=float) / SFREQ
            data = np.column_stack(
                [
                    np.full(EPOCH_SAMPLES, 10 + trial_index),
                    np.full(EPOCH_SAMPLES, 20 + trial_index),
                    np.full(EPOCH_SAMPLES, 30 + trial_index),
                ]
            )
            data_dataset = handle.create_dataset(f"data_{trial_index}", data=data)
            time_dataset = handle.create_dataset(f"time_{trial_index}", data=times.reshape(-1, 1))
            trial_references[trial_index, 0] = data_dataset.ref
            time_references[trial_index, 0] = time_dataset.ref

        group.create_dataset("trial", data=trial_references)
        group.create_dataset("time", data=time_references)


class FieldTripAndEpochTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_directory.name) / "DMTCI_P07D1_inner.mat"
        _make_fieldtrip_fixture(self.path)

        self.config = AnalysisConfig(
            raw_dir=Path(self.temporary_directory.name),
            output_dir=Path(self.temporary_directory.name) / "outputs",
            sampling_frequency=SFREQ,
            epoch_duration_seconds=30.0,
            gap_tolerance_seconds=70.0,
            experimental_windows=(
                ExperimentalWindow("before", 0.0, 60.0),
                ExperimentalWindow("after", 60.0, 150.0),
                ExperimentalWindow("late", 150.0, 260.0),
            ),
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_channel_indices_are_resolved_before_filtering(self) -> None:
        recording = load_fieldtrip_recording(
            self.path,
            selected_channels=["E45", "E9"],
            expected_sampling_frequency=SFREQ,
        )

        self.assertEqual(recording.channels, ("E45", "E9"))
        np.testing.assert_array_equal(recording.trials[0].data[0], [30.0, 20.0])

    def test_blocks_use_70_second_scientific_gap_criterion(self) -> None:
        recording = load_fieldtrip_recording(
            self.path,
            selected_channels=["E45"],
            expected_sampling_frequency=SFREQ,
        )
        blocks = build_epoch_blocks(
            recording,
            gap_tolerance_seconds=self.config.gap_tolerance_seconds,
            epoch_duration_seconds=self.config.epoch_duration_seconds,
            label_time=self.config.label_time,
        )

        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].trial_indices, (0, 1))
        self.assertEqual(blocks[0].n_epochs, 2)
        self.assertEqual(blocks[1].trial_indices, (2,))
        self.assertEqual(blocks[1].n_epochs, 1)

    def test_experimental_boundary_does_not_split_a_block(self) -> None:
        recording = load_fieldtrip_recording(
            self.path,
            selected_channels=["E45"],
            expected_sampling_frequency=SFREQ,
        )
        blocks = build_epoch_blocks(
            recording,
            gap_tolerance_seconds=70.0,
            epoch_duration_seconds=30.0,
            label_time=self.config.label_time,
        )

        labels = [item.experimental_label for item in blocks[0].epoch_metadata]
        self.assertEqual(labels, ["before", "after"])

    def test_times_outside_windows_use_public_outside_label(self) -> None:
        self.assertEqual(self.config.label_time(-1.0), "outside")

    def test_epochs_never_cross_discontinuous_runs(self) -> None:
        sampling_frequency = 10.0
        first_times = np.arange(650) / sampling_frequency
        second_times = 75.0 + np.arange(800) / sampling_frequency
        recording = RecordingData(
            path=Path("DMTCI_P07D1_inner.mat"),
            recording_id="DMTCI_P07D1_inner",
            subject="P07",
            session="D1",
            condition="placebo",
            sampling_frequency=sampling_frequency,
            channels=("E6",),
            trials=(
                TrialData(0, first_times, np.zeros((len(first_times), 1))),
                TrialData(1, second_times, np.ones((len(second_times), 1))),
            ),
        )

        blocks = build_epoch_blocks(recording, 90.0, 30.0, lambda _: "after")

        self.assertEqual(len(blocks), 1)
        block = blocks[0]
        self.assertEqual(block.n_epochs, 4)
        self.assertEqual(len(block.continuous_runs), 2)
        self.assertTrue(np.all(block.epochs[:2] == 0))
        self.assertTrue(np.all(block.epochs[2:] == 1))
        self.assertEqual(
            [item.break_before for item in block.epoch_metadata],
            [False, False, True, False],
        )
        self.assertAlmostEqual(block.epoch_metadata[2].gap_before_seconds, 15.0)
        for item in block.epoch_metadata:
            self.assertAlmostEqual(item.absolute_end - item.absolute_start, 29.9)

        split = build_epoch_blocks(recording, 10.0, 30.0, lambda _: "after")
        self.assertEqual(len(split), 2)


if __name__ == "__main__":
    unittest.main()
