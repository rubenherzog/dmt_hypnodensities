"""Tests for the shared staging schema and adapter behavior."""

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from _fixtures import make_block

from dmt_hypnodensities.epochs import build_epoch_blocks
from dmt_hypnodensities.io import RecordingData, TrialData
from dmt_hypnodensities.staging import stage_block


class StagingTests(unittest.TestCase):
    def test_yasa_returns_probabilities_and_reusable_base_features(self) -> None:
        try:
            import yasa  # noqa: F401
        except ImportError:
            self.skipTest("The optional YASA dependency is not installed.")

        block = make_block()
        result = stage_block(block, ("yasa",), n_jobs=1)

        self.assertEqual(len(result.hypnodensities), block.n_epochs * len(block.channels))
        probability_columns = ["prob_W", "prob_N1", "prob_N2", "prob_N3", "prob_R"]
        np.testing.assert_allclose(
            result.hypnodensities[probability_columns].sum(axis=1), 1.0, atol=1e-6
        )
        self.assertEqual(
            len([column for column in result.yasa_features if column.startswith("yasa_")]),
            21,
        )
        self.assertEqual(set(result.qc["status"]), {"ok"})

    def test_discontinuity_aware_yasa_matches_official_yasa_on_continuous_data(self) -> None:
        try:
            import yasa
        except ImportError:
            self.skipTest("The optional YASA dependency is not installed.")
        from dmt_hypnodensities.staging import _make_raw

        block = make_block()
        actual = stage_block(block, ("yasa",), n_jobs=1).hypnodensities
        raw = _make_raw(block.signal[:, [0]], ("E6",), block.sampling_frequency)
        expected = yasa.SleepStaging(raw, eeg_name="E6").predict().proba
        expected = expected.rename(columns={"WAKE": "prob_W", "REM": "prob_R"})
        expected = expected.rename(
            columns={stage: f"prob_{stage}" for stage in ("N1", "N2", "N3")}
        )
        observed = actual.loc[actual["electrode"].eq("E6")].sort_values("epoch")

        np.testing.assert_allclose(
            observed[["prob_W", "prob_N1", "prob_N2", "prob_N3", "prob_R"]],
            expected[["prob_W", "prob_N1", "prob_N2", "prob_N3", "prob_R"]],
            atol=1e-7,
        )

    def test_single_epoch_yasa_is_reported_without_failing_the_block(self) -> None:
        block = make_block()
        samples_per_epoch = block.epochs.shape[-1]
        one_epoch = replace(
            block,
            signal=block.signal[:samples_per_epoch],
            sample_times=block.sample_times[:samples_per_epoch],
            epochs=block.epochs[:1],
            epoch_metadata=block.epoch_metadata[:1],
        )

        result = stage_block(one_epoch, ("yasa",), n_jobs=1)

        self.assertTrue(result.hypnodensities.empty)
        self.assertEqual(set(result.qc["status"]), {"unsupported_single_epoch"})

    def test_yasa_uses_context_across_single_epoch_continuous_runs(self) -> None:
        try:
            import yasa  # noqa: F401
        except ImportError:
            self.skipTest("The optional YASA dependency is not installed.")

        sampling_frequency = 100.0
        trials = []
        for index in range(4):
            start = index * 35.0
            times = start + np.arange(3000) / sampling_frequency
            signal = (20e-6 * np.sin(2 * np.pi * 8 * times))[:, np.newaxis]
            trials.append(TrialData(index=index, times=times, data=signal))
        recording = RecordingData(
            path=Path("DMTCI_P07D1_inner.mat"),
            recording_id="DMTCI_P07D1_inner",
            subject="P07",
            session="D1",
            condition="placebo",
            sampling_frequency=sampling_frequency,
            channels=("E6",),
            trials=tuple(trials),
        )
        block = build_epoch_blocks(recording, 90.0, 30.0, lambda _: "after")[0]

        result = stage_block(block, ("yasa",), n_jobs=1, min_epochs={"yasa": 2})

        self.assertEqual(len(block.continuous_runs), 4)
        self.assertEqual(len(result.hypnodensities), 4)
        self.assertEqual(set(result.qc["status"]), {"ok"})

    def test_sleepfm_adapter_supports_explicit_multichannel_sets(self) -> None:
        block = make_block()

        def predictor(received_block, channel_set):
            self.assertIs(received_block, block)
            probabilities = np.zeros((received_block.n_epochs, 5))
            probabilities[:, 0 if len(channel_set) == 1 else 2] = 1.0
            return pd.DataFrame(probabilities, columns=["W", "N1", "N2", "N3", "R"])

        result = stage_block(
            block,
            ("sleepfm",),
            sleepfm_predictor=predictor,
            sleepfm_channel_sets=(("E6",), ("E6", "E8")),
        )

        self.assertEqual(len(result.hypnodensities), block.n_epochs * 2)
        self.assertEqual(set(result.hypnodensities["channel_set"]), {"E6", "E6+E8"})
        multichannel = result.hypnodensities["channel_set"] == "E6+E8"
        self.assertTrue(result.hypnodensities.loc[multichannel, "electrode"].isna().all())
        self.assertEqual(set(result.qc["status"]), {"ok"})


if __name__ == "__main__":
    unittest.main()
