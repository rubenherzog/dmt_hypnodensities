"""Regression checks against the feature calculation used in the notebook."""

import unittest

import numpy as np
from _fixtures import make_block

from dmt_hypnodensities.features import (
    extract_block_features,
    extract_relative_bandpower,
    extract_yasa_staging_features,
)


class FeatureRegressionTests(unittest.TestCase):
    def test_relative_bandpower_matches_notebook_call_for_identical_epoch(self) -> None:
        try:
            import yasa
        except ImportError:
            self.skipTest("The optional YASA dependency is not installed.")

        sampling_frequency = 100.0
        times = np.arange(30 * int(sampling_frequency)) / sampling_frequency
        epoch = np.stack(
            (
                np.sin(2 * np.pi * 2 * times) + 0.25 * np.sin(2 * np.pi * 10 * times),
                np.sin(2 * np.pi * 10 * times) + 0.10 * np.sin(2 * np.pi * 20 * times),
            )
        )

        actual = extract_relative_bandpower(
            epoch[np.newaxis, ...], ["E6", "E8"], sampling_frequency
        )
        notebook_reference = yasa.bandpower(
            epoch,
            sf=sampling_frequency,
            ch_names=["E6", "E8"],
            relative=True,
        )

        for channel_index, channel in enumerate(("E6", "E8")):
            row = actual.loc[actual["electrode"] == channel].iloc[0]
            for band in ("Delta", "Theta", "Alpha", "Sigma", "Beta", "Gamma"):
                self.assertEqual(
                    row[f"bp_rel_{band.lower()}"],
                    notebook_reference.iloc[channel_index][band],
                )

    def test_yasa_features_preserve_block_epoch_alignment(self) -> None:
        try:
            import yasa  # noqa: F401
        except ImportError:
            self.skipTest("The optional YASA dependency is not installed.")

        block = make_block()
        features = extract_yasa_staging_features(block, n_jobs=1)

        self.assertEqual(len(features), block.n_epochs * len(block.channels))
        self.assertEqual(set(features["electrode"]), set(block.channels))
        yasa_columns = [column for column in features if column.startswith("yasa_")]
        self.assertEqual(len(yasa_columns), 21)
        self.assertFalse(any(column.endswith("_c7min_norm") for column in yasa_columns))
        self.assertFalse(any(column.endswith("_p2min_norm") for column in yasa_columns))
        self.assertNotIn("yasa_time_hour", yasa_columns)
        self.assertNotIn("yasa_time_norm", yasa_columns)

    def test_handler_reuses_epoch_keys_for_feature_and_spectrum_tables(self) -> None:
        block = make_block()
        result = extract_block_features(
            block,
            analyses=("bandpower", "spectrum"),
            n_jobs=1,
        )

        self.assertEqual(len(result.features), block.n_epochs * len(block.channels))
        self.assertFalse(result.spectra.empty)
        self.assertFalse(
            result.features.duplicated(["recording_id", "block_id", "epoch", "electrode"]).any()
        )

    def test_single_epoch_block_skips_only_yasa_features(self) -> None:
        block = make_block()
        one_epoch_block = block.__class__(
            **{
                **block.__dict__,
                "signal": block.signal[: block.epochs.shape[-1]],
                "sample_times": block.sample_times[: block.epochs.shape[-1]],
                "epochs": block.epochs[:1],
                "epoch_metadata": block.epoch_metadata[:1],
            }
        )
        result = extract_block_features(
            one_epoch_block,
            analyses=("bandpower", "yasa_features"),
            n_jobs=1,
        )

        self.assertEqual(len(result.features), len(block.channels))
        self.assertEqual(result.qc["yasa_features_status"], "unsupported_single_epoch")
        self.assertIn("bp_rel_delta", result.features.columns)


if __name__ == "__main__":
    unittest.main()
