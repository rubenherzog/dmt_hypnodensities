"""Tests for immutable run directories and epoch-table joins."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from dmt_hypnodensities.assembly import join_epoch_features_hypnodensities
from dmt_hypnodensities.config import AnalysisConfig, ExperimentalWindow
from dmt_hypnodensities.runs import finalize_run, prepare_run


class RunAndJoinTests(unittest.TestCase):
    def _config(self, directory: Path) -> AnalysisConfig:
        return AnalysisConfig(
            raw_dir=directory / "raw",
            output_dir=directory / "outputs",
            sampling_frequency=500,
            epoch_duration_seconds=30,
            gap_tolerance_seconds=70,
            experimental_windows=(ExperimentalWindow("before", 0, 30),),
            feature_analyses=("bandpower",),
            stagers=("gssc", "sleepfm"),
        )

    def test_run_refuses_configuration_drift_and_records_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            run = prepare_run(config, "main_v1")
            reopened = prepare_run(config, "main_v1")
            self.assertEqual(run.config_hash, reopened.config_hash)
            self.assertTrue(run.config_path.is_file())

            with self.assertRaises(ValueError):
                prepare_run(replace(config, gap_tolerance_seconds=1), "main_v1")

            finalize_run(run, pd.DataFrame({"status": ["ok", "failed"]}))
            manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed_with_failures")
            self.assertEqual(manifest["n_ok"], 1)

    def test_epoch_join_uses_monoelectrode_keys_and_excludes_multichannel_rows(self) -> None:
        features = pd.DataFrame(
            {
                "recording_id": ["R"],
                "block_id": ["B"],
                "epoch": [0],
                "electrode": ["E6"],
                "bp_rel_alpha": [0.4],
            }
        )
        hypnodensities = pd.DataFrame(
            {
                "recording_id": ["R", "R"],
                "block_id": ["B", "B"],
                "epoch": [0, 0],
                "electrode": ["E6", pd.NA],
                "stager": ["gssc", "sleepfm"],
                "channel_set": ["E6", "E6+E8"],
                "prob_W": [0.5, 0.6],
            }
        )

        joined = join_epoch_features_hypnodensities(features, hypnodensities)

        self.assertEqual(len(joined), 1)
        self.assertEqual(joined.iloc[0]["stager"], "gssc")
        self.assertEqual(joined.iloc[0]["bp_rel_alpha"], 0.4)


if __name__ == "__main__":
    unittest.main()
