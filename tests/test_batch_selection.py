"""Regression tests for the notebook recording-selection policy."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from dmt_hypnodensities.batch import run_batch, select_recordings
from dmt_hypnodensities.config import AnalysisConfig, ExperimentalWindow
from dmt_hypnodensities.pipeline import RecordingResult


class FileSelectionTests(unittest.TestCase):
    def test_prefer_d5_reproduces_notebook_policy(self) -> None:
        names = (
            "DMTCI_P11D1_inner.mat",
            "DMTCI_P11D2_inner.mat",
            "DMTCI_P11D5_inner.mat",
            "DMTCI_P12D1_inner.mat",
            "DMTCI_P12D2_inner.mat",
        )
        selection = select_recordings([Path("/tmp") / name for name in names])

        selected = {path.name for path in selection.paths}
        self.assertEqual(
            selected,
            {
                "DMTCI_P11D1_inner.mat",
                "DMTCI_P11D5_inner.mat",
                "DMTCI_P12D1_inner.mat",
                "DMTCI_P12D2_inner.mat",
            },
        )
        p11_dmt = selection.metadata.query("subject == 'P11' and condition == 'DMT'").iloc[0]
        self.assertEqual(p11_dmt["selected_d_idx"], 5)
        self.assertEqual(p11_dmt["dropped_files"], "DMTCI_P11D2_inner.mat")

    def test_all_policy_keeps_every_candidate(self) -> None:
        paths = [
            Path("/tmp/DMTCI_P11D2_inner.mat"),
            Path("/tmp/DMTCI_P11D5_inner.mat"),
        ]
        selection = select_recordings(paths, policy="all")
        self.assertEqual({path.name for path in selection.paths}, {path.name for path in paths})

    def test_per_recording_status_resumes_without_global_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "DMTCI_P07D1_inner.mat"
            path.touch()
            config = AnalysisConfig(
                raw_dir=root,
                output_dir=root / "outputs",
                sampling_frequency=500,
                epoch_duration_seconds=30,
                gap_tolerance_seconds=70,
                experimental_windows=(ExperimentalWindow("before", 0, 30),),
            )
            result = RecordingResult(
                features=pd.DataFrame(
                    {
                        "recording_id": [path.stem],
                        "block_id": [f"{path.stem}__B000"],
                        "epoch": [0],
                        "electrode": ["E6"],
                    }
                ),
                hypnodensities=pd.DataFrame(),
                spectra=pd.DataFrame(),
                blocks=pd.DataFrame(
                    {
                        "recording_id": [path.stem],
                        "block_id": [f"{path.stem}__B000"],
                        "n_epochs": [1],
                    }
                ),
                staging_qc=pd.DataFrame(),
            )
            with patch("dmt_hypnodensities.batch.process_recording", return_value=result):
                first = run_batch(config, paths=[path], n_jobs=1, show_progress=False)
            (config.output_dir / "batch_summary.csv").unlink()

            with patch(
                "dmt_hypnodensities.batch.process_recording",
                side_effect=AssertionError("completed recording was recomputed"),
            ):
                resumed = run_batch(config, paths=[path], n_jobs=1, show_progress=False)

            self.assertEqual(first.loc[0, "status"], "ok")
            self.assertEqual(resumed.loc[0, "status"], "ok")
            self.assertTrue((config.output_dir / "_status" / f"{path.stem}.json").is_file())


if __name__ == "__main__":
    unittest.main()
