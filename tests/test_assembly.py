"""Tests for assembling canonical per-recording outputs."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dmt_hypnodensities.assembly import assemble_outputs
from dmt_hypnodensities.pipeline import RecordingResult
from dmt_hypnodensities.storage import save_recording_result


class AssemblyTests(unittest.TestCase):
    def test_outputs_are_assembled_from_batch_manifest(self) -> None:
        name = "DMTCI_P07D1_inner"
        result = RecordingResult(
            features=pd.DataFrame(
                {
                    "recording_id": [name],
                    "block_id": [f"{name}__B000"],
                    "epoch": [0],
                    "electrode": ["E6"],
                    "value": [1.0],
                }
            ),
            hypnodensities=pd.DataFrame(
                {
                    "recording_id": [name],
                    "block_id": [f"{name}__B000"],
                    "epoch": [0],
                    "stager": ["gssc"],
                    "channel_set": ["E6"],
                    "prob_W": [1.0],
                }
            ),
            spectra=pd.DataFrame(),
            blocks=pd.DataFrame(
                {"recording_id": [name], "block_id": [f"{name}__B000"], "n_epochs": [1]}
            ),
            staging_qc=pd.DataFrame(),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_recording_result(result, output, name)
            pd.DataFrame({"recording": [f"{name}.mat"], "status": ["ok"]}).to_csv(
                output / "batch_summary.csv", index=False
            )

            assembled = assemble_outputs(output)

            self.assertEqual(len(assembled.features), 1)
            self.assertEqual(len(assembled.hypnodensities), 1)
            self.assertEqual(len(assembled.blocks), 1)


if __name__ == "__main__":
    unittest.main()
