"""Persistence tests for non-signal pipeline outputs."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dmt_hypnodensities.pipeline import RecordingResult
from dmt_hypnodensities.storage import save_recording_result


class StorageTests(unittest.TestCase):
    def test_result_is_saved_as_parquet_and_csv_only(self) -> None:
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("The declared pyarrow dependency is not installed in this environment.")

        result = RecordingResult(
            features=pd.DataFrame({"block_id": ["B000"], "value": [1.0]}),
            hypnodensities=pd.DataFrame(
                {"block_id": ["B000"], "stager": ["gssc"], "prob_W": [1.0]}
            ),
            spectra=pd.DataFrame({"block_id": ["B000"], "frequency_hz": [1.0], "power": [2.0]}),
            blocks=pd.DataFrame({"block_id": ["B000"], "n_epochs": [1]}),
            staging_qc=pd.DataFrame({"block_id": ["B000"], "stager": ["gssc"], "status": ["ok"]}),
        )
        with tempfile.TemporaryDirectory() as directory:
            written = save_recording_result(result, directory, "recording")

            self.assertEqual(
                set(written),
                {"features", "hypnodensities", "spectra", "blocks", "staging_qc"},
            )
            self.assertTrue(all(path.is_file() for path in written.values()))
            self.assertEqual(
                {path.suffix for path in Path(directory).iterdir()}, {".parquet", ".csv"}
            )


if __name__ == "__main__":
    unittest.main()
