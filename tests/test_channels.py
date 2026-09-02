"""Tests for mapping-driven electrode selection."""

import unittest

import pandas as pd

from dmt_hypnodensities.channels import select_analysis_electrodes


class ChannelSelectionTests(unittest.TestCase):
    def test_scientific_montage_excludes_iz_for_every_filter_branch(self) -> None:
        mapping = pd.DataFrame(
            {
                "EGI-257 Label": ["E1", "E2", "E3", "E4", "E5", "E6"],
                "10-20 Label": ["Fz", "P1", "PO7", "Iz", "C5", "PO2"],
            }
        )
        selection = select_analysis_electrodes(
            mapping, available_channels=("E1", "E2", "E3", "E4", "E5")
        )

        self.assertEqual(selection.channels, ("E1", "E2", "E3"))
        self.assertEqual(selection.electrode_to_1020["E3"], "PO7")
        self.assertEqual(selection.unavailable_channels, ("E6",))


if __name__ == "__main__":
    unittest.main()
