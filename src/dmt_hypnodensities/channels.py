"""Automatic electrode selection from the EGI-to-10–20 mapping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

EGI_COLUMN = "EGI-257 Label"
STANDARD_COLUMN = "10-20 Label"


@dataclass(frozen=True)
class ChannelSelection:
    """Channels present in one recording and their standard labels."""

    channels: tuple[str, ...]
    electrode_to_1020: Mapping[str, str]
    unavailable_channels: tuple[str, ...]


def load_electrode_mapping(path: Path | str) -> pd.DataFrame:
    """Load and validate a one-to-one EGI/10–20 mapping."""

    mapping = pd.read_csv(Path(path).expanduser().resolve())
    missing = [column for column in (EGI_COLUMN, STANDARD_COLUMN) if column not in mapping]
    if missing:
        raise ValueError(f"Electrode mapping is missing columns: {missing}")
    mapping = mapping.dropna(subset=[EGI_COLUMN, STANDARD_COLUMN]).copy()
    mapping[EGI_COLUMN] = mapping[EGI_COLUMN].astype(str).str.strip()
    mapping[STANDARD_COLUMN] = mapping[STANDARD_COLUMN].astype(str).str.strip()
    if mapping[EGI_COLUMN].duplicated().any():
        duplicates = sorted(mapping.loc[mapping[EGI_COLUMN].duplicated(False), EGI_COLUMN].unique())
        raise ValueError(f"Duplicate EGI labels in electrode mapping: {duplicates}")
    return mapping.reset_index(drop=True)


def select_analysis_electrodes(
    mapping: pd.DataFrame,
    available_channels: Sequence[str],
) -> ChannelSelection:
    """Select the study montage and intersect it with a recording's channels.

    The scientific montage follows the notebook definition: standard labels ending in
    ``z``, ``1`` or ``2``, plus all ``PO*`` electrodes, excluding ``Iz`` globally.
    Parentheses are explicit so the exclusion applies to every branch.
    """

    labels = mapping[STANDARD_COLUMN]
    selected = mapping.loc[
        (labels.str.endswith(("z", "1", "2")) | labels.str.startswith("PO")) & labels.ne("Iz")
    ]
    available = {str(channel) for channel in available_channels}
    present = selected.loc[selected[EGI_COLUMN].isin(available)]
    channels = tuple(present[EGI_COLUMN])
    if not channels:
        raise ValueError("No mapped analysis electrodes are present in the recording.")
    unavailable = tuple(selected.loc[~selected[EGI_COLUMN].isin(available), EGI_COLUMN])
    return ChannelSelection(
        channels=channels,
        electrode_to_1020=dict(zip(present[EGI_COLUMN], present[STANDARD_COLUMN])),
        unavailable_channels=unavailable,
    )
