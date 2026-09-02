"""Configuration loading and validation."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentalWindow:
    """Half-open experimental interval ``[start, end)`` in recording seconds."""

    label: str
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"Window {self.label!r} must have end > start.")

    def contains(self, time_seconds: float) -> bool:
        return self.start <= time_seconds < self.end


@dataclass(frozen=True)
class AnalysisConfig:
    """Core settings required before staging and feature extraction."""

    raw_dir: Path
    output_dir: Path
    sampling_frequency: float
    epoch_duration_seconds: float
    gap_tolerance_seconds: float
    experimental_windows: tuple[ExperimentalWindow, ...]
    electrode_mapping_path: Path | None = None
    file_selection_policy: str = "prefer_d5"
    feature_analyses: tuple[str, ...] = ()
    stagers: tuple[str, ...] = ()
    stager_min_epochs: tuple[tuple[str, int], ...] = ()
    sleepfm_channel_sets: tuple[tuple[str, ...], ...] = ()
    sleepfm_repository: Path | None = None
    compute_device: str = "cpu"
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if self.sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be positive.")
        if self.epoch_duration_seconds <= 0:
            raise ValueError("epoch_duration_seconds must be positive.")
        if self.gap_tolerance_seconds < 0:
            raise ValueError("gap_tolerance_seconds cannot be negative.")
        if self.compute_device not in {"cpu", "cuda", "auto"}:
            raise ValueError("compute_device must be 'cpu', 'cuda' or 'auto'.")
        if self.file_selection_policy not in {"prefer_d5", "all"}:
            raise ValueError(f"Unsupported file selection policy: {self.file_selection_policy!r}.")
        ordered = sorted(self.experimental_windows, key=lambda item: item.start)
        for previous, current in itertools.pairwise(ordered):
            if current.start < previous.end:
                raise ValueError(
                    f"Experimental windows {previous.label!r} and {current.label!r} overlap."
                )

    def label_time(self, time_seconds: float) -> str:
        for window in self.experimental_windows:
            if window.contains(time_seconds):
                return window.label
        return "outside"


def _as_window(label: str, bounds: Sequence[Any]) -> ExperimentalWindow:
    if len(bounds) != 2:
        raise ValueError(f"Window {label!r} must contain exactly [start, end].")
    return ExperimentalWindow(label=label, start=float(bounds[0]), end=float(bounds[1]))


def analysis_config_from_mapping(data: Mapping[str, Any], base_dir: Path) -> AnalysisConfig:
    data_section = data["data"]
    signal_section = data["signal"]
    segmentation_section = data["segmentation"]
    parallel_section = data.get("parallel", {})
    compute_section = data.get("compute", {})
    features_section = data.get("features", {})
    staging_section = data.get("staging", {})

    raw_dir = Path(data_section["raw_dir"]).expanduser()
    output_dir = Path(data_section["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    mapping_value = data_section.get("electrode_mapping")
    mapping_path = Path(mapping_value).expanduser() if mapping_value else None
    if mapping_path is not None and not mapping_path.is_absolute():
        mapping_path = base_dir / mapping_path

    windows = tuple(
        _as_window(label, bounds) for label, bounds in data["experimental_windows"].items()
    )
    epoch_duration = float(signal_section["epoch_duration_seconds"])
    stager_min_epochs = []
    for name, settings in staging_section.items():
        if not isinstance(settings, Mapping):
            continue
        if "min_epochs" in settings:
            minimum = int(settings["min_epochs"])
        elif "min_duration_seconds" in settings:
            minimum = math.ceil(float(settings["min_duration_seconds"]) / epoch_duration)
        else:
            minimum = 1
        stager_min_epochs.append((name, max(1, minimum)))

    return AnalysisConfig(
        raw_dir=raw_dir,
        output_dir=output_dir,
        electrode_mapping_path=mapping_path.resolve() if mapping_path is not None else None,
        file_selection_policy=str(data_section.get("file_selection_policy", "prefer_d5")),
        sampling_frequency=float(signal_section["sampling_frequency"]),
        epoch_duration_seconds=epoch_duration,
        gap_tolerance_seconds=float(segmentation_section["gap_tolerance_seconds"]),
        experimental_windows=windows,
        feature_analyses=tuple(name for name, enabled in features_section.items() if bool(enabled)),
        stagers=tuple(
            name
            for name, settings in staging_section.items()
            if isinstance(settings, Mapping) and bool(settings.get("enabled", False))
        ),
        stager_min_epochs=tuple(stager_min_epochs),
        sleepfm_channel_sets=tuple(
            tuple(str(channel) for channel in channel_set)
            for channel_set in staging_section.get("sleepfm", {}).get("channel_sets", ())
        ),
        sleepfm_repository=(
            (base_dir / staging_section["sleepfm"]["repository"]).resolve()
            if staging_section.get("sleepfm", {}).get("repository")
            and not Path(staging_section["sleepfm"]["repository"]).expanduser().is_absolute()
            else Path(staging_section["sleepfm"]["repository"]).expanduser().resolve()
            if staging_section.get("sleepfm", {}).get("repository")
            else None
        ),
        compute_device=str(compute_section.get("device", "cpu")).lower(),
        n_jobs=int(parallel_section.get("n_jobs", -1)),
    )


def load_config(path: Path | str) -> AnalysisConfig:
    """Load the core analysis configuration from YAML."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw_config = yaml.safe_load(stream)
    if not isinstance(raw_config, Mapping):
        raise TypeError("The YAML root must be a mapping.")
    return analysis_config_from_mapping(raw_config, base_dir=config_path.parent.parent)


def analysis_config_to_mapping(config: AnalysisConfig) -> dict[str, Any]:
    """Return the fully resolved, serializable configuration used by a run."""

    minimums = dict(config.stager_min_epochs)
    staging: dict[str, Any] = {}
    for name in ("gssc", "yasa", "sleepfm"):
        settings: dict[str, Any] = {
            "enabled": name in config.stagers,
            "min_epochs": int(minimums.get(name, 1)),
        }
        if name == "sleepfm":
            settings["repository"] = (
                str(config.sleepfm_repository) if config.sleepfm_repository is not None else None
            )
            if config.sleepfm_channel_sets:
                settings["channel_sets"] = [list(items) for items in config.sleepfm_channel_sets]
        staging[name] = settings
    return {
        "data": {
            "raw_dir": str(config.raw_dir),
            "output_dir": str(config.output_dir),
            "electrode_mapping": (
                str(config.electrode_mapping_path)
                if config.electrode_mapping_path is not None
                else None
            ),
            "file_selection_policy": config.file_selection_policy,
        },
        "signal": {
            "sampling_frequency": config.sampling_frequency,
            "epoch_duration_seconds": config.epoch_duration_seconds,
        },
        "segmentation": {"gap_tolerance_seconds": config.gap_tolerance_seconds},
        "experimental_windows": {
            window.label: [window.start, window.end]
            for window in config.experimental_windows
        },
        "staging": staging,
        "compute": {"device": config.compute_device},
        "features": {
            name: name in config.feature_analyses
            for name in ("spectrum", "bandpower", "specparam", "catch22", "yasa_features")
        },
        "parallel": {"n_jobs": config.n_jobs},
    }
