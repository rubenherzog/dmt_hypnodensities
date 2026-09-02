"""In-memory adapter for the official local ``sleepfm-clinical`` checkpoints."""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .epochs import EpochBlock

SLEEPFM_CLASSES = ("W", "N1", "N2", "N3", "R")


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _import_official_models(repository: Path):
    models_path = repository / "sleepfm" / "models" / "models.py"
    if not models_path.is_file():
        raise FileNotFoundError(f"SleepFM model source was not found at {models_path}.")
    module_name = "_dmt_hypnodensities_sleepfm_models"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    specification = importlib.util.spec_from_file_location(module_name, models_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import SleepFM models from {models_path}.")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def _without_data_parallel_prefix(state_dict):
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def _torch_device(torch, requested: str = "auto"):
    """Resolve an explicit CPU/CUDA request without changing global runtime settings."""

    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for SleepFM but is not available.")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError("SleepFM device must be 'cpu', 'cuda' or 'auto'.")


def _official_resample_and_standardize(
    signal: np.ndarray,
    source_frequency: float,
    target_frequency: float,
    target_samples: int | None = None,
) -> np.ndarray:
    """Reproduce the released SleepFM preprocessing without writing signal files."""

    try:
        from scipy.signal import butter, filtfilt
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("SleepFM preprocessing requires scipy.") from error

    duration = signal.shape[0] / source_frequency
    target_samples = (
        int(duration * target_frequency) if target_samples is None else int(target_samples)
    )
    if target_samples <= 0:
        raise ValueError("The block is too short to resample for SleepFM.")
    original_time = np.linspace(0, duration, num=signal.shape[0], endpoint=False)
    target_time = np.linspace(0, duration, num=target_samples, endpoint=False)

    output = []
    for channel_signal in signal.T:
        values = np.asarray(channel_signal, dtype=np.float64)
        if source_frequency > target_frequency:
            cutoff = min(target_frequency / 2, source_frequency / 2)
            b, a = butter(4, cutoff / (source_frequency / 2), btype="low", analog=False)
            values = filtfilt(b, a, values)
        values = np.interp(target_time, original_time, values)
        standard_deviation = float(np.std(values))
        values = values - float(np.mean(values))
        if standard_deviation != 0:
            values = values / standard_deviation
        if not np.isfinite(values).all():
            raise ValueError("SleepFM preprocessing produced non-finite samples.")
        output.append(values.astype(np.float32, copy=False))
    return np.stack(output)


class LocalSleepFMPredictor:
    """Run official base and sleep-staging checkpoints directly on an epoch block.

    The released head predicts at 5-second resolution. Six consecutive probability
    vectors are averaged to produce each 30-second hypnodensity expected by this package.
    Signal preprocessing and base encoding are performed independently within every
    strictly continuous run. The resulting embeddings are then passed once to the staging
    head, so an epoch never crosses a gap while the context group remains available to the
    temporal model, including groups made from one-epoch runs.
    """

    def __init__(self, repository: Path | str, device: str = "auto"):
        self.repository = Path(repository).expanduser().resolve()
        self.requested_device = device
        self._torch = None
        self._base_model = None
        self._staging_model = None
        self._base_config = None
        self._staging_config = None
        self._device = None

    def _load(self) -> None:
        if self._base_model is not None:
            return
        try:
            import torch
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError("Local SleepFM inference requires torch.") from error

        models = _import_official_models(self.repository)
        checkpoint_root = self.repository / "sleepfm" / "checkpoints"
        base_directory = checkpoint_root / "model_base"
        staging_directory = checkpoint_root / "model_sleep_staging"
        base_config = _load_json(base_directory / "config.json")
        staging_config = _load_json(staging_directory / "config.json")
        if float(base_config["sampling_duration"]) != 5:
            raise ValueError("This adapter expects the released 5-second SleepFM tokenizer.")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="enable_nested_tensor is True.*",
                category=UserWarning,
            )
            base_model = models.SetTransformer(
                int(base_config["in_channels"]),
                int(base_config["patch_size"]),
                int(base_config["embed_dim"]),
                int(base_config["num_heads"]),
                int(base_config["num_layers"]),
                pooling_head=int(base_config["pooling_head"]),
                dropout=float(base_config["dropout"]),
            )
        base_checkpoint = torch.load(
            base_directory / "best.pt", map_location="cpu", weights_only=True
        )
        base_model.load_state_dict(
            _without_data_parallel_prefix(base_checkpoint["state_dict"]), strict=True
        )

        staging_class = getattr(models, str(staging_config["model"]))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="enable_nested_tensor is True.*",
                category=UserWarning,
            )
            staging_model = staging_class(**staging_config["model_params"])
        staging_checkpoint = torch.load(
            staging_directory / "best.pth", map_location="cpu", weights_only=True
        )
        staging_model.load_state_dict(
            _without_data_parallel_prefix(staging_checkpoint), strict=True
        )

        device = _torch_device(torch, self.requested_device)
        self._torch = torch
        self._device = device
        self._base_model = base_model.to(device).eval()
        self._staging_model = staging_model.to(device).eval()
        self._base_config = base_config
        self._staging_config = staging_config

    def __call__(
        self,
        block: EpochBlock,
        channel_set: Sequence[str],
    ) -> pd.DataFrame:
        self._load()
        if block.n_epochs == 0:
            return pd.DataFrame(columns=SLEEPFM_CLASSES)
        channels = tuple(channel_set)
        if not channels:
            raise ValueError("SleepFM requires at least one electrode.")
        maximum_channels = int(self._base_config.get("BAS_CHANNELS", 10))
        if len(channels) > maximum_channels:
            raise ValueError(
                f"The released SleepFM base checkpoint supports at most "
                f"{maximum_channels} EEG electrodes per set."
            )
        missing = sorted(set(channels) - set(block.channels))
        if missing:
            raise ValueError(f"SleepFM channels are absent from the block: {missing}")
        run_epochs = sum(run.n_epochs for run in block.continuous_runs)
        if run_epochs != block.n_epochs:
            raise ValueError(
                f"SleepFM received {run_epochs} run epochs for a {block.n_epochs}-epoch block."
            )

        channel_indices = [block.channels.index(channel) for channel in channels]
        target_frequency = float(self._base_config["sampling_freq"])

        torch = self._torch
        patch_size = int(self._base_config["patch_size"])
        patches_per_chunk = int(
            float(self._base_config["sampling_duration"]) * 60 * target_frequency // patch_size
        )
        samples_per_chunk = patches_per_chunk * patch_size
        embeddings = []
        with torch.inference_mode():
            for run in block.continuous_runs:
                signal = run.signal[:, channel_indices]
                prepared = _official_resample_and_standardize(
                    signal,
                    run.sampling_frequency,
                    target_frequency,
                    target_samples=round(
                        run.n_epochs
                        * run.epochs.shape[-1]
                        / run.sampling_frequency
                        * target_frequency
                    ),
                )
                for start in range(0, prepared.shape[1], samples_per_chunk):
                    chunk = prepared[:, start : start + samples_per_chunk]
                    if chunk.shape[1] % patch_size:
                        raise ValueError(
                            "SleepFM input length is not an exact number of patches."
                        )
                    tensor = torch.from_numpy(chunk).unsqueeze(0).to(self._device)
                    channel_mask = torch.zeros(
                        (1, len(channels)), dtype=torch.bool, device=self._device
                    )
                    _, temporal = self._base_model(tensor, channel_mask)
                    embeddings.append(temporal)

            if not embeddings:
                raise RuntimeError("SleepFM received no continuous-run embeddings.")
            temporal = torch.cat(embeddings, dim=1).unsqueeze(1)
            temporal_mask = torch.zeros(temporal.shape[:3], dtype=torch.bool, device=self._device)
            logits, _ = self._staging_model(temporal, temporal_mask)
            probabilities_5s = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        tokens_per_epoch = round(block.epochs.shape[-1] / block.sampling_frequency / 5.0)
        expected_tokens = block.n_epochs * tokens_per_epoch
        if tokens_per_epoch != 6 or probabilities_5s.shape[0] != expected_tokens:
            raise RuntimeError(
                f"Expected {expected_tokens} SleepFM 5-second predictions; "
                f"received {probabilities_5s.shape[0]}."
            )
        probabilities_30s = probabilities_5s.reshape(
            block.n_epochs, tokens_per_epoch, len(SLEEPFM_CLASSES)
        ).mean(axis=1)
        return pd.DataFrame(probabilities_30s, columns=SLEEPFM_CLASSES)
