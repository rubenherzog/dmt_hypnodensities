"""Sleep staging over artifact-safe epochs with explicit contextual gaps."""

from __future__ import annotations

import io
import logging
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .epochs import EpochBlock

_STAGES = ("W", "N1", "N2", "N3", "R")
_ALIASES = {
    "w": "W",
    "wake": "W",
    "n1": "N1",
    "s1": "N1",
    "n2": "N2",
    "s2": "N2",
    "n3": "N3",
    "s3": "N3",
    "r": "R",
    "rem": "R",
}
SleepFMPredictor = Callable[[EpochBlock, tuple[str, ...]], object]


@dataclass(frozen=True)
class StagingResult:
    """Hypnodensities, reusable YASA features and per-adapter QC."""

    hypnodensities: pd.DataFrame
    yasa_features: pd.DataFrame
    qc: pd.DataFrame


def _usable_signal(block: EpochBlock) -> np.ndarray:
    if block.n_epochs == 0:
        return block.signal[:0]
    return block.signal[: block.n_epochs * block.epochs.shape[-1]]


def _make_raw(signal: np.ndarray, channels: Sequence[str], sampling_frequency: float):
    try:
        import mne
    except ImportError as error:  # pragma: no cover - optional installation
        raise ImportError("Sleep staging requires mne.") from error

    info = mne.create_info(list(channels), sampling_frequency, ch_types=["eeg"] * len(channels))
    return mne.io.RawArray(signal.T, info, verbose="error")


def _make_gssc_epochs(block: EpochBlock):
    """Filter each continuous run independently, then stack epochs for context."""

    try:
        import mne
    except ImportError as error:  # pragma: no cover - optional installation
        raise ImportError("GSSC staging requires mne.") from error

    samples_per_epoch = block.epochs.shape[-1]
    epoch_parts = []
    info = None
    for run in block.continuous_runs:
        raw = _make_raw(run.signal, block.channels, block.sampling_frequency)
        raw.filter(0.3, 30.0, verbose="error")
        info = raw.info.copy()
        epoch_parts.append(
            raw.get_data()
            .reshape(len(block.channels), run.n_epochs, samples_per_epoch)
            .transpose(1, 0, 2)
        )
    data = np.concatenate(epoch_parts, axis=0)
    return mne.EpochsArray(data, info, tmin=0.0, verbose="error")


def _normalise_probabilities(values: object) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        probabilities = values.copy().reset_index(drop=True)
        rename = {
            column: _ALIASES[str(column).strip().lower()]
            for column in probabilities.columns
            if str(column).strip().lower() in _ALIASES
        }
        probabilities = probabilities.rename(columns=rename)
        missing = [stage for stage in _STAGES if stage not in probabilities]
        if missing:
            raise ValueError(f"Missing staging probabilities: {missing}")
        probabilities = probabilities[list(_STAGES)]
    else:
        array = np.asarray(values)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != len(_STAGES):
            raise ValueError(
                "Staging probabilities must have shape (epoch, 5) in W/N1/N2/N3/R order."
            )
        probabilities = pd.DataFrame(array, columns=_STAGES)

    probabilities = probabilities.astype(float)
    probabilities = probabilities.rename(columns={stage: f"prob_{stage}" for stage in _STAGES})
    return probabilities


def _hypnodensity_rows(
    probabilities: object,
    stager: str,
    channel_set: Sequence[str],
) -> pd.DataFrame:
    output = _normalise_probabilities(probabilities)
    probability_columns = [f"prob_{stage}" for stage in _STAGES]
    output.insert(0, "epoch", np.arange(len(output), dtype=int))
    output.insert(1, "stager", stager)
    output.insert(2, "channel_set", "+".join(channel_set))
    output.insert(3, "electrode", channel_set[0] if len(channel_set) == 1 else pd.NA)
    output["stage"] = output[probability_columns].idxmax(axis=1).str.removeprefix("prob_")
    return output


def _qc_row(
    block: EpochBlock,
    stager: str,
    channel_set: Sequence[str],
    status: str,
    returned_epochs: int = 0,
    error: str = "",
) -> dict[str, object]:
    return {
        "recording_id": block.recording_id,
        "block_id": block.block_id,
        "stager": stager,
        "channel_set": "+".join(channel_set),
        "status": status,
        "expected_epochs": block.n_epochs,
        "returned_epochs": returned_epochs,
        "error": error,
    }


def _run_gssc(
    block: EpochBlock,
    device: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    try:
        from importlib.resources import files

        import gssc.nets
        import torch
        from gssc.infer import EEGInfer
    except ImportError as error:  # pragma: no cover - optional installation
        raise ImportError("GSSC staging requires the optional gssc dependency.") from error

    if block.n_epochs == 0:
        return pd.DataFrame(), [
            _qc_row(block, "gssc", (channel,), "no_epochs") for channel in block.channels
        ]

    epochs = _make_gssc_epochs(block)
    # GSSC 0.0.9 bundles complete model objects and predates the Torch 2.6 change of
    # ``torch.load(weights_only=True)``. Load only its two installed package resources
    # with the historical mode; never change Torch's global default.
    signal_model = torch.load(files(gssc.nets).joinpath("sig_net_v1.pt"), weights_only=False)
    context_model = torch.load(files(gssc.nets).joinpath("gru_net_v1.pt"), weights_only=False)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for GSSC but is not available.")
    use_cuda = torch.cuda.is_available() if device == "auto" else device == "cuda"
    with warnings.catch_warnings(), redirect_stdout(io.StringIO()):
        warnings.filterwarnings(
            "ignore",
            message="WARNING: CUDA is available.*",
            category=UserWarning,
        )
        model = EEGInfer(net=signal_model, con_net=context_model, use_cuda=use_cuda)
    tables = []
    qc = []
    for channel in block.channels:
        try:
            with redirect_stdout(io.StringIO()):
                _, _, probabilities = model.mne_infer(
                    epochs.copy(), eeg=[channel], filter=False
                )
            table = _hypnodensity_rows(probabilities, "gssc", (channel,))
            if len(table) != block.n_epochs:
                raise RuntimeError(f"Expected {block.n_epochs} epochs, GSSC returned {len(table)}.")
            tables.append(table)
            qc.append(_qc_row(block, "gssc", (channel,), "ok", len(table)))
        except Exception as error:  # noqa: BLE001 - preserve per-channel QC
            qc.append(_qc_row(block, "gssc", (channel,), "failed", error=str(error)))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(), qc


def _select_yasa_base_features(features: pd.DataFrame) -> pd.DataFrame:
    context_suffixes = ("_c7min_norm", "_p2min_norm")
    excluded = {"time_hour", "time_norm"}
    keep = [
        column
        for column in features.columns
        if column == "epoch" or (column not in excluded and not column.endswith(context_suffixes))
    ]
    output = features[keep].copy()
    output = output.rename(
        columns={column: f"yasa_{column}" for column in output if column != "epoch"}
    )
    return output


def _yasa_contextual_features(base_features: pd.DataFrame) -> pd.DataFrame:
    """Reproduce YASA 0.7 context features after discontinuity-safe base extraction."""

    try:
        from sklearn.preprocessing import robust_scale
    except ImportError as error:  # pragma: no cover - installed with YASA
        raise ImportError("YASA context reconstruction requires scikit-learn.") from error

    features = base_features.copy().reset_index(drop=True)
    features.index.name = "epoch"
    centered = features.rolling(
        window=15,
        center=True,
        min_periods=1,
        win_type="triang",
    ).mean()
    centered[centered.columns] = robust_scale(centered, quantile_range=(5, 95))
    centered = centered.add_suffix("_c7min_norm")
    past = features.rolling(window=4, min_periods=1).mean()
    past[past.columns] = robust_scale(past, quantile_range=(5, 95))
    past = past.add_suffix("_p2min_norm")
    features = features.join(centered).join(past)
    times = np.arange(len(features), dtype=float) * 30.0
    features["time_hour"] = times / 3600
    features["time_norm"] = times / times[-1]
    float_columns = features.select_dtypes(np.float64).columns
    features[float_columns] = features[float_columns].astype(np.float32)
    return features.sort_index(axis=1)


def _yasa_eeg_base_features(data_uv: np.ndarray, sampling_frequency: float) -> pd.DataFrame:
    """Calculate YASA 0.7 EEG base features before its lossy float32 downcast."""

    try:
        from yasa.staging import (
            ant,
            bandpower_from_psd_ndarray,
            filter_data,
            sliding_window,
            sp_sig,
            sp_stats,
            trapezoid,
        )
    except ImportError as error:  # pragma: no cover - pinned YASA dependency
        raise ImportError("YASA 0.7 feature internals are unavailable.") from error

    sf = sampling_frequency
    filtered = filter_data(data_uv, sf, l_freq=0.4, h_freq=30, verbose=False)
    samples_per_epoch = round(30 * sf)
    if filtered.size == samples_per_epoch:
        # YASA 0.7's generic sliding-window helper rejects equality (window ==
        # signal length), although one complete 30-s epoch is valid. Bypass only
        # that helper; the filtered samples and all feature formulas stay unchanged.
        epochs = filtered[np.newaxis, :]
    else:
        _, epochs = sliding_window(filtered, sf=sf, window=30)
    mobility, complexity = ant.hjorth_params(epochs, axis=1)
    features = {
        "std": np.std(epochs, ddof=1, axis=1),
        "iqr": sp_stats.iqr(epochs, rng=(25, 75), axis=1),
        "skew": sp_stats.skew(epochs, axis=1),
        "kurt": sp_stats.kurtosis(epochs, axis=1),
        "nzc": ant.num_zerocross(epochs, axis=1),
        "hmob": mobility,
        "hcomp": complexity,
    }
    frequencies, power = sp_sig.welch(
        epochs,
        sf,
        window="hamming",
        nperseg=int(5 * sf),
        average="median",
    )
    bands = [
        (0.4, 1, "sdelta"),
        (1, 4, "fdelta"),
        (4, 8, "theta"),
        (8, 12, "alpha"),
        (12, 16, "sigma"),
        (16, 30, "beta"),
    ]
    bandpower = bandpower_from_psd_ndarray(power, frequencies, bands=bands)
    for index, (_, _, name) in enumerate(bands):
        features[name] = bandpower[index]
    delta = features["sdelta"] + features["fdelta"]
    features["dt"] = delta / features["theta"]
    features["ds"] = delta / features["sigma"]
    features["db"] = delta / features["beta"]
    features["at"] = features["alpha"] / features["theta"]
    broad = np.logical_and(frequencies >= 0.4, frequencies <= 30)
    features["abspow"] = trapezoid(power[:, broad], dx=frequencies[1] - frequencies[0])
    features["perm"] = np.apply_along_axis(
        ant.perm_entropy,
        axis=1,
        arr=epochs,
        normalize=True,
    )
    features["higuchi"] = np.apply_along_axis(ant.higuchi_fd, axis=1, arr=epochs)
    features["petrosian"] = ant.petrosian_fd(epochs, axis=1)
    return pd.DataFrame(features).add_prefix("eeg_")


def _run_yasa_channel(
    block: EpochBlock,
    channel_index: int,
    channel: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import yasa
    except ImportError as error:  # pragma: no cover - optional installation
        raise ImportError("YASA staging requires the optional yasa dependency.") from error

    base_parts = []
    staging = None
    for run in block.continuous_runs:
        raw = _make_raw(
            run.signal[:, channel_index, np.newaxis],
            (channel,),
            block.sampling_frequency,
        )
        logger = yasa.logger
        previous_level = logger.level
        logger.setLevel(logging.ERROR)
        try:
            staging = yasa.SleepStaging(raw, eeg_name=channel)
        finally:
            logger.setLevel(previous_level)
        base_parts.append(_yasa_eeg_base_features(staging.data[0], staging.sf))
    base_internal = pd.concat(base_parts, ignore_index=True)
    contextual = _yasa_contextual_features(base_internal)
    staging._features = contextual
    staging.feature_name_ = contextual.columns.tolist()
    prediction = staging.predict()
    if hasattr(prediction, "proba") and prediction.proba is not None:
        probabilities = prediction.proba
    else:
        probabilities = staging.predict_proba()
    hypnodensities = _hypnodensity_rows(probabilities, "yasa", (channel,))
    base_features = base_internal.copy()
    base_features.insert(0, "epoch", np.arange(len(base_features), dtype=int))
    base_features = _select_yasa_base_features(base_features)
    base_features["electrode"] = channel
    return hypnodensities, base_features


def run_yasa(
    block: EpochBlock,
    n_jobs: int = 1,
) -> StagingResult:
    """Run YASA once per channel and reuse its fitted features for staging."""

    if block.n_epochs == 0:
        qc = [_qc_row(block, "yasa", (channel,), "no_epochs") for channel in block.channels]
        return StagingResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(qc))
    if block.n_epochs == 1:
        qc = [
            _qc_row(block, "yasa", (channel,), "unsupported_single_epoch")
            for channel in block.channels
        ]
        base = pd.DataFrame({"epoch": [0] * len(block.channels), "electrode": block.channels})
        return StagingResult(pd.DataFrame(), base, pd.DataFrame(qc))

    def one(channel_index: int, channel: str):
        try:
            hypnodensities, features = _run_yasa_channel(block, channel_index, channel)
            if len(hypnodensities) != block.n_epochs or len(features) != block.n_epochs:
                raise RuntimeError(
                    f"Expected {block.n_epochs} epochs; YASA returned "
                    f"{len(hypnodensities)} staging and {len(features)} feature rows."
                )
            return (
                hypnodensities,
                features,
                _qc_row(block, "yasa", (channel,), "ok", len(hypnodensities)),
            )
        except Exception as error:  # noqa: BLE001 - preserve per-channel QC
            return (
                pd.DataFrame(),
                pd.DataFrame(),
                _qc_row(block, "yasa", (channel,), "failed", error=str(error)),
            )

    results = Parallel(n_jobs=n_jobs)(
        delayed(one)(channel_index, channel) for channel_index, channel in enumerate(block.channels)
    )
    hypnodensities = [result[0] for result in results if not result[0].empty]
    features = [result[1] for result in results if not result[1].empty]
    return StagingResult(
        pd.concat(hypnodensities, ignore_index=True) if hypnodensities else pd.DataFrame(),
        pd.concat(features, ignore_index=True)
        if features
        else pd.DataFrame(columns=["epoch", "electrode"]),
        pd.DataFrame([result[2] for result in results]),
    )


def _run_sleepfm(
    block: EpochBlock,
    predictor: SleepFMPredictor | None,
    channel_sets: Sequence[Sequence[str]] | None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    sets = (
        [tuple(items) for items in channel_sets]
        if channel_sets
        else [(channel,) for channel in block.channels]
    )
    if predictor is None:
        return pd.DataFrame(), [
            _qc_row(block, "sleepfm", items, "not_configured") for items in sets
        ]

    available = set(block.channels)
    tables = []
    qc = []
    for items in sets:
        missing = sorted(set(items) - available)
        if missing:
            qc.append(
                _qc_row(
                    block,
                    "sleepfm",
                    items,
                    "missing_channels",
                    error=f"Missing channels: {missing}",
                )
            )
            continue
        try:
            probabilities = predictor(block, items)
            table = _hypnodensity_rows(probabilities, "sleepfm", items)
            if len(table) != block.n_epochs:
                raise RuntimeError(
                    f"Expected {block.n_epochs} epochs, SleepFM returned {len(table)}."
                )
            tables.append(table)
            qc.append(_qc_row(block, "sleepfm", items, "ok", len(table)))
        except Exception as error:  # noqa: BLE001 - preserve per-channel-set QC
            qc.append(_qc_row(block, "sleepfm", items, "failed", error=str(error)))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(), qc


def _attach_block_metadata(table: pd.DataFrame, block: EpochBlock) -> pd.DataFrame:
    if table.empty:
        return table
    metadata = pd.DataFrame(
        [
            {
                "epoch": item.epoch_in_block,
                "continuous_run_id": item.continuous_run_id,
                "epoch_in_run": item.epoch_in_run,
                "break_before": item.break_before,
                "gap_before_seconds": item.gap_before_seconds,
                "absolute_start": item.absolute_start,
                "absolute_end": item.absolute_end,
                "experimental_label": item.experimental_label,
            }
            for item in block.epoch_metadata
        ]
    )
    output = table.merge(metadata, on="epoch", how="left", validate="many_to_one")
    output.insert(0, "block_id", block.block_id)
    output.insert(0, "recording_id", block.recording_id)
    return output


def stage_block(
    block: EpochBlock,
    stagers: Iterable[str],
    n_jobs: int = 1,
    min_epochs: Mapping[str, int] | None = None,
    sleepfm_predictor: SleepFMPredictor | None = None,
    sleepfm_channel_sets: Sequence[Sequence[str]] | None = None,
    device: str = "cpu",
) -> StagingResult:
    """Run stagers while keeping preprocessing boundaries at every signal gap."""

    selected = tuple(dict.fromkeys(str(stager).lower() for stager in stagers))
    unknown = sorted(set(selected) - {"gssc", "yasa", "sleepfm"})
    if unknown:
        raise ValueError(f"Unsupported stagers: {unknown}")

    hypnodensity_tables = []
    qc_tables = []
    yasa_features = pd.DataFrame()
    if "gssc" in selected:
        minimum = max(1, int((min_epochs or {}).get("gssc", 1)))
        if block.n_epochs < minimum:
            hypnodensities = pd.DataFrame()
            qc = [
                _qc_row(block, "gssc", (channel,), "insufficient_epochs")
                for channel in block.channels
            ]
        else:
            hypnodensities, qc = _run_gssc(block, device=device)
        if not hypnodensities.empty:
            hypnodensity_tables.append(hypnodensities)
        qc_tables.append(pd.DataFrame(qc))
    if "yasa" in selected:
        minimum = max(1, int((min_epochs or {}).get("yasa", 1)))
        if block.n_epochs < minimum:
            qc = [
                _qc_row(block, "yasa", (channel,), "insufficient_epochs")
                for channel in block.channels
            ]
            yasa_result = StagingResult(
                pd.DataFrame(),
                pd.DataFrame(columns=["epoch", "electrode"]),
                pd.DataFrame(qc),
            )
        else:
            yasa_result = run_yasa(block, n_jobs=n_jobs)
        if not yasa_result.hypnodensities.empty:
            hypnodensity_tables.append(yasa_result.hypnodensities)
        yasa_features = yasa_result.yasa_features
        qc_tables.append(yasa_result.qc)
    if "sleepfm" in selected:
        minimum = max(1, int((min_epochs or {}).get("sleepfm", 1)))
        if block.n_epochs < minimum:
            sets = sleepfm_channel_sets or [(channel,) for channel in block.channels]
            hypnodensities = pd.DataFrame()
            qc = [_qc_row(block, "sleepfm", items, "insufficient_epochs") for items in sets]
        else:
            hypnodensities, qc = _run_sleepfm(block, sleepfm_predictor, sleepfm_channel_sets)
        if not hypnodensities.empty:
            hypnodensity_tables.append(hypnodensities)
        qc_tables.append(pd.DataFrame(qc))

    hypnodensities = (
        pd.concat(hypnodensity_tables, ignore_index=True) if hypnodensity_tables else pd.DataFrame()
    )
    return StagingResult(
        hypnodensities=_attach_block_metadata(hypnodensities, block),
        yasa_features=yasa_features,
        qc=pd.concat(qc_tables, ignore_index=True) if qc_tables else pd.DataFrame(),
    )
