"""Feature extraction from the shared block and epoch representations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .epochs import EpochBlock

_BANDPOWER_COLUMNS = ("Delta", "Theta", "Alpha", "Sigma", "Beta", "Gamma")
_SUPPORTED_ANALYSES = frozenset({"bandpower", "spectrum", "specparam", "catch22", "yasa_features"})
_KEY_COLUMNS = ["recording_id", "block_id", "epoch", "electrode"]


@dataclass(frozen=True)
class FeatureResult:
    """Tabular outputs produced from one continuous block.

    ``features`` contains one row per epoch and electrode. ``spectra`` is kept
    separately because it has one row per frequency as well and therefore a different
    cardinality. Neither table contains raw signals or epoch arrays.
    """

    features: pd.DataFrame
    spectra: pd.DataFrame
    qc: dict[str, object]


@dataclass(frozen=True)
class _SpectrumArray:
    frequencies: np.ndarray
    power: np.ndarray  # epoch x channel x frequency


def _validate_epochs(
    epochs: np.ndarray,
    channels: Sequence[str],
    sampling_frequency: float,
) -> np.ndarray:
    epoch_array = np.asarray(epochs)
    if epoch_array.ndim != 3:
        raise ValueError("epochs must have shape (epoch, channel, sample).")
    if epoch_array.shape[1] != len(channels):
        raise ValueError("The channel dimension does not match the channel names.")
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive.")
    return epoch_array


def extract_relative_bandpower(
    epochs: np.ndarray,
    channels: Sequence[str],
    sampling_frequency: float,
) -> pd.DataFrame:
    """Calculate YASA relative bandpower for every epoch and channel."""

    epoch_array = _validate_epochs(epochs, channels, sampling_frequency)
    try:
        import yasa
    except ImportError as error:  # pragma: no cover - optional installation
        raise ImportError("Relative bandpower requires the optional 'yasa' dependency.") from error

    rows = []
    for epoch_index, epoch in enumerate(epoch_array):
        bandpower = yasa.bandpower(
            epoch,
            sf=sampling_frequency,
            ch_names=list(channels),
            relative=True,
        )
        if "Chan" not in bandpower.columns:
            bandpower = bandpower.rename_axis("Chan").reset_index()

        for _, values in bandpower.iterrows():
            row = {"epoch": epoch_index, "electrode": str(values["Chan"])}
            for band in _BANDPOWER_COLUMNS:
                if band in values.index:
                    row[f"bp_rel_{band.lower()}"] = float(values[band])
            rows.append(row)

    columns = ["epoch", "electrode"] + [f"bp_rel_{band.lower()}" for band in _BANDPOWER_COLUMNS]
    return pd.DataFrame(rows).reindex(columns=columns)


def _calculate_welch(
    epochs: np.ndarray,
    sampling_frequency: float,
    window_seconds: float = 4.0,
) -> _SpectrumArray:
    try:
        from scipy.signal import welch
    except ImportError as error:  # pragma: no cover - optional installation
        raise ImportError("Spectrum extraction requires scipy.") from error

    samples = epochs.shape[-1]
    nperseg = min(samples, round(window_seconds * sampling_frequency))
    frequencies, power = welch(
        epochs,
        fs=sampling_frequency,
        nperseg=nperseg,
        axis=-1,
    )
    return _SpectrumArray(frequencies=frequencies, power=power)


def _spectrum_table(spectrum: _SpectrumArray, channels: Sequence[str]) -> pd.DataFrame:
    n_epochs, n_channels, n_frequencies = spectrum.power.shape
    return pd.DataFrame(
        {
            "epoch": np.repeat(np.arange(n_epochs), n_channels * n_frequencies),
            "electrode": np.tile(np.repeat(np.asarray(channels), n_frequencies), n_epochs),
            "frequency_hz": np.tile(spectrum.frequencies, n_epochs * n_channels),
            "power": spectrum.power.reshape(-1),
        }
    )


def _fit_one_specparam(
    frequencies: np.ndarray,
    power: np.ndarray,
    frequency_range: tuple[float, float],
) -> dict[str, float]:
    try:
        from scipy.signal import savgol_filter
        from specparam import SpectralModel
    except ImportError as error:  # pragma: no cover - optional installation
        raise ImportError("Specparam extraction requires scipy and specparam.") from error

    valid = np.isfinite(frequencies) & np.isfinite(power)
    frequencies = frequencies[valid]
    power = power[valid]
    window = min(7, power.size if power.size % 2 else power.size - 1)
    if window >= 3:
        power = savgol_filter(power, window_length=window, polyorder=2)
    power = np.maximum(power, np.finfo(float).eps)

    low = max(frequency_range[0], float(frequencies.min()))
    high = min(frequency_range[1], float(frequencies.max()))
    if low >= high:
        return _empty_specparam_values()

    try:
        model = SpectralModel(verbose=False)
        model.fit(frequencies, power, [low, high])
        aperiodic = np.asarray(model.get_params("aperiodic")).reshape(-1)
        return {
            "specparam_offset": float(aperiodic[0]),
            "specparam_exponent": float(aperiodic[-1]),
            "specparam_r2": _as_scalar(model.get_metrics("gof")),
            "specparam_error": _as_scalar(model.get_metrics("error")),
        }
    except Exception:  # noqa: BLE001 - numerical fit failures are represented as NaN
        return _empty_specparam_values()


def _as_scalar(value: object) -> float:
    array = np.asarray(value).reshape(-1)
    return float(array[0]) if array.size else np.nan


def _empty_specparam_values() -> dict[str, float]:
    return {
        "specparam_offset": np.nan,
        "specparam_exponent": np.nan,
        "specparam_r2": np.nan,
        "specparam_error": np.nan,
    }


def extract_specparam(
    spectrum: _SpectrumArray,
    channels: Sequence[str],
    n_jobs: int = 1,
    frequency_range: tuple[float, float] = (1.0, 45.0),
) -> pd.DataFrame:
    """Fit specparam using the Welch spectrum already calculated for the block."""

    locations = [
        (epoch, channel)
        for epoch in range(spectrum.power.shape[0])
        for channel in range(spectrum.power.shape[1])
    ]
    values = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one_specparam)(
            spectrum.frequencies,
            spectrum.power[epoch, channel],
            frequency_range,
        )
        for epoch, channel in locations
    )
    rows = []
    for (epoch, channel), result in zip(locations, values):
        rows.append({"epoch": epoch, "electrode": channels[channel], **result})
    return pd.DataFrame(rows)


def _catch22_one(signal: np.ndarray) -> dict[str, float]:
    try:
        import pycatch22
    except ImportError as error:  # pragma: no cover - optional installation
        raise ImportError("Catch22 extraction requires pycatch22.") from error

    result = pycatch22.catch22_all(np.asarray(signal, dtype=float))
    return {f"c22_{name}": float(value) for name, value in zip(result["names"], result["values"])}


def extract_catch22(
    epochs: np.ndarray,
    channels: Sequence[str],
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Calculate the canonical 22 Catch22 features per epoch and electrode."""

    locations = [
        (epoch, channel) for epoch in range(epochs.shape[0]) for channel in range(epochs.shape[1])
    ]
    values = Parallel(n_jobs=n_jobs)(
        delayed(_catch22_one)(epochs[epoch, channel]) for epoch, channel in locations
    )
    return pd.DataFrame(
        [
            {"epoch": epoch, "electrode": channels[channel], **result}
            for (epoch, channel), result in zip(locations, values)
        ]
    )


def extract_yasa_staging_features(
    block: EpochBlock,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Return the 21 base features from ``YASA SleepStaging.get_features``.

    Each channel is processed using the complete usable signal of the continuous block,
    but contextual and time-position columns are discarded. The discarded signal tail is
    excluded so that YASA and the shared epoch tensor have identical row counts.
    """

    from .staging import run_yasa

    result = run_yasa(block, n_jobs=n_jobs)
    output = result.yasa_features
    statuses = set(result.qc["status"]) if not result.qc.empty else {"unknown"}
    output.attrs["status"] = statuses.pop() if len(statuses) == 1 else "partial"
    return output


def _base_feature_table(block: EpochBlock) -> pd.DataFrame:
    rows = []
    for metadata in block.epoch_metadata:
        for electrode in block.channels:
            rows.append(
                {
                    "recording_id": block.recording_id,
                    "block_id": block.block_id,
                    "epoch": metadata.epoch_in_block,
                    "continuous_run_id": metadata.continuous_run_id,
                    "epoch_in_run": metadata.epoch_in_run,
                    "break_before": metadata.break_before,
                    "gap_before_seconds": metadata.gap_before_seconds,
                    "electrode": electrode,
                    "absolute_start": metadata.absolute_start,
                    "absolute_end": metadata.absolute_end,
                    "label_time": metadata.label_time,
                    "experimental_label": metadata.experimental_label,
                }
            )
    return pd.DataFrame(rows)


def _with_block_keys(table: pd.DataFrame, block: EpochBlock) -> pd.DataFrame:
    output = table.copy()
    output.insert(0, "block_id", block.block_id)
    output.insert(0, "recording_id", block.recording_id)
    return output


def extract_block_features(
    block: EpochBlock,
    analyses: Iterable[str],
    n_jobs: int = 1,
    precomputed_yasa_features: pd.DataFrame | None = None,
) -> FeatureResult:
    """Run selected analyses from one shared block/epoch construction.

    Valid analysis names are ``bandpower``, ``spectrum``, ``specparam``, ``catch22``
    and ``yasa_features``. Welch is calculated only once when both spectrum and
    specparam are requested.
    """

    selected = frozenset(analyses)
    unknown = sorted(selected - _SUPPORTED_ANALYSES)
    if unknown:
        raise ValueError(f"Unsupported feature analyses: {unknown}")

    features = _base_feature_table(block)
    spectra = pd.DataFrame(columns=_KEY_COLUMNS + ["frequency_hz", "power"])
    if block.n_epochs == 0:
        return FeatureResult(
            features=features,
            spectra=spectra,
            qc={"feature_status": "no_epochs"},
        )

    epoch_array = _validate_epochs(block.epochs, block.channels, block.sampling_frequency)
    tables = []
    qc = {"feature_status": "ok"}
    if "bandpower" in selected:
        tables.append(
            extract_relative_bandpower(epoch_array, block.channels, block.sampling_frequency)
        )

    spectrum = None
    if selected.intersection({"spectrum", "specparam"}):
        spectrum = _calculate_welch(epoch_array, block.sampling_frequency)
    if "spectrum" in selected and spectrum is not None:
        spectra = _with_block_keys(_spectrum_table(spectrum, block.channels), block)
    if "specparam" in selected and spectrum is not None:
        tables.append(extract_specparam(spectrum, block.channels, n_jobs=n_jobs))
    if "catch22" in selected:
        tables.append(extract_catch22(epoch_array, block.channels, n_jobs=n_jobs))
    if "yasa_features" in selected:
        yasa_features = precomputed_yasa_features
        if yasa_features is None:
            yasa_features = extract_yasa_staging_features(block, n_jobs=n_jobs)
        qc["yasa_features_status"] = yasa_features.attrs.get("status", "unknown")
        tables.append(yasa_features)

    for table in tables:
        table = _with_block_keys(table, block)
        features = features.merge(
            table,
            on=_KEY_COLUMNS,
            how="left",
            validate="one_to_one",
        )
    return FeatureResult(features=features, spectra=spectra, qc=qc)
