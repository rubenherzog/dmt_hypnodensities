# dmt_hypnodensities

`dmt_hypnodensities` is a Python pipeline for block-aware EEG analysis in continuous-infusion studies. It extracts hypnodensities, sleep-staging features, spectral measures, and quality-control tables from FieldTrip/HDF5 recordings.

## What it provides

- Gap-aware segmentation and 30-second epoching that never crosses missing samples.
- Shared staging adapters for GSSC, YASA, and SleepFM.
- EEG features including bandpower, Welch spectra, specparam, Catch22, and YASA features.
- Reproducible batch processing with resumable per-recording outputs.
- Validated assembly, statistical summaries, and plotting helpers.
- Tabular persistence using Parquet, CSV, and JSON without storing raw signals or model embeddings.

## Installation

The core package supports Python 3.10 or newer.

```bash
python -m pip install -e .
```

Install optional integrations and development tools with:

```bash
python -m pip install -e '.[features,gssc,sleepfm,stats,plots,dev]'
```

## Quick start

```python
from dmt_hypnodensities import load_config, process_recording, save_recording_result

config = load_config("configs/analysis.yaml")
result = process_recording("/path/to/recording.mat", config)
save_recording_result(result, config.output_dir, "recording_id")
```

`result` contains `features`, `hypnodensities`, `spectra`, `blocks`, and `staging_qc` tables. Channel selection can be automatic through the mapping in `configs/`, or explicit by passing a list of channel labels.

## Batch workflow

```python
from dmt_hypnodensities import load_config, run_batch

config = load_config("configs/analysis.yaml")
summary = run_batch(config)
```

The example notebooks cover the main workflow:

1. `notebooks/01_run_batch.ipynb` runs extraction and persistence.
2. `notebooks/02_hypnodensity_analysis.ipynb` analyzes staging tables.
3. `notebooks/03_feature_associations.ipynb` joins features and hypnodensities.

The gap-tolerance sensitivity script is available at `scripts/run_gap_tolerance_sweep.py`.

## Data and outputs

Raw recordings and external model repositories are intentionally kept outside the repository. Configure their locations in `configs/analysis.yaml`. Generated outputs are ignored by Git; this keeps the repository focused on source code, configuration, tests, and reproducible examples.

## Development

Run the test suite with:

```bash
pytest -q
```

Run linting with:

```bash
ruff check src tests
```

See [`REPRODUCTION.md`](REPRODUCTION.md) for implementation-level reproducibility notes and [`TEST_REPORT.md`](TEST_REPORT.md) for validation coverage.
