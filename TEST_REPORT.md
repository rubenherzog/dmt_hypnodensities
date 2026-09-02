# Validation Coverage

The test suite covers the public data contracts and the main processing paths without requiring raw recordings or external model checkpoints.

## Covered areas

- FieldTrip/HDF5 channel and recording loading.
- Gap-aware run detection, epoch construction, and experimental labels.
- Channel mapping and analysis-channel selection.
- Feature extraction schemas and regression checks.
- GSSC, YASA, and SleepFM staging adapters using local or synthetic predictors.
- Batch selection, resumable processing, and output persistence.
- Output assembly and key validation.
- Hypnodensity statistics, correlations, and plotting helpers.

## Running the tests

```bash
pytest -q
```

Optional integrations may require their own dependencies or external checkpoints. Tests that depend on those resources should be run in an environment where the relevant extras are installed and configured.
