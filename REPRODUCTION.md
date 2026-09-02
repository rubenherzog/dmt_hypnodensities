# Reproducibility Notes

This document records the implementation contracts that affect reproducibility. It is intended for users who need to understand how discontinuous recordings are segmented and how outputs are aligned.

## Segmentation contract

The pipeline applies two distinct steps:

1. It identifies strictly continuous runs from the original timestamps.
2. It creates complete 30-second epochs within each run and discards incomplete tails.
3. It groups epoch-bearing runs when the effective gap is within `gap_tolerance_seconds`.
4. It uses those groups for contextual models while filtering and extracting features per run.

Contextual grouping never turns a discontinuous recording into a continuous waveform. Experimental labels such as `before`, `after`, `late`, and `outside` are assigned to epochs after segmentation and do not create artificial block boundaries.

## Alignment and persistence

Outputs are aligned using recording, block, epoch, electrode, and stager identifiers. The pipeline stores tables and compact quality-control information, but does not persist raw signals, epochs, tensors, or embeddings.

The batch workflow can reuse complete per-recording results. Configuration and output manifests should be kept with each analysis run so that tables can be traced to their inputs and settings.

## External models

GSSC, YASA, and SleepFM are optional integrations. External repositories and checkpoints are not included in this project; configure their local paths before enabling the corresponding stager.
