"""File-level batch execution with joblib."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .config import AnalysisConfig
from .pipeline import process_recording
from .storage import save_recording_result

_FILE_PATTERN = re.compile(r"(?P<subject>P\d+)(?P<session>D(?P<d_index>\d+))_inner\.mat$")


@dataclass(frozen=True)
class FileSelection:
    """Selected recordings and an auditable decision table."""

    paths: tuple[Path, ...]
    metadata: pd.DataFrame


def discover_recordings(
    raw_directory: Path | str,
    pattern: str = "*_inner.mat",
) -> tuple[Path, ...]:
    """Return all matching recordings in deterministic filename order."""

    directory = Path(raw_directory).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Raw-data directory does not exist: {directory}")
    recordings = tuple(sorted(directory.glob(pattern)))
    if not recordings:
        raise FileNotFoundError(f"No recordings matching {pattern!r} in {directory}.")
    return recordings


def select_recordings(
    paths: Sequence[Path | str],
    policy: str = "prefer_d5",
) -> FileSelection:
    """Apply the exact recording policy used by the joint extraction notebook."""

    if policy not in {"prefer_d5", "all"}:
        raise ValueError(f"Unsupported file selection policy: {policy!r}.")
    rows = []
    for value in sorted(Path(path).expanduser().resolve() for path in paths):
        match = _FILE_PATTERN.search(value.name)
        if match is None:
            raise ValueError(f"Cannot parse subject/session from {value.name!r}.")
        d_index = int(match.group("d_index"))
        rows.append(
            {
                "path": value,
                "file": value.name,
                "subject": match.group("subject"),
                "condition": "placebo" if d_index == 1 else "DMT",
                "d_index": d_index,
            }
        )
    if not rows:
        return FileSelection((), pd.DataFrame())

    candidates = pd.DataFrame(rows)
    decisions = []
    selected_paths = []
    for (subject, condition), group in candidates.groupby(["subject", "condition"], sort=False):
        group = group.sort_values(["d_index", "file"], ascending=[False, True])
        if policy == "all":
            kept = group
        elif condition == "DMT":
            d5 = group.loc[group["d_index"].eq(5)].sort_values("file")
            kept = d5.iloc[[0]] if not d5.empty else group.iloc[[0]]
        else:
            d1 = group.loc[group["d_index"].eq(1)].sort_values("file")
            kept = (
                d1.iloc[[0]] if not d1.empty else group.sort_values(["d_index", "file"]).iloc[[0]]
            )
        kept_names = set(kept["file"])
        dropped = sorted(set(group["file"]) - kept_names)
        for row in kept.itertuples(index=False):
            selected_paths.append(row.path)
            decisions.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "selected_file": row.file,
                    "selected_d_idx": int(row.d_index),
                    "dropped_files": ";".join(dropped),
                    "n_candidates": len(group),
                    "policy": policy,
                }
            )
    return FileSelection(paths=tuple(sorted(selected_paths)), metadata=pd.DataFrame(decisions))


def _process_one(
    path: Path,
    config: AnalysisConfig,
    channels: Sequence[str] | None,
    analyses: Sequence[str] | None,
    stagers: Sequence[str] | None,
    sleepfm_channel_sets: Sequence[Sequence[str]] | None,
    show_block_progress: bool,
) -> dict[str, object]:
    try:
        result = process_recording(
            path,
            config,
            channels=channels,
            analyses=analyses,
            stagers=stagers,
            # File-level parallelism owns the joblib layer; avoid nesting pools.
            n_jobs=1,
            sleepfm_channel_sets=sleepfm_channel_sets,
            show_progress=show_block_progress,
        )
        written = save_recording_result(result, config.output_dir, path.stem)
        row = {
            "recording": path.name,
            "status": "ok",
            "n_blocks": len(result.blocks),
            "n_epochs": int(result.blocks["n_epochs"].sum()) if not result.blocks.empty else 0,
            "n_feature_rows": len(result.features),
            "n_hypnodensity_rows": len(result.hypnodensities),
            "outputs": ";".join(str(item) for item in written.values()),
            "error": "",
        }
    except Exception as error:  # noqa: BLE001 - one bad recording must not abort the batch
        row = {
            "recording": path.name,
            "status": "failed",
            "n_blocks": 0,
            "n_epochs": 0,
            "n_feature_rows": 0,
            "n_hypnodensity_rows": 0,
            "outputs": "",
            "error": f"{type(error).__name__}: {error}",
        }
    status_directory = config.output_dir / "_status"
    status_directory.mkdir(parents=True, exist_ok=True)
    status_path = status_directory / f"{path.stem}.json"
    temporary = status_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(status_path)
    return row


def run_batch(
    config: AnalysisConfig,
    paths: Sequence[Path | str] | None = None,
    channels: Sequence[str] | None = None,
    analyses: Sequence[str] | None = None,
    stagers: Sequence[str] | None = None,
    sleepfm_channel_sets: Sequence[Sequence[str]] | None = None,
    n_jobs: int | None = None,
    reuse_completed: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Process recordings in parallel and persist each result immediately.

    With ``channels=None`` each worker selects the mapped montage present in that file.
    Joblib controls only file-level parallelism; numerical-library and accelerator thread
    settings are deliberately left untouched.
    """

    candidates = (
        discover_recordings(config.raw_dir)
        if paths is None
        else tuple(Path(path).expanduser().resolve() for path in paths)
    )
    selection = select_recordings(candidates, policy=config.file_selection_policy)
    recording_paths = selection.paths
    workers = config.n_jobs if n_jobs is None else int(n_jobs)
    existing_path = config.output_dir / "batch_summary.csv"
    existing = pd.read_csv(existing_path) if reuse_completed and existing_path.is_file() else pd.DataFrame()
    reusable: dict[str, dict[str, object]] = {}
    prior_rows = existing.to_dict("records") if not existing.empty else []
    status_directory = config.output_dir / "_status"
    for path in recording_paths:
        status_path = status_directory / f"{path.stem}.json"
        if status_path.is_file():
            prior_rows.append(json.loads(status_path.read_text(encoding="utf-8")))
    if prior_rows:
        for row in prior_rows:
            outputs = [Path(item) for item in str(row.get("outputs", "")).split(";") if item]
            if row.get("status") == "ok" and outputs and all(path.is_file() for path in outputs):
                reusable[str(row["recording"])] = row

    pending = [path for path in recording_paths if path.name not in reusable]
    computed = []
    if pending:
        generated = Parallel(n_jobs=workers, return_as="generator_unordered")(
            delayed(_process_one)(
                path,
                config,
                channels,
                analyses,
                stagers,
                sleepfm_channel_sets,
                workers == 1 and show_progress,
            )
            for path in pending
        )
        progress = tqdm(
            generated,
            total=len(pending),
            desc="Recordings",
            unit="recording",
            disable=not show_progress,
        )
        for row in progress:
            computed.append(row)
            progress.set_postfix_str(
                f"{Path(str(row['recording'])).stem}: {row['n_blocks']} bloques · {row['status']}"
            )
    computed_by_name = {str(row["recording"]): row for row in computed}
    rows = [
        reusable[path.name] if path.name in reusable else computed_by_name[path.name]
        for path in recording_paths
    ]
    summary = pd.DataFrame(rows)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for table, path in (
        (summary, config.output_dir / "batch_summary.csv"),
        (selection.metadata, config.output_dir / "file_selection.csv"),
    ):
        temporary = path.with_suffix(path.suffix + ".tmp")
        table.to_csv(temporary, index=False)
        temporary.replace(path)
    return summary
