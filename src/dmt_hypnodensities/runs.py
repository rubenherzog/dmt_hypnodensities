"""Reproducible run directories for compute-once cohort analyses."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd
import yaml

from .config import AnalysisConfig, analysis_config_to_mapping

_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class AnalysisRun:
    """Paths and resolved configuration for one immutable analysis definition."""

    root: Path
    recordings: Path
    tables: Path
    figures: Path
    config: AnalysisConfig
    config_hash: str
    source_hash: str

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def config_path(self) -> Path:
        return self.root / "resolved_config.yaml"


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _package_versions() -> dict[str, str | None]:
    names = (
        "dmt-hypnodensities",
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "yasa",
        "gssc",
        "torch",
        "statsmodels",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _source_hash() -> str:
    digest = hashlib.sha256()
    package_root = Path(__file__).resolve().parent
    for path in sorted(package_root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_run(
    config: AnalysisConfig,
    run_name: str,
    reuse_existing: bool = True,
) -> AnalysisRun:
    """Create or reopen a named run, refusing configuration drift."""

    if not _RUN_NAME.fullmatch(run_name):
        raise ValueError("run_name may contain only letters, numbers, '.', '_' and '-'.")
    root = config.output_dir.expanduser().resolve() / run_name
    recordings = root / "recordings"
    tables = root / "tables"
    figures = root / "figures"
    run_config = replace(config, output_dir=recordings)
    resolved = analysis_config_to_mapping(run_config)
    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    source_hash = _source_hash()
    run = AnalysisRun(root, recordings, tables, figures, run_config, config_hash, source_hash)

    if root.exists() and not reuse_existing:
        raise FileExistsError(f"Run already exists: {root}")
    for directory in (recordings, tables, figures):
        directory.mkdir(parents=True, exist_ok=True)

    if run.manifest_path.is_file():
        existing = json.loads(run.manifest_path.read_text(encoding="utf-8"))
        if existing.get("config_sha256") != config_hash:
            raise ValueError(
                f"Run {run_name!r} already exists with a different configuration. "
                "Use a new run name."
            )
        if existing.get("source_sha256") != source_hash:
            raise ValueError(
                f"Run {run_name!r} was created with different package source code. "
                "Use a new run name."
            )
    else:
        manifest = {
            "run_name": run_name,
            "status": "prepared",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config_hash,
            "source_sha256": source_hash,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(),
        }
        _atomic_text(run.manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _atomic_text(run.config_path, yaml.safe_dump(resolved, sort_keys=False))
    return run


def finalize_run(run: AnalysisRun, batch_summary: pd.DataFrame) -> None:
    """Record batch completion without changing the immutable run definition."""

    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    counts = batch_summary.get("status", pd.Series(dtype=str)).value_counts()
    failed = int(counts.get("failed", 0))
    manifest.update(
        {
            "status": "completed" if failed == 0 else "completed_with_failures",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_recordings": len(batch_summary),
            "n_ok": int(counts.get("ok", 0)),
            "n_failed": failed,
        }
    )
    _atomic_text(run.manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
