"""Build the minimal data package needed by notebook 02 and PSD analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


RUN_NAME = "gap_sensitivity_000s_gssc_yasa_sleepfm_cpu_v1"
NOTEBOOK = Path("notebooks/02_hypnodensity_analysis.ipynb")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(project_root: Path, destination: Path) -> None:
    source_recordings = project_root / "outputs" / RUN_NAME / "recordings"
    if not source_recordings.is_dir():
        raise FileNotFoundError(f"Missing source recordings directory: {source_recordings}")
    if not (project_root / NOTEBOOK).is_file():
        raise FileNotFoundError(f"Missing notebook: {project_root / NOTEBOOK}")

    hypnodensity = sorted(source_recordings.glob("*_inner_hypnodensities.parquet"))
    spectra = sorted(source_recordings.glob("*_inner_spectra.parquet"))
    if not hypnodensity or not spectra:
        raise FileNotFoundError("The source run does not contain hypnodensity and spectra tables.")

    hypno_ids = {path.name.removesuffix("_inner_hypnodensities.parquet") for path in hypnodensity}
    spectra_ids = {path.name.removesuffix("_inner_spectra.parquet") for path in spectra}
    if hypno_ids != spectra_ids:
        raise ValueError(f"Hypnodensity/PSD recording mismatch: {hypno_ids ^ spectra_ids}")

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    notebook_target = destination / NOTEBOOK
    notebook_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(project_root / NOTEBOOK, notebook_target)

    target_recordings = destination / "outputs" / RUN_NAME / "recordings"
    target_recordings.mkdir(parents=True, exist_ok=True)
    manifest_files: list[dict[str, object]] = []
    for source in [*hypnodensity, *spectra]:
        target = target_recordings / source.name
        shutil.copy2(source, target)
        manifest_files.append({
            "path": target.relative_to(destination).as_posix(),
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
        })

    # The notebook writes its results into this location. The marker makes the
    # directory survive copying/zipping without shipping any previous results.
    figures = destination / "outputs" / "hypnodensity_analysis_000s_gssc_v3" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    (figures / ".gitkeep").touch()

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": RUN_NAME,
        "notebook": NOTEBOOK.as_posix(),
        "n_recordings": len(hypno_ids),
        "included_data": {
            "hypnodensities": len(hypnodensity),
            "spectra_psd": len(spectra),
        },
        "excluded_data": ["features", "blocks", "staging_qc", "batch_summary", "file_selection", "raw EEG"],
        "files": manifest_files,
    }
    (destination / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    requirements = "\n".join([
        "jupyterlab",
        "numpy",
        "pandas>=2.1",
        "pyarrow>=14",
        "matplotlib",
        "seaborn",
        "statsmodels>=0.14",
        "",
    ])
    (destination / "requirements-transfer.txt").write_text(requirements, encoding="utf-8")

    readme = f"""# Transfer pack: notebook 02 + PSD

Este paquete contiene la unidad mínima para ejecutar `notebooks/02_hypnodensity_analysis.ipynb`
y trabajar con la PSD ya calculada, sin regenerar EEG, épocas ni features.

Contenido:

- `outputs/{RUN_NAME}/recordings/*_inner_hypnodensities.parquet`: datos que lee el notebook 02.
- `outputs/{RUN_NAME}/recordings/*_inner_spectra.parquet`: PSD/Welch por electrodo, época y frecuencia.
- `MANIFEST.json`: inventario, tamaños y SHA-256 de cada tabla.
- `requirements-transfer.txt`: dependencias del notebook.

No se incluyen EEG crudo, `features`, `blocks`, `staging_qc`, modelos ni resultados previos.

## Uso

1. Descomprime este directorio.
2. Entra en la carpeta descomprimida y crea un entorno Python.
3. Instala `pip install -r requirements-transfer.txt`.
4. Abre `notebooks/02_hypnodensity_analysis.ipynb` desde la raíz del paquete y ejecuta todas las celdas.

Las tablas PSD se pueden cargar con:

```python
from pathlib import Path
import pandas as pd

psd = pd.concat(
    [pd.read_parquet(path) for path in Path("outputs/{RUN_NAME}/recordings").glob("*_inner_spectra.parquet")],
    ignore_index=True,
)
```

Las columnas PSD principales son `frequency_hz` y `power`; las columnas de identificación
incluyen recording, bloque, época y electrodo.
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path("archive/transfer_packs/02_hypnodensity_psd_000s"))
    args = parser.parse_args()
    build(Path.cwd().resolve(), args.destination.resolve())
    print(f"Transfer pack written to {args.destination.resolve()}")


if __name__ == "__main__":
    main()
