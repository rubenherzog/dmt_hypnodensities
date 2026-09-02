# Transfer pack: notebook 02 + PSD

Este paquete contiene la unidad mínima para ejecutar `notebooks/02_hypnodensity_analysis.ipynb`
y trabajar con la PSD ya calculada, sin regenerar EEG, épocas ni features.

Contenido:

- `outputs/gap_sensitivity_000s_gssc_yasa_sleepfm_cpu_v1/recordings/*_inner_hypnodensities.parquet`: datos que lee el notebook 02.
- `outputs/gap_sensitivity_000s_gssc_yasa_sleepfm_cpu_v1/recordings/*_inner_spectra.parquet`: PSD/Welch por electrodo, época y frecuencia.
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
    [pd.read_parquet(path) for path in Path("outputs/gap_sensitivity_000s_gssc_yasa_sleepfm_cpu_v1/recordings").glob("*_inner_spectra.parquet")],
    ignore_index=True,
)
```

Las columnas PSD principales son `frequency_hz` y `power`; las columnas de identificación
incluyen recording, bloque, época y electrodo.
