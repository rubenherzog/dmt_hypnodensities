# Informe de validación científica y técnica

Fecha: 2026-09-01  
Repositorio: `dmt_hypnodensities`  
Datos: `/media/rherzoga/T7 Touch/DMTx/Clean` (solo lectura)

> **Nota de vigencia.** Las fases y conteos siguientes documentan la validación del run
> anterior `main_70s_sleepfm_cpu_v1`. La auditoría posterior detectó que ese epoching podía
> atravesar gaps. La implementación vigente crea épocas dentro de runs estrictamente
> continuos, agrupa contexto con 90 s y usa GSSC/YASA/SleepFM. Sus resultados mínimos y conteos se
> documentan en `REPRODUCTION.md`; este informe histórico se conserva y no es ground truth
> para la nueva corrida.

## Entorno

- Python 3.10.21
- numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, h5py 3.16.0
- MNE 1.12.1, YASA 0.7.0, PyArrow 25.0.1
- scikit-learn 1.7.2, specparam 2.0.0rc7, pycatch22 0.5.0
- joblib 1.6.0, pytest 9.1.1, ruff 0.16.5
- `pip check`: sin dependencias rotas.

`ruff check src tests`: correcto. `ruff check .` también inspecciona el checkout externo
de SleepFM y devuelve 229 avisos de ese código de terceros; no se modificó.

## Resultados por fase

### Fase 1 — Estado reproducible: superada

`pytest -q`: 20 passed, 6 warnings, 5.7 s. Las advertencias corresponden a:

- `InconsistentVersionWarning` al cargar el `LabelEncoder` de YASA serializado con
  scikit-learn 0.24.2 y ejecutado con 1.7.2.
- dos avisos de PyTorch sobre `TransformerEncoder` en el checkout oficial de SleepFM.

No se reinstalaron ni actualizaron dependencias.

### Fase 2 — Selección y carga: superada

La carpeta contiene 37 archivos candidatos y `prefer_d5` selecciona 36. Se verificó:

- placebo: D1, o el menor índice disponible;
- DMT: D5, o el mayor índice disponible;
- P11-DMT: se selecciona `DMTCI_P11D5_inner.mat` y se descarta
  `DMTCI_P11D2_inner.mat`;
- subject, session y condition se extraen correctamente de los nombres;
- los tres MAT de referencia tienen 185 canales, E6 está en el índice físico 5 y la
  frecuencia inferida es 500 Hz.

### Fase 3 — Bloques, épocas y labels: superada

Con gaps de 70 s y épocas de 30 s, cada archivo de referencia produce un bloque:

| archivo | bloques | épocas | before | after | late | outside |
|---|---:|---:|---:|---:|---:|---:|
| P07D1 | 1 | 101 | 10 | 46 | 21 | 24 |
| P07D5 | 1 | 100 | 11 | 36 | 26 | 27 |
| P12D2 | 1 | 59 | 10 | 37 | 0 | 12 |

Los conteos coinciden con las referencias. Los bloques no se cortan en fronteras
experimentales. Se corrigió un bug de nomenclatura: el tiempo fuera de las ventanas ahora
usa el label público `outside` en lugar de `outside_analysis`.

### Fase 4 — Features: superada en P07D1/E6

Se recalculó la extracción completa sobre las mismas 101 épocas físicas:

- 101 filas de features;
- 21 features base de YASA, sin features contextuales ni `time_hour`/`time_norm`;
- 22 features Catch22 y 4 features specparam;
- 101.101 filas de espectro, con 1.001 frecuencias por época;
- claves únicas y cero valores no finitos en esta ejecución;
- spectrum y specparam reutilizan el mismo Welch interno.

### Fase 5 — Staging: superada para GSSC/YASA; SleepFM con cobertura sintética/local

En los tres MAT y E6, GSSC y YASA devolvieron una llamada por bloque y exactamente
101/100/59 filas por stager. Las probabilidades fueron finitas, estuvieron en [0, 1],
sumaron 1 con error máximo 1.9e-7 y mantuvieron claves únicas.

Los tests cubren bloques de una época, bloques con varias épocas y sets SleepFM mono- y
multicanal mediante predictor de contrato; el smoke test local del checkpoint oficial
cubre una época real del adaptador. SleepFM está deshabilitado en la configuración principal.

### Fase 6 — Persistencia y ensamblaje: superada con outputs existentes

`assemble_outputs` cargó únicamente los outputs canónicos indicados por
`batch_summary.csv`, sin incorporar el parquet comparativo de tres stagers. Validó claves
de features y bloques y no encontró duplicados ni referencias de bloques desconocidas.
Los outputs existentes de features contienen solo tablas, sin señales ni tensores.

### Fase 7 — Stats y plots: superada

Sobre las 101 épocas de P07D1/E6 ya calculadas:

- entropía de Shannon manual frente a la implementación: diferencia máxima 0;
- Pearson y Spearman ejecutados;
- ajuste FDR-BH presente;
- correlaciones alineadas por recording, block, epoch y channel_set;
- resumen de grupos generado;
- plots de entropía y heatmap generados con backend no interactivo `Agg`.

### Fase 8 — Batch completo: no superada; fallo diagnosticado

Se intentó `run_batch` sobre las 36 grabaciones con la configuración vigente (`n_jobs=-1`),
escribiendo en `outputs_validation_batch/` para preservar `outputs/`. Tras unos 3 min 26 s,
los workers concurrentes de GSSC agotaron la GPU disponible de 4 GB. Se observaron errores
de asignación CUDA/OOM; el proceso fue detenido de forma controlada y no dejó outputs
parciales en el directorio de validación.

Esto es una limitación operativa reproducible de la concurrencia configurada, no una
corrección científica. No se cambiaron límites de GPU, BLAS, CPU o threads, ni se añadió
un modo legacy. Para completar el batch hace falta decidir explícitamente una estrategia
de ejecución que no lance simultáneamente tantos inferidores GSSC en la GPU.

## Correcciones y tests añadidos

- `src/dmt_hypnodensities/config.py`: label externo normalizado a `outside`.
- `tests/test_io_epochs.py`: regresión para el label público `outside`.

No se modificaron los datos originales ni se sobrescribieron los outputs existentes.

## Estado final

Las fases 1–7 cumplen los contratos comprobados. La selección, carga, bloques, épocas,
features, staging de GSSC/YASA, persistencia, ensamblaje, estadísticas y plots son
reproducibles en el entorno actual. La validación de cohorte completa queda pendiente por
el OOM de GPU al usar la configuración paralela vigente; no debe resolverse alterando
silenciosamente la configuración o las decisiones científicas.

## Pruebas de performance y paralelización

También se midió GSSC sobre P07D1/E6, un bloque de 101 épocas:

| ejecución | tiempo total |
|---|---:|
| una llamada con GPU | 3.59 s |
| una llamada con CPU (`CUDA_VISIBLE_DEVICES=''`) | 3.50 s |

La GPU no fue más rápida en este tamaño porque la carga de modelos y preparación domina.
El proceso CPU tuvo un RSS máximo aproximado de 1.43 GB y el proceso GPU de 1.84 GB.

En dos grabaciones, sin GPU y sin persistencia, la ejecución pasó de 20.78 s con un worker
a 10.92 s con dos workers (1.9x). Esto indica que el paralelismo externo por grabación es
útil para CPU, pero no justifica lanzar 20 procesos indiscriminadamente: cada worker carga
modelos y arrays grandes y aumenta la presión de RAM e I/O.

La jerarquía recomendada para esta máquina (20 CPUs lógicas, 14 núcleos físicos y una GPU
de 3.68 GiB) es:

1. GPU: un único worker GSSC/SleepFM, procesando grabaciones y bloques en serie; no usar
   joblib externo para inferencia GPU.
2. CPU-only: paralelismo externo por grabación, inicialmente 4 workers y después probar
   6–8 si la RAM lo permite; mantener `n_jobs=1` dentro de cada worker para evitar pools
   anidados.
3. Features CPU: pueden compartir esa capa externa; no combinar simultáneamente una capa
   externa por grabación con otra interna por época/canal.

La comparación definitiva GPU frente a CPU para toda la cohorte requiere desacoplar staging
GPU y features CPU o añadir una selección explícita de dispositivo. No se aplicó ese cambio
durante esta validación.

Posteriormente se decidió explícitamente ejecutar todo el pipeline en CPU. GSSC ahora crea
`EEGInfer(use_cuda=False)` y SleepFM devuelve siempre `torch.device("cpu")`; se añadió un
test de regresión para este último contrato. El paralelismo externo por grabación queda así
seguro para la cohorte.
