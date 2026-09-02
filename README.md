# dmt_hypnodensities

Pipeline compacto para extraer hipnodensidades y features EEG del estudio de infusión
continua de DMT.

## Principios del diseño

- Los MAT originales se leen desde una ruta externa y nunca se copian al repositorio.
- Cada archivo se carga una vez por ejecución.
- Primero se separan *runs* estrictamente continuos ante cualquier muestra faltante.
- Las épocas completas de 30 segundos se crean dentro de cada run: ninguna época contiene
  muestras de ambos lados de un gap y la cola incompleta de cada run se descarta.
- Después se agrupan runs consecutivos separados por hasta 90 segundos (parametrizable).
  El grupo aporta contexto temporal a los stagers, pero nunca se presenta como señal continua.
- `before`, `after` y `late` son labels de época; nunca dividen artificialmente un bloque.
- El label de una época se determina provisionalmente con la mediana de sus timestamps
  originales, lo que conserva una asignación única incluso cuando el bloque cruza ventanas.
- Staging y features se calculan en una misma ejecución.
- No se guardan señales, épocas, tensores ni embeddings.
- Los resultados tabulares grandes se guardan en Parquet y las tablas finales en CSV.
- Antes del procesamiento batch se aplica de forma explícita la misma selección de
  grabaciones del notebook original; la decisión queda registrada en una tabla auditable.

## Estado

La primera fase implementa y valida el núcleo compartido:

1. lectura FieldTrip/HDF5 con índices de canal correctos;
2. detección de bloques;
3. epoching y labels experimentales;
4. extracción conjunta de bandpower, espectros Welch, specparam y Catch22;
5. los 21 features base de `yasa.SleepStaging.get_features()`;
6. core común de staging para GSSC, YASA y SleepFM;
7. handler por grabación y persistencia tabular;
8. selección reproducible de archivos y ejecución batch con `joblib`;
9. ensamblaje validado de las tablas producidas;
10. estadísticas de hipnodensidad y una API pequeña de plots;
11. tests sintéticos y validación con archivos reales.

GSSC, YASA y SleepFM están conectados directamente. SleepFM preprocesa y codifica cada run
continuo por separado y concatena después sus embeddings para el head contextual, sin unir
waveforms a través de gaps. La corrida principal fija `compute.device: cpu`, por lo que
GSSC y SleepFM no seleccionan CUDA. El paquete no modifica límites
de CPU, BLAS, threads ni variables de entorno;
`joblib` controla el paralelismo entre grabaciones.

La auditoría contra el pipeline anterior, incluidas las diferencias deliberadas de
epoching, se documenta en [`REPRODUCTION.md`](REPRODUCTION.md). La lógica histórica no
forma parte de la API ni de la configuración del paquete.

## Persistencia

El pipeline no guarda señales ni épocas. Los únicos outputs persistentes serán:

- Parquet para features, staging y espectros;
- CSV para tablas finales y resúmenes;
- JSON para el manifiesto de ejecución;
- PNG/PDF para figuras.

El HDF5 se usa exclusivamente para leer los MAT externos de FieldTrip.

## Uso desde un notebook

```python
from dmt_hypnodensities import load_config, process_recording, save_recording_result

config = load_config("configs/analysis.yaml")
result = process_recording(
    "/media/rherzoga/T7 Touch/DMTx/Clean/DMTCI_P07D1_inner.mat",
    config,
)

result.features   # una fila por bloque, época y electrodo
result.hypnodensities  # probabilidades W/N1/N2/N3/R por stager
result.spectra    # una fila adicional por frecuencia
result.blocks     # QC compacto, sin señales
result.staging_qc # éxito, incompatibilidad o error por stager y canal

save_recording_result(result, config.output_dir, "P07D1")
```

El workflow completo está separado en tres notebooks delgados:

1. `notebooks/01_run_batch.ipynb`: única lectura de MAT, extracción conjunta y persistencia;
2. `notebooks/02_hypnodensity_analysis.ipynb`: Wilcoxon, LME, efectos y figuras;
3. `notebooks/03_feature_associations.ipynb`: join validado, asociaciones y figuras.

La corrida principal usa `RUN_NAME = "main_hybrid90_gssc_yasa_sleepfm_cpu_v1"`. Su directorio contiene
`resolved_config.yaml`, `manifest.json`, outputs por grabación, tablas y figuras. Reabrir
la misma corrida reutiliza resultados completos; cambiar su configuración bajo el mismo
nombre produce un error para impedir que se mezclen análisis incompatibles.

Con `channels=None` —el default— se lee `configs/egi257_to_1020_mapping.csv`, se aplica la
selección científica del estudio y se intersecta con los canales realmente disponibles en
el MAT. `electrode` conserva el nombre EGI correcto y `el_10-20` contiene su label estándar.
También puede pasarse `channels=["E6", "E8"]` para una selección explícita.

## Entorno

El entorno validado usa Python 3.10, YASA 0.7 y PyArrow. Desde el repositorio:

```bash
conda create -n dmt_hypnodensities python=3.10
conda activate dmt_hypnodensities
python -m pip install -e '.[features,gssc,sleepfm,stats,plots,dev]'
```

YASA distribuye actualmente un encoder serializado con scikit-learn 0.24.2; una versión
moderna de scikit-learn emite `InconsistentVersionWarning` al cargarlo. No se silencia esa
advertencia y los tests verifican cardinalidad, schema y normalización del resultado.

Los análisis se activan en `configs/analysis.yaml`. Aunque se pidan simultáneamente
`spectrum` y `specparam`, Welch se calcula una sola vez. De YASA se conservan únicamente
los 21 features base; se descartan las 42 transformaciones contextuales y las columnas
`time_hour` y `time_norm`.
La configuración exige por defecto 2 épocas para YASA. La recomendación de 5 minutos que
emite la librería no es una restricción técnica. Un grupo contextual de una sola época queda
como `insufficient_epochs`; GSSC y los demás análisis compatibles continúan.

## Contrato de staging

```python
from dmt_hypnodensities import stage_block

staging = stage_block(block, stagers=("gssc", "yasa"), n_jobs=-1)
staging.hypnodensities
staging.yasa_features
staging.qc
```

La tabla de hipnodensidades utiliza siempre `prob_W`, `prob_N1`, `prob_N2`, `prob_N3` y
`prob_R`, además de `stager`, `channel_set`, `electrode`, `block_id` y `epoch`. GSSC filtra
cada run continuo por separado y concatena después sus épocas para el modelo contextual.
YASA remuestrea, filtra y extrae los 21 features base dentro de cada run; luego reconstruye
sus features contextuales sobre el grupo. Así ambos aprovechan el contexto de gaps de hasta
90 s sin filtrar ni construir épocas a través de datos ausentes.

SleepFM puede seguir recibiendo un predictor alternativo con el contrato:

```python
predictor(block, channel_set) -> probabilidades_de_forma_(n_epochs, 5)
```

Cada llamada recibe un grupo de épocas. El adaptador local separa sus runs antes de filtrar,
remuestrear, normalizar y calcular embeddings. Solo los embeddings se concatenan, en orden,
antes de una única llamada al head contextual. Los conjuntos multicanal se declaran en
`staging.sleepfm.channel_sets`; sin esa opción se ejecuta el mismo análisis monoelectrodo
por canal que GSSC/YASA.

El predictor local se crea automáticamente si SleepFM está habilitado y
`staging.sleepfm.repository` apunta al checkout oficial. La revisión validada es
`zou-group/sleepfm-clinical@2bcbae04c3592f61352addb7ac3d4193f0a3ca25`:

```bash
git clone https://github.com/zou-group/sleepfm-clinical .external/sleepfm-clinical
git -C .external/sleepfm-clinical checkout 2bcbae04c3592f61352addb7ac3d4193f0a3ca25
```

`.external/` y los checkpoints están ignorados por Git. El código oficial tiene licencia
CC BY-NC 4.0. Su head de staging devuelve probabilidades cada 5 segundos en el orden
W/N1/N2/N3/R; el adaptador promedia cada grupo de seis probabilidades para obtener una
hipnodensidad de 30 segundos. El head contextual recibe en una sola llamada todos los
embeddings del bloque, incluso si el bloque contiene una sola época. Los embeddings base
se calculan en chunks de 5 minutos, como en el pipeline oficial, y solo se conservan en RAM.
El checkpoint publicado admite hasta 10 electrodos EEG dentro de un mismo `channel_set`.

## Batch

```python
from dmt_hypnodensities import load_config, run_batch

config = load_config("configs/analysis.yaml")
summary = run_batch(config)  # todos los *_inner.mat, selección automática
```

`run_batch` usa `joblib` a nivel de archivo, guarda cada resultado al terminar y escribe
`batch_summary.csv`. Antes de empezar aplica `data.file_selection_policy: prefer_d5`, que
reproduce exactamente la política de `WF_DMTx_GSSC_FEATURES_JOINT.ipynb`:

- se agrupan los archivos por sujeto y condición;
- para placebo se usa D1, o el menor D disponible si falta D1;
- para DMT se usa D5, o el mayor D disponible si falta D5;
- nunca se procesan dos archivos de la misma combinación sujeto–condición.

La selección, los candidatos descartados y la regla aplicada se escriben en
`file_selection.csv`. En los datos actuales esto selecciona 36 de 37 archivos: para P11-DMT
se conserva `DMTCI_P11D5_inner.mat` y se descarta `DMTCI_P11D2_inner.mat`. Esta política
selecciona grabaciones; no cambia la detección posterior de bloques ni elimina épocas.

Dentro de cada worker se usa `n_jobs=1` para evitar pools anidados. No se modifican límites
de CPU, BLAS, threads ni variables de GPU.

El batch muestra una barra `tqdm` por grabaciones completadas y el postfix indica la última
grabación, su número de bloques y su estado. Con `n_jobs=1` también se muestra una barra
interna por bloques; en paralelo se omite para evitar barras entremezcladas provenientes de
varios procesos. La salida repetitiva interna de GSSC se captura localmente, sin ocultar
excepciones ni el QC. Cada worker escribe `_status/<grabación>.json` de forma atómica: si el
kernel se interrumpe antes de crear `batch_summary.csv`, la próxima ejecución reutiliza las
grabaciones completas y reintenta únicamente las incompletas o fallidas.

Para una ejecución exploratoria que deliberadamente incluya todos los candidatos puede
usarse `data.file_selection_policy: all`. No es la configuración del análisis principal.

### Sensibilidad a la tolerancia de gaps

El script `scripts/run_gap_tolerance_sweep.py` repite íntegramente el workflow pesado del
notebook 01 para tolerancias de 0, 15, 30, 60, 90 y 120 segundos:

```bash
python scripts/run_gap_tolerance_sweep.py
```

Cada tolerancia usa una corrida inmutable y una carpeta distinta, por ejemplo
`outputs/gap_sensitivity_000s_gssc_yasa_sleepfm_cpu_v1/` y
`outputs/gap_sensitivity_090s_gssc_yasa_sleepfm_cpu_v1/`. No comparte resultados entre tolerancias.
El coordinador ejecuta cada tolerancia en un proceso Python nuevo y espera su finalización;
al salir el proceso, el sistema operativo libera modelos, arrays, memoria nativa y workers
antes de comenzar la siguiente. Una interrupción puede reanudarse con el mismo comando:
solo se reutilizan grabaciones completas dentro de su propia tolerancia.

## Ensamblaje, estadísticas y plots

```python
from dmt_hypnodensities import (
    add_hypnodensity_entropy,
    assemble_outputs,
    fit_mixed_models,
    pairwise_stager_correlations,
    paired_condition_wilcoxon,
    plot_condition_change_violins,
    plot_electrode_variance_violins,
    plot_entropy_distribution,
    plot_hypnodensity_condition_violins,
    plot_paired_condition_changes,
    plot_ranked_stage_features,
    plot_stage_feature_correlation_heatmap,
    plot_stager_correlation_heatmap,
    prepare_epoch_cohen_d,
    prepare_treatment_effects,
    prepare_within_condition_changes,
    stage_feature_effect_correlations,
    summarize_hypnodensities,
)

tables = assemble_outputs(config.output_dir)

hypnodensities = add_hypnodensity_entropy(tables.hypnodensities)
correlations = pairwise_stager_correlations(
    tables.hypnodensities,
    stagers=("yasa", "gssc", "sleepfm"),
)
summary = summarize_hypnodensities(tables.hypnodensities)

# Test pareado DMT-placebo por stager, ventana y stage. La unidad es el sujeto:
# primero se promedian sus épocas y electrodos dentro de cada condición.
wilcoxon = paired_condition_wilcoxon(tables.hypnodensities)

# Cambios within-condition: follow-up menos baseline, con dirección explícita.
changes = prepare_within_condition_changes(
    tables.hypnodensities,
    value_columns=("prob_W", "prob_N1", "prob_N2", "prob_N3", "prob_R"),
    delta_types=("abs", "rel", "logit"),
)

# LME equivalente al análisis principal de los notebooks:
# cambio logit ~ condición, intercepto aleatorio por sujeto y VC por electrodo.
lme = fit_mixed_models(
    changes[changes["contrast"].eq("before_to_after")],
    outcomes=tuple(f"prob_{stage}__delta_logit" for stage in ("W", "N1", "N2", "N3", "R")),
    fixed_effects="condition",
    variance_components={"electrode": "0 + C(electrode)"},
)

# Efecto neto usado por el notebook de asociaciones:
# [after-before en DMT] menos [after-before en placebo].
effects = prepare_treatment_effects(
    changes,
    value_columns=("prob_W", "prob_N1", "prob_N2", "prob_N3", "prob_R"),
    delta_types=("abs", "rel"),
)

# Cohen's d descriptivo entre distribuciones de épocas, con signo follow-up - baseline.
epoch_effect_sizes = prepare_epoch_cohen_d(tables.hypnodensities)

# Figuras canónicas de los notebooks, ahora con agregación explícita por sujeto.
fig_hypno, axes_hypno = plot_hypnodensity_condition_violins(
    tables.hypnodensities,
    wilcoxon_results=wilcoxon,
    stager="gssc",
)
fig_variance, axes_variance = plot_electrode_variance_violins(
    tables.hypnodensities,
    stager="gssc",
)
fig_effects, axes_effects = plot_condition_change_violins(
    epoch_effect_sizes,
    stager="gssc",
)
fig_pairs, axes_pairs = plot_paired_condition_changes(
    epoch_effect_sizes,
    stager="gssc",
    value_template="prob_{stage}__cohen_d",
    ylabel="Cohen's d",
)

fig_entropy, ax_entropy = plot_entropy_distribution(hypnodensities)
fig_correlations, ax_correlations = plot_stager_correlation_heatmap(correlations)
```

`assemble_outputs` se guía por las grabaciones exitosas de `batch_summary.csv`, carga solo
los outputs canónicos y valida claves y referencias entre bloques, features, espectros e
hipnodensidades. No busca archivos mediante sufijos ambiguos, por lo que no incorpora por
accidente tablas temporales o comparaciones previas.

Las correlaciones alinean estrictamente cada observación por grabación, bloque, época y
`channel_set`; calculan W/N1/N2/N3/R y entropía de Shannon, con ajuste FDR-BH por defecto.
La API inferencial reproduce la intención de los notebooks sin conservar sus ambigüedades:
Wilcoxon empareja por identificador de sujeto y no por posición de filas; todos los cambios
son `follow-up - baseline`; y los efectos netos son `DMT - placebo`. `fit_mixed_models`
permite además modelos de intensidad (`fixed_effects="condition * intensity_label"`) y
modelos de intercepto sobre efectos netos (`fixed_effects="1"`). Devuelve coeficientes,
errores estándar, estadísticos z, intervalos de confianza, convergencia, warnings, errores
y valores p ajustados, todos en una tabla serializable. Cuando los cambios se preparan
conjuntamente para stages y features, `stage_feature_effect_correlations` reproduce la
matriz exploratoria Pearson/Spearman del notebook, incluyendo el N pairwise y sin presentar
p-values exploratorios como evidencia confirmatoria.

La API de plots conserva las figuras centrales de los notebooks: violines DMT/placebo por
ventana, varianza entre electrodos, Cohen's d por contraste, pares conectados por sujeto,
estratificación por intensidad, heatmaps stage×feature, rankings centrados en REM y scatters
stage–feature. Los violines se construyen con medias por sujeto para no representar épocas o
electrodos correlacionados como observaciones independientes. Para el layout por intensidad
se usa `plot_hypnodensity_condition_violins(..., stratify_by="intensity_label")`; para las
asociaciones se usan `plot_stage_feature_correlation_heatmap` y
`plot_ranked_stage_features`. Todas las funciones reciben tablas ya calculadas, devuelven
`(figure, axes)`, no modifican el DataFrame y no guardan archivos implícitamente.

## Datos

La configuración inicial apunta a:

```text
/media/rherzoga/T7 Touch/DMTx/Clean
```

La carpeta contiene datos externos y no forma parte del repositorio.
