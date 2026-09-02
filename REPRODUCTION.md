# Auditoría de reproducción y continuidad

La lógica histórica se utilizó únicamente como referencia interna. No existe un modo
`legacy` en la API ni en la configuración de producción.

## Qué se reproduce

- La lectura FieldTrip obtiene la señal física indicada por `label`, usando el índice real
  de cada electrodo.
- Sobre una misma matriz `epoch × channel × sample`, el bandpower relativo coincide con la
  llamada a `yasa.bandpower` del notebook (diferencia absoluta 0 en E6 de tres MAT reales).
- En señal totalmente continua, la implementación discontinua-aware de YASA reproduce sus
  probabilidades oficiales exactamente.
- La selección `prefer_d5` conserva la política de archivos del notebook y se persiste en
  `file_selection.csv`.

## Corrección científica del epoching

El pipeline anterior concatenaba señal separada por gaps y dividía después cada bloque en
épocas. Una época podía contener el final de un segmento y el comienzo de otro, como si las
muestras fueran adyacentes. Ese comportamiento no se reproduce porque crea una época
artefactual.

El pipeline actual aplica dos niveles distintos:

1. separa runs ante toda discontinuidad de timestamps;
2. crea solo épocas completas de 30 s dentro de cada run y descarta su cola incompleta;
3. agrupa los runs ya epochados si el gap efectivo es `<= gap_tolerance_seconds`;
4. permite contexto entre esas épocas, pero filtra y extrae features siempre por run.

Los labels `before`, `after`, `late` y `outside` se asignan después y nunca producen cortes
artificiales.

## Chequeo real mínimo de la regla híbrida a 90 s

Se evaluó E6, sin recalcular features pesados, en P07D1 y P07D5. Los outputs están en
`outputs/validation_hybrid90_e6/`.

| Archivo | grupos contextuales | runs con épocas | épocas válidas |
|---|---:|---:|---:|
| P07D1 | 3 | 31 | 50 |
| P07D5 | 2 | 19 | 37 |

GSSC y YASA devolvieron todas esas épocas. YASA se ejecuta desde 2 épocas; la recomendación
de 5 minutos de la librería no se interpreta como requisito. Las diferencias de las
hipnodensidades promedio respecto al run anterior son esperables porque ahora cambia el
conjunto de épocas y porque el preprocesamiento ya no atraviesa gaps. No deben interpretarse
como una comparación pareada época a época.

La tabla final de este chequeo es
`outputs/validation_hybrid90_e6/mean_hypnodensity_comparison_yasa_final.csv`; el parquet y
QC por grabación se conservaron sin sobrescribir el batch anterior.

## Corrida anterior

`outputs/main_70s_sleepfm_cpu_v1/` se preserva como evidencia del análisis completado, pero
sus épocas se construyeron antes de esta corrección y no son ground truth. La nueva corrida
usa el nombre `main_hybrid90_gssc_yasa_sleepfm_cpu_v1` para impedir que ambos resultados se
mezclen. SleepFM fue validado adicionalmente sobre los 3 grupos y 50 épocas de P07D1/E6:
preprocesó 31 runs por separado y devolvió las 50 hipnodensidades sin fallos.
