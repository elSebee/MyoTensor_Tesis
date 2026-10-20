# Preregistro del análisis — Experimentación v2

**Fecha de registro:** 2026-10-19
**Estado:** se fija ANTES de re-ejecutar los Exp 01–07 v2. Commitear este archivo antes de generar cualquier resultado v2. Cualquier cambio posterior se anota en la sección "Desviaciones", con fecha y motivo.

---

## 1. Datos

- 10 participantes sanos (`S01–S10`), 2 músculos (ED y FDS), una sesión por músculo y sujeto, 10 sets × 3 gestos activos (Palma, Puño, Paz) + Reposo.
- Excluidos: `FDS/S11` y `FDS/bishoo` (no tienen registro ED pareado).
- Ventana W = 250 ms, paso S = 50 ms, fs = 1 kHz.
- Particiones (`src/splits.py`), siempre por set completo y en orden cronológico:
  - intra: entrenamiento = sets 1–7, prueba = sets 8–10;
  - calibración k: sets 1…k (k = 1, 2, 3), prueba fija = sets 4–10;
  - LOSO: los 9 sujetos restantes; prueba = sets 4–10 del sujeto excluido.
- La configuración de preprocesamiento (steady-state/completa × %MVC/sin MVC) se decide en el Exp 01 v2 con el criterio de la sección 5. No se cambia después.

## 2. Modelos y regímenes

- Modelos: SVM (RBF), RF, CNN-LSTM, CNN-TCN.
- Regímenes de calibración (Exp 05 v2):
  - **R0** zero-shot: modelo LOSO sin datos del usuario.
  - **R1** adaptado: DL = LOSO + fine-tuning con los sets 1…k; ML = reentrenamiento con los 9 sujetos + los sets 1…k del usuario.
  - **R2** personalizado: entrenado solo con los sets 1…k del usuario.
- Semillas: 42 para todo; DL se repite con las semillas {42, 43, 44} y se usa la media por sujeto.

## 3. Métrica primaria

**F1-macro por sujeto** sobre los sets 4–10, en el régimen **R1, k = 1**, para cada músculo por separado. Se calcula offline (PC); la versión medida en el dispositivo (Exp 06 v2) se reporta como confirmatoria.

## 4. Hipótesis y contrastes

### 4.1 Contraste primario (hipótesis de la tesis)
- H1: Deep Learning tiene un F1-macro mayor que Machine Learning clásico.
- Por sujeto: F1_DL = media(CNN-TCN, CNN-LSTM); F1_ML = media(SVM, RF).
- Prueba pareada bilateral (N = 10), α = 0,05, una prueba por músculo (2 pruebas), corrección **Holm** sobre las 2.
- **Equivalencia (TOST)** con margen **±5 pp** (α = 0,05; equivale a que el IC90 de la diferencia esté dentro de ±5 pp).
  - Justificación del margen: la DE inter-sujeto calibrada observada en v1 fue de ~8–10 pp; 5 pp es la mitad de esa variabilidad y menor que la diferencia de desempeño que un usuario percibe en control discreto de 4 clases.
- Lectura de resultados:
  - p_Holm < 0,05 y Δ > 0 → evidencia de superioridad de DL;
  - TOST significativo → paradigmas equivalentes dentro de ±5 pp;
  - ninguno de los dos → no concluyente.
  - Nunca se escribirá "hipótesis refutada" solo porque p ≥ 0,05.

### 4.2 Contrastes secundarios (corrección Holm dentro de cada familia)
- **F1 · Sitio muscular:** ED vs FDS por modelo (4) + promedio de modelos (1), en R1 con k = 1; y ED vs FDS en R0 (1). Total: 6 pruebas.
- **F2 · Efecto de la calibración:** R1 (k = 1) vs R0, por modelo y músculo (8 pruebas).
- **F3 · Modelo a modelo:** las 6 comparaciones pareadas entre los 4 modelos, por músculo (12 pruebas).

### 4.3 Procedimiento estadístico (igual para todo contraste)
1. Diferencias pareadas por sujeto; Shapiro-Wilk sobre las diferencias.
2. Si p_Shapiro > 0,05 → t pareada; si no → Wilcoxon de rangos con signo (exacto).
3. Siempre bilateral. No se usan pruebas unilaterales.
4. Se reporta: n, medias, Δ medio, IC95 de Δ (t; bootstrap BCa con 10 000 réplicas si no hay normalidad), estadístico, p, p_Holm, tamaño de efecto (d_z de Cohen para t; r rango-biserial para Wilcoxon) y la prueba usada.

## 5. Criterios de decisión fijados de antemano

- **Exp 01 v2 (preprocesamiento):** se elige la configuración con mayor F1 medio en LOSO + calibración (R1, k = 1), promediando modelos y músculos. Si la diferencia con la alternativa más simple (completa y/o sin MVC) es < 1 pp, se elige la más simple.
- **Exp 02 v2 (tuning):** se elige la configuración con mayor F1 medio en CV agrupada por sujeto. Las configuraciones DL que no convierten a INT8 completo sin Flex ops quedan excluidas antes de comparar.
- **Viabilidad en el dispositivo:** un modelo se declara viable si (a) cabe en la partición de aplicación con OTA dual, (b) t_dsp + t_inf < 50 ms (p99 medido en el ESP32), (c) F1 medio en el dispositivo ≥ 80 % (R1, k = 1, ED o FDS) y (d) concordancia dispositivo–PC ≥ 99 %. Los modelos no viables no entran a la matriz de decisión.
- **Latencia de respuesta (lazo cerrado):** objetivo ≤ 300 ms (Englehart & Hudgins, 2003); se reporta también frente a 100–125 ms (Farrell & Weir, 2007). No se atribuye ningún umbral a Nyquist ni a Shannon.
- **Matriz de decisión:** criterios = F1 en el dispositivo, latencia de respuesta, SRAM total y Flash. Pesos: 40/25/20/15. Normalización min-max entre los modelos viables. Se reporta además un análisis de sensibilidad (10 000 vectores de pesos Dirichlet(1,1,1,1)) con el % de veces que gana cada modelo.

## 6. Qué NO se hará

- Reportar F1 de un solo sujeto (S01) como representativo del sistema.
- Elegir la prueba (uni/bilateral, paramétrica o no) después de ver los datos.
- Escribir a mano latencias, memorias o puntajes: todo sale de `results_io.save_result()` o de los logs del banco HIL.

## 7. Desviaciones

| Fecha | Cambio | Motivo |
|---|---|---|
| — | — | — |
