"""
=============================================================
 postprocess.py — Pipeline de post-proceso del CSV maestro
=============================================================
 Etapa A — Onset dinamico (reemplaza el offset fijo de 250ms)
   Usa una ventana deslizante de RMS sobre la columna 'filtered'.
   El umbral de activacion viene de la calibracion del sujeto
   (onset_threshold_v) almacenado en el metadata JSON.
   Fallback: si no hay calibracion, calcula el umbral desde los
   segmentos de reposo del propio CSV.

 Etapa B — Control de calidad por repeticion (valid_flag)
   Para cada par (gesto, repeticion) calcula:
     - duracion activa (muestras con restimulus != 0)
     - energia RMS del segmento
   Compara contra la mediana de las N repeticiones del mismo gesto
   usando MAD (Median Absolute Deviation). Si alguna feature supera
   2*MAD del resto, la repeticion se marca con valid_flag = 0.

 Punto de entrada:
   run_postprocess(csv_path, meta_path) → csv_path

 Columnas finales del CSV:
   timestamp_us, raw, centered, filtered, voltage_v,
   emg_norm, stimulus, repetition, restimulus, valid_flag
=============================================================
"""

import json
import sys

import numpy as np
import pandas as pd


# ── Calculo de RMS deslizante ──────────────────────────────────

def compute_rms(signal: np.ndarray, window: int) -> np.ndarray:
    """RMS deslizante centrado, vectorizado con pandas rolling.

    Args:
        signal: Array 1D de la señal.
        window: Numero de muestras de la ventana.

    Returns:
        Array del mismo tamaño con los valores RMS.
    """
    return np.sqrt(
        pd.Series(signal.astype(np.float64) ** 2)
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


# ── Etapa A: restimulus por onset dinamico ─────────────────────

def compute_restimulus(
    stim: np.ndarray,
    rms: np.ndarray,
    threshold_v: float,
    min_active: int,
) -> np.ndarray:
    """Calcula restimulus usando RMS dinamico en lugar de offset fijo.

    Para cada segmento activo (stim != 0):
      - Busca hacia adelante el primer sample donde rms > threshold_v → onset real
      - Busca hacia atras el ultimo sample donde rms > threshold_v  → offset real
      - Si la duracion activa < min_active, el segmento se descarta (restim = 0)

    Args:
        stim:        Array de stimulus (etiqueta del cue visual).
        rms:         Array de RMS de la señal.
        threshold_v: Umbral de voltaje para considerar gesto activo.
        min_active:  Minimo de muestras activas para validar un gesto.

    Returns:
        Array restimulus con las mismas etiquetas que stim, pero con
        bordes recortados segun la energia real de la señal.
    """
    restim = np.zeros_like(stim)

    changes    = np.where(np.diff(stim) != 0)[0] + 1
    boundaries = np.concatenate(([0], changes, [len(stim)]))

    for i in range(len(boundaries) - 1):
        seg_start = boundaries[i]
        seg_end   = boundaries[i + 1]
        val       = stim[seg_start]

        if val == 0:
            continue

        # Buscar onset real (hacia adelante)
        onset = seg_end   # default: no onset encontrado
        for j in range(seg_start, seg_end):
            if rms[j] > threshold_v:
                onset = j
                break

        # Buscar offset real (hacia atras)
        offset = onset    # default: misma posicion → duracion 0
        for j in range(seg_end - 1, onset - 1, -1):
            if rms[j] > threshold_v:
                offset = j + 1
                break

        # Filtrar por duracion minima
        if (offset - onset) >= min_active:
            restim[onset:offset] = val

    return restim


# ── Etapa B: valid_flag por control de calidad MAD ─────────────

def compute_valid_flags(
    df: pd.DataFrame,
    signal_col: str = "emg_norm",
    stim_col: str   = "restimulus",
    rep_col: str    = "repetition_id",
) -> np.ndarray:
    """Marca repeticiones atipicas usando MAD sobre duracion y energia.

    Compara cada repeticion de un mismo gesto contra la distribucion
    de las N repeticiones. Usa MAD (robusto a outliers) en lugar de
    desviacion estandar.

    Una repeticion es invalida (valid_flag = 0) si:
      |duracion - mediana| > 2 * MAD_duracion
      O
      |energia_rms - mediana| > 2 * MAD_energia

    Args:
        df:         DataFrame con el CSV.
        signal_col: Columna de señal para calcular energia.
        stim_col:   Columna de etiqueta (restimulus).
        rep_col:    Columna de numero de repeticion.

    Returns:
        Array de enteros (1=valida, 0=invalida) de mismo largo que df.
    """
    valid    = np.ones(len(df), dtype=np.int8)
    restim   = df[stim_col].values
    rep      = df[rep_col].values
    sig_col  = signal_col if signal_col in df.columns else "filtered"
    signal   = df[sig_col].values

    gesture_ids = [g for g in np.unique(restim) if g != 0]

    for gid in gesture_ids:
        reps_in_gesture = np.unique(rep[restim == gid])
        reps_in_gesture = reps_in_gesture[reps_in_gesture > 0]

        if len(reps_in_gesture) < 3:
            # Con menos de 3 reps no hay robustez estadistica suficiente
            continue

        masks     = []
        durations = []
        energies  = []

        for r in reps_in_gesture:
            mask = (restim == gid) & (rep == r)
            masks.append(mask)
            seg = signal[mask]
            durations.append(float(np.sum(mask)))
            energies.append(float(np.sqrt(np.mean(seg ** 2))) if len(seg) > 0 else 0.0)

        durations = np.array(durations)
        energies  = np.array(energies)

        med_dur    = np.median(durations)
        med_energy = np.median(energies)
        mad_dur    = np.median(np.abs(durations - med_dur))
        mad_energy = np.median(np.abs(energies  - med_energy))

        for mask, dur, energy in zip(masks, durations, energies):
            bad_dur    = (mad_dur    > 0) and (abs(dur    - med_dur)    > 2 * mad_dur)
            bad_energy = (mad_energy > 0) and (abs(energy - med_energy) > 2 * mad_energy)
            if bad_dur or bad_energy:
                valid[mask] = 0

    return valid


# ── Punto de entrada principal ─────────────────────────────────

def run_postprocess(csv_path: str, meta_path: str) -> str:
    """Pipeline completo: restimulus dinamico + valid_flag.

    Lee los parametros de calibracion y onset desde el metadata JSON.
    Si no hay calibracion disponible, calcula el umbral desde el
    propio CSV usando los segmentos de reposo (stimulus == 0).

    Args:
        csv_path:  Ruta al CSV de la sesion.
        meta_path: Ruta al JSON de metadata de la sesion.

    Returns:
        Ruta al CSV modificado (mismo que la entrada).
    """
    # ── Cargar metadata ───────────────────────────────────────
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    calibration = meta.get("calibration", {})
    onset_cfg   = meta.get("onset_config", {})
    fs_hz       = int(meta.get("fs_hz", 1000))

    rms_window_ms  = int(onset_cfg.get("rms_window_ms", 100))
    min_active_ms  = int(onset_cfg.get("min_active_ms", 200))
    rms_window     = max(1, int(rms_window_ms * fs_hz / 1000))
    min_active     = max(1, int(min_active_ms * fs_hz / 1000))

    # ── Cargar CSV ────────────────────────────────────────────
    df = pd.read_csv(csv_path)

    if "stimulus" not in df.columns:
        raise ValueError(f"El CSV no contiene columna 'stimulus': {csv_path}")

    sig_col = "emg_norm" if "emg_norm" in df.columns else "filtered"
    signal  = df[sig_col].values.astype(np.float64)
    stim    = df["stimulus"].values

    # ── Determinar umbral de onset ────────────────────────────
    threshold_v = calibration.get("onset_threshold_v", None)

    if threshold_v is None or threshold_v <= 0:
        # Fallback: calcula umbral desde los propios segmentos de reposo
        rest_signal = signal[stim == 0]
        if len(rest_signal) > 0:
            noise_mean  = float(np.mean(np.abs(rest_signal)))
            noise_std   = float(np.std(np.abs(rest_signal)))
            threshold_v = noise_mean + 3.0 * noise_std
        else:
            threshold_v = float(np.percentile(np.abs(signal), 25))
        print(f"[POST] Umbral auto-calculado desde reposo: {threshold_v:.4f}V")
    else:
        print(f"[POST] Umbral desde calibracion: {threshold_v:.4f}V")

    # ── Etapa A: RMS y restimulus ─────────────────────────────
    rms    = compute_rms(signal, rms_window)
    restim = compute_restimulus(stim, rms, threshold_v, min_active)
    df["restimulus"] = restim

    # ── Etapa B: valid_flag ───────────────────────────────────
    df["valid_flag"] = compute_valid_flags(
        df,
        signal_col=sig_col,
        rep_col="repetition_id" if "repetition_id" in df.columns else "repetition",
    )

    # ── Guardar ───────────────────────────────────────────────
    df.to_csv(csv_path, index=False)

    # ── Estadisticas de resumen ───────────────────────────────
    total_active_stim  = int(np.sum(stim != 0))
    total_active_restim = int(np.sum(restim != 0))
    trimmed            = total_active_stim - total_active_restim
    n_invalid          = int(np.sum(df["valid_flag"] == 0))
    n_valid            = int(np.sum(df["valid_flag"] == 1))

    print(f"[POST] restimulus: {total_active_restim} muestras activas "
          f"({trimmed} recortadas por onset dinamico)")
    print(f"[POST] valid_flag: {n_valid} validas | {n_invalid} invalidas")

    return csv_path


# ── CLI standalone ─────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python postprocess.py <archivo.csv> <archivo_metadata.json>")
        sys.exit(1)

    run_postprocess(sys.argv[1], sys.argv[2])
    print(f"[OK] {sys.argv[1]} actualizado con restimulus y valid_flag")
