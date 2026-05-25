"""
=============================================================
 preprocess.py — Pipeline de preprocesamiento y onset dinámico
=============================================================
 Etapa A — Onset dinámico (re-stimulus)
   Usa la clase EMGPreprocessor con ventana deslizante RMS causal.
   Corrige el bug del original aplicando el umbral normalizado
   (onset_threshold_v / mvc_voltage_v) sobre la señal normalizada (emg_norm).
   De esta forma, la comparación matemática es 100% consistente.

 Etapa B — Control de calidad por repetición (valid_flag)
   Para cada par (gesto, repetición) calcula la duración activa y la energía.
   Usa MAD (Median Absolute Deviation) para invalidar outliers.

 Punto de entrada:
   run_preprocess(csv_path, meta_path) → csv_path
=============================================================
"""

import json
import sys
import numpy as np
import pandas as pd

class EMGPreprocessor:
    """
    Clase para el preprocesamiento de señales sEMG y detección automática de onset (re-stimulus).
    """
    def __init__(self, fs: float, mvc_voltage_v: float, onset_threshold_v: float,
                 rms_window_ms: float = 100.0, min_active_ms: float = 200.0):
        self.fs = fs
        self.mvc_voltage_v = mvc_voltage_v
        self.onset_threshold_v = onset_threshold_v
        
        # Umbral normalizado respecto a MVC (crucial para comparar con emg_norm)
        self.threshold_normalized = onset_threshold_v / mvc_voltage_v if mvc_voltage_v > 0 else 0.0
        
        # Convertimos los tiempos en milisegundos a número de muestras
        self.rms_window_samples = int(np.round(fs * rms_window_ms / 1000.0))
        self.min_active_samples = int(np.round(fs * min_active_ms / 1000.0))
        
        # Sanitización de ventanas mínimas
        self.rms_window_samples = max(1, self.rms_window_samples)
        self.min_active_samples = max(1, self.min_active_samples)
        
    def normalize_mvc(self, signal: np.ndarray) -> np.ndarray:
        """Paso 1: Normalización respecto al voltaje MVC."""
        return signal / self.mvc_voltage_v
        
    def compute_rms(self, signal: np.ndarray) -> np.ndarray:
        """Paso 2: Cálculo de RMS mediante ventana deslizante causal O(N) con cumsum."""
        x2 = signal ** 2
        w = self.rms_window_samples
        
        if signal.ndim == 1:
            cumsum = np.cumsum(np.insert(x2, 0, 0.0))
            rms = np.sqrt((cumsum[w:] - cumsum[:-w]) / w)
            return np.concatenate([np.zeros(w - 1), rms])
        else:
            cumsum = np.cumsum(np.insert(x2, 0, 0.0, axis=0), axis=0)
            rms = np.sqrt((cumsum[w:, :] - cumsum[:-w, :]) / w)
            padding = np.zeros((w - 1, signal.shape[1]))
            return np.concatenate([padding, rms], axis=0)


# ── Etapa A: restimulus por onset dinámico ─────────────────────

def detect_hysteresis_blocks(
    rms: np.ndarray,
    th_onset: float,
    th_offset: float,
    min_active: int,
    min_silence: int
):
    # 1. Hysteresis binarization
    n = len(rms)
    active = np.zeros(n, dtype=bool)
    state = False
    for i in range(n):
        if not state:
            if rms[i] >= th_onset:
                state = True
        else:
            if rms[i] < th_offset:
                state = False
        active[i] = state
        
    # 2. Merge silence gaps (Min Silence)
    active_int = active.astype(np.int8)
    diff = np.diff(np.concatenate(([0], active_int, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    if len(starts) > 1:
        for i in range(len(starts) - 1):
            gap = starts[i+1] - ends[i]
            if gap < min_silence:
                active[ends[i]:starts[i+1]] = True
                
    # Recalculate intervals after merging
    active_int = active.astype(np.int8)
    diff = np.diff(np.concatenate(([0], active_int, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    # 3. Filter short spikes (Min Active)
    filtered_intervals = []
    for s, e in zip(starts, ends):
        if (e - s) >= min_active:
            filtered_intervals.append((s, e))
            
    return filtered_intervals

def compute_restimulus(
    stim: np.ndarray,
    rep: np.ndarray,
    rms_normalized: np.ndarray,
    threshold_normalized: float,
    min_active: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula el restimulus dinámico adaptando de forma local y dinámica los tiempos 
    de inicio (onset) y fin (offset) de cada repetición de gesto individual,
    basándose en el nivel de ruido del reposo previo y corrigiendo el tiempo de reacción.
    Retorna la tupla (restimulus, re_repetition_id).
    """
    restim = np.zeros_like(stim)
    re_rep = np.zeros_like(rep)
    
    changes = np.where(np.diff(stim) != 0)[0] + 1
    boundaries = np.concatenate(([0], changes, [len(stim)]))
    
    for i in range(len(boundaries) - 1):
        seg_start = boundaries[i]
        seg_end = boundaries[i + 1]
        val = stim[seg_start]
        r_val = rep[seg_start]
        
        # Omitimos los periodos etiquetados como reposo visual
        if val == 0:
            continue
            
        # 1. Definir una ventana de reposo de referencia antes del inicio del gesto
        # La fase de preparación es de 2.0s y reposo es de 3.0s, por lo que tomamos una ventana limpia
        baseline_start = max(0, seg_start - 2500)
        baseline_end = max(1, seg_start - 300)
        baseline_rms = rms_normalized[baseline_start:baseline_end]
        
        # Si la ventana es muy corta (p. ej., primera muestra de la sesión), usamos el umbral global
        if len(baseline_rms) < 300:
            onset_threshold = threshold_normalized
        else:
            noise_mean = np.mean(baseline_rms)
            noise_std = np.std(baseline_rms)
            onset_threshold = noise_mean + 3.0 * noise_std
            
        # 2. Buscar el inicio real de la contracción (onset) alrededor del estímulo visual
        search_onset_start = max(0, seg_start - 300)
        search_onset_end = min(len(stim), seg_start + 1200)
        
        onset_idx = None
        for idx in range(search_onset_start, search_onset_end):
            if rms_normalized[idx] >= onset_threshold:
                onset_idx = idx
                break
                
        if onset_idx is None:
            # Fallback fisiológico de retraso de reacción (300 ms) si no se detecta cruce
            final_start = seg_start + 300
        else:
            final_start = onset_idx
            
        # 3. Buscar el final real de la contracción (offset) alrededor del estímulo visual
        # Buscamos de atrás hacia adelante a partir de seg_end + 1.0s hasta seg_end - 0.5s
        search_offset_start = max(0, seg_end - 500)
        search_offset_end = min(len(stim), seg_end + 1000)
        
        offset_idx = None
        for idx in range(search_offset_end - 1, search_offset_start - 1, -1):
            if rms_normalized[idx] >= onset_threshold:
                offset_idx = idx + 1
                break
                
        if offset_idx is None:
            # Fallback fisiológico de relajación (200 ms) si no se detecta cruce
            final_end = seg_end + 200
        else:
            final_end = offset_idx
            
        # 4. Restricciones físicas: no invadir estímulos visuales de gestos diferentes
        left_limit = 0
        for j in range(seg_start - 1, -1, -1):
            if stim[j] != 0 and stim[j] != val:
                left_limit = j + 1
                break
                
        right_limit = len(stim)
        for j in range(seg_end, len(stim)):
            if stim[j] != 0 and stim[j] != val:
                right_limit = j
                break
                
        final_start = max(final_start, left_limit)
        final_end = min(final_end, right_limit)
        
        # Rellenar el segmento activo para esta repetición
        restim[final_start:final_end] = val
        re_rep[final_start:final_end] = r_val
        
    return restim, re_rep


# ── Etapa B: valid_flag por control de calidad MAD ─────────────

def compute_valid_flags(
    df: pd.DataFrame,
    signal_col: str = "emg_norm",
    stim_col: str   = "restimulus",
    rep_col: str    = "re_repetition_id",
) -> np.ndarray:
    """Marca repeticiones atípicas usando MAD sobre duración y energía."""
    valid    = np.ones(len(df), dtype=np.int8)
    restim   = df[stim_col].values
    
    # Usar re_repetition_id si está en el dataframe, sino repetition_id
    r_col = rep_col if rep_col in df.columns else ("repetition_id" if "repetition_id" in df.columns else "repetition")
    rep   = df[r_col].values
    
    sig_col  = signal_col if signal_col in df.columns else "filtered"
    signal   = df[sig_col].values

    gesture_ids = [g for g in np.unique(restim) if g != 0]

    for gid in gesture_ids:
        reps_in_gesture = np.unique(rep[restim == gid])
        reps_in_gesture = reps_in_gesture[reps_in_gesture > 0]

        if len(reps_in_gesture) < 3:
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

        # Usamos un umbral robusto de Z-score estándar de 3.5 recomendado internacionalmente.
        # En una distribución normal, 1.4826 * MAD equivale a 1.0 Desviación Estándar.
        # Esto permite variabilidad natural biológica pero detecta outliers de verdad.
        thresh_dur = 3.5 * (1.4826 * mad_dur) if mad_dur > 0 else 0.0
        thresh_energy = 3.5 * (1.4826 * mad_energy) if mad_energy > 0 else 0.0

        for mask, dur, energy in zip(masks, durations, energies):
            bad_dur    = (thresh_dur > 0) and (abs(dur - med_dur) > thresh_dur)
            bad_energy = (thresh_energy > 0) and (abs(energy - med_energy) > thresh_energy)
            if bad_dur or bad_energy:
                valid[mask] = 0

    return valid


# ── Punto de entrada principal ─────────────────────────────────

def run_preprocess(csv_path: str, meta_path: str) -> str:
    """Pipeline completo: normalización, RMS causal, restimulus y valid_flag."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    calibration = meta.get("calibration", {})
    onset_cfg   = meta.get("onset_config", {})
    fs_hz       = int(meta.get("fs_hz", 1000))

    rms_window_ms  = int(onset_cfg.get("rms_window_ms", 100))
    min_active_ms  = int(onset_cfg.get("min_active_ms", 200))
    
    # ── Cargar CSV ────────────────────────────────────────────
    df = pd.read_csv(csv_path)

    if "stimulus" not in df.columns:
        raise ValueError(f"El CSV no contiene columna 'stimulus': {csv_path}")

    # ── Eliminar set_id si existe por redundancia ──────────────
    if "set_id" in df.columns:
        df = df.drop(columns=["set_id"])

    # ── Extraer parámetros del metadata ───────────────────────
    mvc_voltage_v = float(calibration.get("mvc_voltage_v", 1.0))
    onset_threshold_v = float(calibration.get("onset_threshold_v", 0.0))
    
    # Fallback si no hay calibración válida
    if mvc_voltage_v <= 1e-6:
        mvc_voltage_v = 1.0
        
    raw_emg = df['filtered'].values.astype(np.float64)
    stim = df["stimulus"].values
    rest_signal = raw_emg[stim == 0]
    
    # Detección de umbral sospechosamente bajo (menor a 1.5 * desvío estándar de la señal en reposo)
    is_suspiciously_low = False
    if len(rest_signal) > 0:
        rest_std = float(np.std(rest_signal))
        if onset_threshold_v < 1.5 * rest_std:
            is_suspiciously_low = True
            
    if onset_threshold_v <= 0 or is_suspiciously_low:
        # Auto-cálculo de umbral basado en periodos de reposo en la columna 'filtered'
        if len(rest_signal) > 0:
            noise_mean = float(np.mean(np.abs(rest_signal)))
            noise_std = float(np.std(np.abs(rest_signal)))
            onset_threshold_v = noise_mean + 3.0 * noise_std
        else:
            noise_mean = 0.0
            noise_std = 0.01
            onset_threshold_v = float(np.percentile(np.abs(raw_emg), 25))
            
        # Actualizar metadata si se recalculó por ser incorrecto o muy bajo
        old_val = float(calibration.get("onset_threshold_v", 0.0))
        if old_val != onset_threshold_v:
            calibration["onset_threshold_v"] = onset_threshold_v
            calibration["noise_mean"] = noise_mean
            calibration["noise_std"] = noise_std
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            print(f"[PREPROCESS] Metadata actualizado con umbral corregido: {onset_threshold_v:.4f} V (era {old_val:.4f} V)")
            
    print(f"[PREPROCESS] Parámetros cargados:")
    print(f"  MVC voltaje de referencia: {mvc_voltage_v:.4f} V")
    print(f"  Umbral absoluto de onset:  {onset_threshold_v:.4f} V")

    # Instanciamos nuestro EMGPreprocessor de alto rendimiento
    preprocessor = EMGPreprocessor(
        fs=fs_hz,
        mvc_voltage_v=mvc_voltage_v,
        onset_threshold_v=onset_threshold_v,
        rms_window_ms=rms_window_ms,
        min_active_ms=min_active_ms
    )
    
    # Si la columna emg_norm no existe o está desactualizada, la recalculamos
    raw_emg = df['filtered'].values.astype(np.float64)
    emg_norm = preprocessor.normalize_mvc(raw_emg)
    df["emg_norm"] = emg_norm
    
    # Calculamos el RMS sobre la señal normalizada
    rms_norm = preprocessor.compute_rms(emg_norm)
    
    # ── Etapa A: RMS, restimulus y re_repetition_id ────────────
    stim = df["stimulus"].values
    rep_col = "repetition_id" if "repetition_id" in df.columns else "repetition"
    rep = df[rep_col].values
    
    restim, re_rep = compute_restimulus(
        stim=stim,
        rep=rep,
        rms_normalized=rms_norm,
        threshold_normalized=preprocessor.threshold_normalized,
        min_active=preprocessor.min_active_samples
    )
    df["restimulus"] = restim
    df["re_repetition_id"] = re_rep

    # ── Etapa B: valid_flag ───────────────────────────────────
    df["valid_flag"] = compute_valid_flags(
        df,
        signal_col="emg_norm",
        stim_col="restimulus",
        rep_col="re_repetition_id"
    )

    # ── Guardar CSV actualizado ───────────────────────────────
    df.to_csv(csv_path, index=False)

    # ── Estadísticas de resumen ───────────────────────────────
    total_active_stim   = int(np.sum(stim != 0))
    total_active_restim = int(np.sum(restim != 0))
    trimmed             = total_active_stim - total_active_restim
    n_invalid           = int(np.sum(df["valid_flag"] == 0))
    n_valid             = int(np.sum(df["valid_flag"] == 1))

    print(f"[PREPROCESS] restimulus: {total_active_restim} muestras activas "
          f"({trimmed} recortadas por onset dinámico)")
    print(f"[PREPROCESS] valid_flag: {n_valid} válidas | {n_invalid} inválidas")

    return csv_path


# ── CLI standalone ─────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python preprocess.py <archivo.csv> <archivo_metadata.json>")
        sys.exit(1)

    run_preprocess(sys.argv[1], sys.argv[2])
    print(f"[OK] {sys.argv[1]} procesado con restimulus y valid_flag (EMGPreprocessor)")
