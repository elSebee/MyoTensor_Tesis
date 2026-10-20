"""
=============================================================
 dataset.py — Carga, Partición y Segmentación Temporal de sEMG
 Soporta los 4 escenarios de preprocesamiento del Experimento 02
=============================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from . import config

def load_and_segment_dataset(
    subject,
    muscle,
    window_size=None,
    overlap=None,
    apply_mvc=True,
    steady_state_only=True,
    trim_onset_ms=500,
    trim_offset_ms=500,
    channel_cols=None
):
    """
    Carga y segmenta los datos de un sujeto y músculo específico.
    Soporta adquisición monocanal o multicanal arbitraria (C >= 1).
    
    Args:
        subject (str): ID del sujeto, ej. 'S01'.
        muscle (str): 'FDS' o 'ED'.
        window_size (int, optional): Tamaño de ventana en muestras/ms. Por defecto config.WINDOW_SIZE (250).
        overlap (int, optional): Paso o stride entre ventanas consecutivas. Por defecto config.STRIDE (50).
        apply_mvc (bool): Si True, utiliza la señal normalizada por MVC ('emg_norm'). Si False, usa 'filtered' (mV).
        steady_state_only (bool): Si True, aplica recorte simétrico de transientes (500 ms onset / 500 ms offset)
                                  reteniendo la meseta isométrica continua y filtrando con 'valid_flag' == 1.
                                  Si False, conserva el registro completo continuo.
        trim_onset_ms (int): Muestras/ms a recortar al inicio de cada contracción (fase de ascenso). Por defecto 500.
        trim_offset_ms (int): Muestras/ms a recortar al final de cada contracción (fase de relajación). Por defecto 500.
        channel_cols (list of str, optional): Lista de columnas de canal (ej. ['ch0', 'ch1']). Si es None, usa sig_col.
        
    Returns:
        X_train, y_train, X_test, y_test: Arreglos numpy segmentados (N, W) o (N, W, C).
    """
    if window_size is None:
        window_size = config.WINDOW_SIZE
    if overlap is None:
        overlap = config.STRIDE

    # 1. Definir la ruta base
    default_path = Path(__file__).resolve().parent.parent / "datasets" / "raw" / "myotensor_proto" / muscle
    base_dir = Path(os.environ.get("RAW_DATA_PATH", default_path))
        
    # Buscar archivos del sujeto
    subject_dir = base_dir / subject
    csv_files = sorted(list(subject_dir.glob("session_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron CSV's para {subject} en {subject_dir} (base: {base_dir})")
        
    df_raw = pd.read_csv(csv_files[0])
    
    # 1. Filtro de calidad (fallos técnicos reales de hardware)
    if "valid_flag" in df_raw.columns:
        df_clean = df_raw[df_raw["valid_flag"] == 1].copy()
    else:
        df_clean = df_raw.copy()
    
    rep_col = "re_repetition_id" if "re_repetition_id" in df_clean.columns else "repetition_id"
    
    # 2. Técnica Steady-State canónica: Recorte simétrico de bordes en contracciones activas
    if steady_state_only:
        trimmed_indices = []
        for (gid, r), gdf in df_clean.groupby(["restimulus", rep_col]):
            if gid == 0:
                trimmed_indices.extend(gdf.index)
            else:
                L = len(gdf)
                if L > (trim_onset_ms + trim_offset_ms + window_size):
                    plateau = gdf.iloc[trim_onset_ms : L - trim_offset_ms]
                    trimmed_indices.extend(plateau.index)
                else:
                    trimmed_indices.extend(gdf.index)
        df_clean = df_clean.loc[sorted(trimmed_indices)].copy()
    
    # 3. Aislamiento estricto de bloques temporales continuos (dt <= 2.0 ms)
    class_change = df_clean["restimulus"] != df_clean["restimulus"].shift()
    rep_change   = (df_clean["restimulus"] != 0) & (df_clean[rep_col] != df_clean[rep_col].shift())
    
    if "timestamp_us" in df_clean.columns:
        time_jump = df_clean["timestamp_us"].diff() > 2000
    else:
        time_jump = pd.Series(False, index=df_clean.index)
        
    df_clean["continuous_block_id"] = (class_change | rep_change | time_jump).cumsum()
    
    # 3. Dividir entrenamiento y prueba estratificado por repeticiones (Split First continuo)
    df_active = df_clean[df_clean["restimulus"] != 0].copy()
    df_reposo = df_clean[df_clean["restimulus"] == 0].copy()
    
    gss_active = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx_act, test_idx_act = next(gss_active.split(df_active, groups=df_active[rep_col]))
    df_train_act = df_active.iloc[train_idx_act].copy()
    df_test_act  = df_active.iloc[test_idx_act].copy()
    
    gss_reposo = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx_rep, test_idx_rep = next(gss_reposo.split(df_reposo, groups=df_reposo["continuous_block_id"]))
    df_train_rep = df_reposo.iloc[train_idx_rep].copy()
    df_test_rep  = df_reposo.iloc[test_idx_rep].copy()
    
    df_train = pd.concat([df_train_act, df_train_rep]).sort_index()
    df_test  = pd.concat([df_test_act, df_test_rep]).sort_index()
    
    # 4. Selección de columna según normalización MVC o canales especificados
    if channel_cols is not None:
        sig_cols = [c for c in channel_cols if c in df_clean.columns]
        if not sig_cols:
            raise ValueError(f"Ninguna columna de channel_cols={channel_cols} se encontró en el DataFrame")
    else:
        sig_col = "emg_norm" if (apply_mvc and "emg_norm" in df_clean.columns) else "filtered"
        if sig_col not in df_clean.columns:
            sig_col = "filtered"
        sig_cols = [sig_col]

    def extract_windows(df, W, S):
        X_list, y_list = [], []
        groups = df.groupby(['continuous_block_id', 'restimulus'])
        for (b_id, gesture_class), group_df in groups:
            signal_block = group_df[sig_cols].values  # Shape: (L, len(sig_cols))
            L = len(signal_block)
            if L < W: 
                continue
            for start in range(0, L - W + 1, S):
                window = signal_block[start:start + W]
                if len(sig_cols) == 1:
                    window = window.squeeze(-1)  # Retener compatibilidad (W,) si es monocanal
                X_list.append(window)
                y_list.append(gesture_class)
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)
    
    X_train, y_train = extract_windows(df_train, window_size, overlap)
    X_test, y_test   = extract_windows(df_test, window_size, overlap)
    
    return X_train, y_train, X_test, y_test


def load_pooled_dataset(
    muscle,
    subjects=None,
    window_size=None,
    overlap=None,
    apply_mvc=True,
    steady_state_only=True,
    trim_onset_ms=500,
    trim_offset_ms=500,
    channel_cols=None,
    return_subject_ids=False
):
    """
    Carga y segmenta de forma agregada (pooled) los datos de múltiples sujetos para un músculo específico.
    Conserva la partición estratificada por repetición (80/20) de cada sujeto de manera independiente
    antes de la concatenación, garantizando que el conjunto de prueba contenga repeticiones no vistas
    de todos los sujetos evaluados (sin filtración intra-sujeto).
    
    Args:
        muscle (str): 'FDS' o 'ED'.
        subjects (list of str, optional): Lista de IDs de sujetos. Por defecto S01 a S10.
        window_size (int, optional): Tamaño de ventana en muestras/ms. Por defecto config.WINDOW_SIZE.
        overlap (int, optional): Paso o stride entre ventanas. Por defecto config.STRIDE.
        apply_mvc (bool): Si True, usa normalización por %MVC ('emg_norm').
        steady_state_only (bool): Si True, recorta transientes onset/offset y filtra por valid_flag == 1.
        trim_onset_ms (int): Muestras/ms a recortar al inicio de contracción activa.
        trim_offset_ms (int): Muestras/ms a recortar al final de contracción activa.
        return_subject_ids (bool): Si True, retorna tuplas con los IDs de sujeto correspondientes a cada ventana.
        
    Returns:
        X_train, y_train, X_test, y_test: Arreglos numpy concatenados poblacionales.
        (Opcional si return_subject_ids=True: subj_train, subj_test como arrays 1D de strings)
    """
    if subjects is None:
        subjects = [f"S{i:02d}" for i in range(1, 11)]
        
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list   = [], []
    subj_train_list, subj_test_list = [], []
    
    for s in subjects:
        X_tr, y_tr, X_te, y_te = load_and_segment_dataset(
            subject=s,
            muscle=muscle,
            window_size=window_size,
            overlap=overlap,
            apply_mvc=apply_mvc,
            steady_state_only=steady_state_only,
            trim_onset_ms=trim_onset_ms,
            trim_offset_ms=trim_offset_ms,
            channel_cols=channel_cols
        )
        X_train_list.append(X_tr)
        y_train_list.append(y_tr)
        X_test_list.append(X_te)
        y_test_list.append(y_te)
        
        if return_subject_ids:
            subj_train_list.append(np.array([s] * len(y_tr)))
            subj_test_list.append(np.array([s] * len(y_te)))
            
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test  = np.concatenate(X_test_list, axis=0)
    y_test  = np.concatenate(y_test_list, axis=0)
    
    if return_subject_ids:
        subj_train = np.concatenate(subj_train_list, axis=0)
        subj_test  = np.concatenate(subj_test_list, axis=0)
        return X_train, y_train, X_test, y_test, subj_train, subj_test
        
    return X_train, y_train, X_test, y_test
