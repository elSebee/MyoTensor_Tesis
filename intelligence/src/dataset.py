"""
=============================================================
 dataset.py — Carga y Segmentación Temporal de sEMG
=============================================================

API v2 (Fase 0 del plan de re-ejecución):
  - load_session()            -> DataFrame limpio de una sesión + metadatos, con `set_id`.
  - load_windows()            -> WindowSet con todas las ventanas de un sujeto (SIN partir).
  - load_continuous_windows() -> WindowSet en flujo continuo (incluye transiciones).
  - concat_windows()          -> concatenación poblacional de WindowSets.

La partición train/test NO se hace aquí: se hace en `splits.py`, siempre por set
completo (`set_id`), nunca por ventana.

Las funciones `load_and_segment_dataset` y `load_pooled_dataset` se conservan solo
por compatibilidad con los notebooks antiguos (Exp 01–07 v1) y están obsoletas.
"""

import os
import json
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from . import config

DEFAULT_SUBJECTS = [f"S{i:02d}" for i in range(1, 11)]
MUSCLES = ("ED", "FDS")
TIME_JUMP_US = 2000  # salto temporal > 2 ms rompe la continuidad de un bloque


def get_data_root(muscle: str) -> Path:
    """Directorio de datos crudos de un músculo (sobrescribible con RAW_DATA_PATH)."""
    default_path = Path(__file__).resolve().parent.parent / "datasets" / "raw" / "myotensor_proto" / muscle
    return Path(os.environ.get("RAW_DATA_PATH", default_path))


@lru_cache(maxsize=64)
def _read_session(subject: str, muscle: str) -> Tuple[pd.DataFrame, str]:
    subject_dir = get_data_root(muscle) / subject
    csv_files = sorted(subject_dir.glob("session_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron CSV's para {subject} en {subject_dir}")
    if len(csv_files) > 1:
        raise RuntimeError(f"{subject_dir} contiene {len(csv_files)} sesiones; se espera exactamente una.")
    meta_files = sorted(subject_dir.glob("session_*_metadata.json"))
    meta_json = meta_files[0].read_text(encoding="utf-8") if meta_files else "{}"
    return pd.read_csv(csv_files[0]), meta_json


def load_session(subject: str, muscle: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Carga la sesión de un sujeto/músculo y agrega la columna `set_id` (1..n_sets).

    Asignación de set:
      - muestras activas (restimulus > 0): `re_repetition_id` (set real de la contracción);
      - reposo (restimulus == 0): `repetition_id`, que corresponde al set del gesto
        que sigue (el reposo de preparación precede a cada gesto).
    El reposo final (repetition_id = n_sets + 1, si existiera) se asigna al último set.

    El índice del DataFrame es el número de fila del CSV original, lo que permite
    reconstruir cualquier ventana desde el archivo crudo (usado por el banco HIL).
    """
    df_raw, meta_json = _read_session(subject, muscle)
    meta = json.loads(meta_json)
    n_sets = int(meta.get("n_sets", 10))

    df = df_raw[df_raw["valid_flag"] == 1].copy() if "valid_flag" in df_raw.columns else df_raw.copy()
    active = df["restimulus"].to_numpy() > 0
    set_id = np.where(active, df["re_repetition_id"].to_numpy(), df["repetition_id"].to_numpy())
    df["set_id"] = np.clip(set_id, 1, n_sets).astype(np.int16)
    return df, meta


@dataclass
class WindowSet:
    """
    Conjunto de ventanas con su trazabilidad completa.

    X        : (N, W) monocanal o (N, W, C) multicanal.
    y        : etiqueta (0 = Reposo, 1 = Palma, 2 = Puño, 3 = Paz).
    set_id   : set (1..10) al que pertenece la ventana -> unidad de partición.
    block_id : bloque continuo (misma clase y set, sin saltos temporales) dentro del sujeto.
    start    : fila del CSV original donde comienza la ventana.
    subject  : ID de sujeto por ventana.
    pure     : True si todas las muestras de la ventana tienen la misma etiqueta
               (siempre True en modo steady-state/segmentado; puede ser False en flujo continuo).
    """
    X: np.ndarray
    y: np.ndarray
    set_id: np.ndarray
    block_id: np.ndarray
    start: np.ndarray
    subject: np.ndarray
    pure: np.ndarray
    muscle: str
    config: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask_or_idx) -> "WindowSet":
        sel = np.asarray(mask_or_idx)
        return WindowSet(
            X=self.X[sel], y=self.y[sel], set_id=self.set_id[sel], block_id=self.block_id[sel],
            start=self.start[sel], subject=self.subject[sel], pure=self.pure[sel],
            muscle=self.muscle, config=dict(self.config),
        )

    def groups(self, by: str = "set") -> np.ndarray:
        """
        Etiquetas de grupo para validación cruzada agrupada.
          by='set'         -> set_id (intra-sujeto)
          by='subject'     -> sujeto (inter-sujeto)
          by='subject_set' -> combinación sujeto+set (pooled agrupado por repetición)
        """
        if by == "set":
            return self.set_id.astype(np.int64)
        if by == "subject":
            return self.subject
        if by == "subject_set":
            return np.char.add(np.char.add(self.subject.astype(str), "_"), self.set_id.astype(str))
        raise ValueError(f"by='{by}' no reconocido (use 'set', 'subject' o 'subject_set').")


def _signal_columns(df: pd.DataFrame, apply_mvc: bool, channel_cols: Optional[Sequence[str]]) -> List[str]:
    if channel_cols is not None:
        cols = [c for c in channel_cols if c in df.columns]
        if not cols:
            raise ValueError(f"Ninguna columna de channel_cols={channel_cols} se encontró en el DataFrame")
        return cols
    if apply_mvc and "emg_norm" in df.columns:
        return ["emg_norm"]
    return ["filtered"]


def _trim_steady_state(df: pd.DataFrame, trim_onset_ms: int, trim_offset_ms: int, window_size: int) -> pd.DataFrame:
    """Recorta onset/offset de cada contracción activa; el reposo se conserva completo."""
    keep = []
    for (gid, _), gdf in df.groupby(["restimulus", "set_id"], sort=False):
        if gid == 0:
            keep.append(gdf.index.to_numpy())
            continue
        L = len(gdf)
        if L > (trim_onset_ms + trim_offset_ms + window_size):
            keep.append(gdf.index.to_numpy()[trim_onset_ms:L - trim_offset_ms])
        else:
            keep.append(gdf.index.to_numpy())
    return df.loc[np.sort(np.concatenate(keep))]


def _time_jumps(df: pd.DataFrame) -> np.ndarray:
    jumps = np.zeros(len(df), dtype=bool)
    if "timestamp_us" in df.columns:
        jumps[1:] = np.diff(df["timestamp_us"].to_numpy()) > TIME_JUMP_US
    # Huecos en el índice (filas descartadas por recorte o valid_flag) también rompen continuidad
    idx = df.index.to_numpy()
    jumps[1:] |= np.diff(idx) != 1
    return jumps


def _slide(sig: np.ndarray, W: int, S: int) -> Tuple[np.ndarray, np.ndarray]:
    """Ventanas deslizantes de un bloque (L, C). Devuelve (ventanas, offsets de inicio)."""
    L = sig.shape[0]
    if L < W:
        return np.empty((0, W, sig.shape[1]), dtype=np.float32), np.empty(0, dtype=np.int64)
    starts = np.arange(0, L - W + 1, S)
    view = np.lib.stride_tricks.sliding_window_view(sig, W, axis=0)  # (L-W+1, C, W)
    return np.transpose(view[starts], (0, 2, 1)).astype(np.float32), starts


def load_windows(
    subject: str,
    muscle: str,
    window_size: Optional[int] = None,
    stride: Optional[int] = None,
    apply_mvc: bool = True,
    steady_state_only: bool = True,
    trim_onset_ms: int = 500,
    trim_offset_ms: int = 500,
    channel_cols: Optional[Sequence[str]] = None,
) -> WindowSet:
    """
    Segmenta TODAS las ventanas de un sujeto/músculo sin partir en train/test.

    Cada ventana queda contenida en un único bloque continuo (misma clase y mismo set),
    por lo que nunca cruza una frontera de set: partir por `set_id` es libre de fuga.
    """
    W = window_size or config.WINDOW_SIZE
    S = stride or config.STRIDE
    df, _ = load_session(subject, muscle)
    if steady_state_only:
        df = _trim_steady_state(df, trim_onset_ms, trim_offset_ms, W)

    label_change = np.zeros(len(df), dtype=bool)
    label_change[1:] = (np.diff(df["restimulus"].to_numpy()) != 0) | (np.diff(df["set_id"].to_numpy()) != 0)
    block = np.cumsum(label_change | _time_jumps(df))

    cols = _signal_columns(df, apply_mvc, channel_cols)
    sig_all = df[cols].to_numpy(dtype=np.float32)
    rows = df.index.to_numpy()
    y_all = df["restimulus"].to_numpy()
    set_all = df["set_id"].to_numpy()

    X_l, y_l, set_l, blk_l, start_l = [], [], [], [], []
    bounds = np.flatnonzero(np.diff(block)) + 1
    for seg in np.split(np.arange(len(df)), bounds):
        win, offs = _slide(sig_all[seg], W, S)
        if len(offs) == 0:
            continue
        first = seg[offs]
        X_l.append(win)
        y_l.append(y_all[first])
        set_l.append(set_all[first])
        blk_l.append(np.full(len(offs), block[seg[0]]))
        start_l.append(rows[first])

    X = np.concatenate(X_l) if X_l else np.empty((0, W, len(cols)), dtype=np.float32)
    if len(cols) == 1:
        X = X[..., 0]
    n = len(X)
    return WindowSet(
        X=X,
        y=np.concatenate(y_l).astype(np.int32) if y_l else np.empty(0, np.int32),
        set_id=np.concatenate(set_l).astype(np.int16) if set_l else np.empty(0, np.int16),
        block_id=np.concatenate(blk_l).astype(np.int32) if blk_l else np.empty(0, np.int32),
        start=np.concatenate(start_l).astype(np.int64) if start_l else np.empty(0, np.int64),
        subject=np.full(n, subject, dtype="<U8"),
        pure=np.ones(n, dtype=bool),
        muscle=muscle,
        config=dict(mode="segmented", window_size=W, stride=S, apply_mvc=apply_mvc,
                    steady_state_only=steady_state_only, trim_onset_ms=trim_onset_ms,
                    trim_offset_ms=trim_offset_ms, signal_cols=cols),
    )


def load_continuous_windows(
    subject: str,
    muscle: str,
    window_size: Optional[int] = None,
    stride: Optional[int] = None,
    apply_mvc: bool = True,
    label_mode: str = "last",
    channel_cols: Optional[Sequence[str]] = None,
) -> WindowSet:
    """
    Ventanas deslizantes sobre la sesión completa, como las ve el firmware en tiempo real
    (incluye inicio/fin de contracción y transiciones entre clases).

    label_mode:
      'last'   -> etiqueta de la última muestra (decisión causal: lo que el usuario hace "ahora").
      'center' -> etiqueta de la muestra central.
    `set_id` se toma de la misma muestra que la etiqueta. `pure` indica si la ventana tiene
    una sola etiqueta; las impuras son las transiciones.
    """
    W = window_size or config.WINDOW_SIZE
    S = stride or config.STRIDE
    if label_mode not in ("last", "center"):
        raise ValueError("label_mode debe ser 'last' o 'center'.")
    pos = W - 1 if label_mode == "last" else W // 2

    df, _ = load_session(subject, muscle)
    block = np.cumsum(_time_jumps(df))
    cols = _signal_columns(df, apply_mvc, channel_cols)
    sig_all = df[cols].to_numpy(dtype=np.float32)
    rows = df.index.to_numpy()
    y_all = df["restimulus"].to_numpy()
    set_all = df["set_id"].to_numpy()

    X_l, y_l, set_l, blk_l, start_l, pure_l = [], [], [], [], [], []
    bounds = np.flatnonzero(np.diff(block)) + 1
    for seg in np.split(np.arange(len(df)), bounds):
        win, offs = _slide(sig_all[seg], W, S)
        if len(offs) == 0:
            continue
        y_seg = y_all[seg]
        lab_view = np.lib.stride_tricks.sliding_window_view(y_seg, W)[offs]
        X_l.append(win)
        y_l.append(y_seg[offs + pos])
        set_l.append(set_all[seg][offs + pos])
        blk_l.append(np.full(len(offs), block[seg[0]]))
        start_l.append(rows[seg[offs]])
        pure_l.append((lab_view == lab_view[:, :1]).all(axis=1))

    X = np.concatenate(X_l)
    if len(cols) == 1:
        X = X[..., 0]
    n = len(X)
    return WindowSet(
        X=X,
        y=np.concatenate(y_l).astype(np.int32),
        set_id=np.concatenate(set_l).astype(np.int16),
        block_id=np.concatenate(blk_l).astype(np.int32),
        start=np.concatenate(start_l).astype(np.int64),
        subject=np.full(n, subject, dtype="<U8"),
        pure=np.concatenate(pure_l),
        muscle=muscle,
        config=dict(mode="continuous", window_size=W, stride=S, apply_mvc=apply_mvc,
                    label_mode=label_mode, signal_cols=cols),
    )


def concat_windows(window_sets: Sequence[WindowSet]) -> WindowSet:
    """Concatena WindowSets (p. ej. varios sujetos del mismo músculo y configuración)."""
    if not window_sets:
        raise ValueError("Lista vacía de WindowSets.")
    muscles = {w.muscle for w in window_sets}
    if len(muscles) > 1:
        raise ValueError(f"No se pueden mezclar músculos: {muscles}")
    cfg0 = {k: v for k, v in window_sets[0].config.items()}
    for w in window_sets[1:]:
        if w.config != cfg0:
            raise ValueError("Los WindowSets tienen configuraciones de segmentación distintas.")
    cat = lambda attr: np.concatenate([getattr(w, attr) for w in window_sets])
    return WindowSet(
        X=cat("X"), y=cat("y"), set_id=cat("set_id"), block_id=cat("block_id"),
        start=cat("start"), subject=cat("subject"), pure=cat("pure"),
        muscle=window_sets[0].muscle, config=cfg0,
    )


def load_pooled_windows(muscle: str, subjects: Optional[Sequence[str]] = None, **kwargs) -> WindowSet:
    """Atajo: load_windows() para varios sujetos + concat_windows()."""
    subjects = list(subjects or DEFAULT_SUBJECTS)
    return concat_windows([load_windows(s, muscle, **kwargs) for s in subjects])


# =============================================================
#  API LEGADA (v1) — obsoleta, conservada por compatibilidad
# =============================================================

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

    .. deprecated:: Fase 0
        Partición aleatoria por grupos (no cronológica) y sin trazabilidad de set.
        Use `load_windows()` + `splits.py`.
    """
    warnings.warn(
        "load_and_segment_dataset() está obsoleta: use dataset.load_windows() + splits.py",
        DeprecationWarning, stacklevel=2,
    )
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
