import numpy as np

NOISE_STD = 0.005

def mean_absolute_value(x):
    return np.mean(np.abs(x))

def root_mean_square(x):
    return np.sqrt(np.mean(np.square(x)))

def waveform_length(x):
    return np.sum(np.abs(np.diff(x)))

def zero_crossings(x, threshold=NOISE_STD):
    x = np.asarray(x).flatten()
    sign_changes = (x[:-1] * x[1:]) < 0
    above_threshold = np.abs(x[:-1] - x[1:]) > threshold
    return np.sum(sign_changes & above_threshold)

def slope_sign_changes(x, threshold=NOISE_STD):
    x = np.asarray(x).flatten()
    dx = np.diff(x)
    sign_changes = (dx[:-1] * dx[1:]) < 0
    above_threshold = (np.abs(x[:-2] - x[1:-1]) > threshold) | (np.abs(x[2:] - x[1:-1]) > threshold)
    return np.sum(sign_changes & above_threshold)

def variance(x):
    return np.var(x)

def get_dynamic_threshold(subject, muscle):
    import json
    import glob
    import os
    
    try:
        from pathlib import Path
        default_path = Path(__file__).parent.parent / "datasets" / "raw" / "myotensor_proto" / muscle
        base_dir = str(Path(os.environ.get("RAW_DATA_PATH", default_path)))
        meta_file = glob.glob(f"{base_dir}/{subject}/*_metadata.json")[0]
        with open(meta_file, "r") as f:
            meta = json.load(f)
        return meta.get("calibration", {}).get("normalized_noise_std", 0.005)
    except Exception as e:
        print(f"⚠️ Aviso: Usando ruido por defecto. Error cargando JSON para {subject}-{muscle}: {e}")
        return 0.005

def extract_features(X_tensors, subject="S01", muscle="FDS"):
    """
    Toma un arreglo de tensores 2D (N, W) o 3D (N, W, C) y extrae las 6 características temporales
    por cada canal electromiográfico.
    
    Args:
        X_tensors: numpy array de forma (N, W) o (N, W, C)
        subject: ID del sujeto
        muscle: Músculo o etiqueta anatómica
        
    Returns:
        numpy array de forma (N, 6 * C)
    """
    X_arr = np.asarray(X_tensors)
    if X_arr.ndim == 2:
        # (N, W) -> (N, W, 1)
        X_arr = np.expand_dims(X_arr, axis=-1)
    elif X_arr.ndim == 1:
        # Una única ventana (W,) -> (1, W, 1)
        X_arr = np.expand_dims(X_arr, axis=(0, -1))
        
    N, W, C = X_arr.shape
    noise_threshold = get_dynamic_threshold(subject, muscle)
        
    all_features = []
    for window_idx in range(N):
        window_feats = []
        for c in range(C):
            ch_signal = X_arr[window_idx, :, c]
            mav = mean_absolute_value(ch_signal)
            rms = root_mean_square(ch_signal)
            wl  = waveform_length(ch_signal)
            zc  = zero_crossings(ch_signal, threshold=noise_threshold)
            ssc = slope_sign_changes(ch_signal, threshold=noise_threshold)
            var = variance(ch_signal)
            window_feats.extend([mav, rms, wl, zc, ssc, var])
        all_features.append(window_feats)
        
    return np.array(all_features, dtype=np.float32)

def scale_features(X_train, X_test=None):
    """
    Escala las características usando StandardScaler y devuelve el scaler para exportación.
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    return X_train_scaled, scaler


def extract_features_pooled(X_windows_by_subject, subjects, muscle):
    """
    Extrae características para múltiples sujetos aplicando el umbral de ruido calibrado de cada uno.
    
    Args:
        X_windows_by_subject (dict o list of np.ndarray): Diccionario {subject: X_windows} o lista ordenada.
        subjects (list of str): Lista de IDs de sujeto.
        muscle (str): 'FDS' o 'ED'.
        
    Returns:
        np.ndarray: Arreglo de características (N_total, 6).
    """
    feat_list = []
    if isinstance(X_windows_by_subject, dict):
        for s in subjects:
            if s in X_windows_by_subject and len(X_windows_by_subject[s]) > 0:
                feat_list.append(extract_features(X_windows_by_subject[s], subject=s, muscle=muscle))
    else:
        for X_s, s in zip(X_windows_by_subject, subjects):
            if len(X_s) > 0:
                feat_list.append(extract_features(X_s, subject=s, muscle=muscle))
                
    if feat_list:
        return np.concatenate(feat_list, axis=0)
    return np.empty((0, 6), dtype=np.float32)

