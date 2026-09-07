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
    Toma un arreglo de tensores 3D o 2D y extrae las 6 características temporales.
    
    Args:
        X_tensors: numpy array de forma (N_ventanas, W) o (N_ventanas, W, 1)
        subject: ID del sujeto
        muscle: Músculo
        
    Returns:
        numpy array de forma (N_ventanas, 6)
    """
    if X_tensors.ndim == 3:
        X_tensors = np.squeeze(X_tensors, axis=-1)
        
    noise_threshold = get_dynamic_threshold(subject, muscle)
        
    features = []
    for window in X_tensors:
        mav = mean_absolute_value(window)
        rms = root_mean_square(window)
        wl  = waveform_length(window)
        zc  = zero_crossings(window, threshold=noise_threshold)
        ssc = slope_sign_changes(window, threshold=noise_threshold)
        var = variance(window)
        features.append([mav, rms, wl, zc, ssc, var])
        
    return np.array(features)

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
