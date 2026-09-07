import pandas as pd
import numpy as np
from pathlib import Path
import os

def load_and_segment_dataset(subject, muscle, window_size=300, overlap=150):
    """
    Carga y segmenta los datos de un sujeto y músculo específico.
    
    Args:
        subject (str): ID del sujeto, ej. 'S01'.
        muscle (str): 'FDS' o 'ED'.
        window_size (int): Tamaño de la ventana en milisegundos/muestras.
        overlap (int): Traslape entre ventanas consecutivas. (S)
        
    Returns:
        X_train, y_train, X_test, y_test: Arreglos numpy segmentados.
    """
    # 1. Definir la ruta base
    try:
        default_path = Path(__file__).parent.parent / "datasets" / "raw" / "myotensor_proto" / muscle
        base_dir = Path(os.environ.get("RAW_DATA_PATH", default_path))
    except:
        base_dir = Path(__file__).parent.parent / "datasets" / "raw" / "myotensor_proto" / muscle
        
    # Buscar archivos del sujeto
    subject_dir = base_dir / subject
    csv_files = sorted(list(subject_dir.glob("session_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron CSV's para {subject} en {subject_dir} el path es {default_path}")
        
    df_raw = pd.read_csv(csv_files[0])
    
    # Filtro de calidad
    df_clean = df_raw[df_raw["valid_flag"] == 1].copy()
    
    # 2. Dividir entrenamiento y prueba (Split First) estratificado
    # Para mantener el módulo simple e independiente, usaremos split por bloque/repetición básico
    # Asumimos que los primeros 80% de bloques son train y 20% test
    
    # Para hacer esto rápido y robusto, usaremos GroupShuffleSplit o división temporal
    from sklearn.model_selection import GroupShuffleSplit
    
    df_active = df_clean[df_clean["restimulus"] != 0].copy()
    df_reposo = df_clean[df_clean["restimulus"] == 0].copy()
    
    gss_active = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx_act, test_idx_act = next(gss_active.split(df_active, groups=df_active["re_repetition_id"]))
    df_train_act = df_active.iloc[train_idx_act].copy()
    df_test_act = df_active.iloc[test_idx_act].copy()
    
    reposo_blocks = (df_clean["restimulus"] != df_clean["restimulus"].shift()).cumsum()
    df_reposo["block_id"] = reposo_blocks[df_clean["restimulus"] == 0]
    
    gss_reposo = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx_rep, test_idx_rep = next(gss_reposo.split(df_reposo, groups=df_reposo["block_id"]))
    df_train_rep = df_reposo.iloc[train_idx_rep].copy()
    df_test_rep = df_reposo.iloc[test_idx_rep].copy()
    
    df_train = pd.concat([df_train_act, df_train_rep]).sort_index()
    df_test = pd.concat([df_test_act, df_test_rep]).sort_index()
    
    df_train["group_id"] = np.where(df_train["restimulus"] == 0, df_train["block_id"], df_train["re_repetition_id"])
    df_test["group_id"]  = np.where(df_test["restimulus"] == 0, df_test["block_id"], df_test["re_repetition_id"])
    
    def extract_windows(df, W, S):
        X_list, y_list = [], []
        groups = df.groupby(['group_id', 'restimulus'])
        for (g_id, gesture_class), group_df in groups:
            sig_col = "emg_norm" if "emg_norm" in group_df.columns else "filtered"
            signal_block = group_df[sig_col].values
            L = len(signal_block)
            if L < W: continue
            for start in range(0, L - W + 1, S):
                window = signal_block[start:start + W]
                X_list.append(window)
                y_list.append(gesture_class)
        return np.array(X_list), np.array(y_list)
    
    X_train, y_train = extract_windows(df_train, window_size, overlap)
    X_test, y_test   = extract_windows(df_test, window_size, overlap)
    
    return X_train, y_train, X_test, y_test
