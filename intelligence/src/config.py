"""
=============================================================
 config.py — Configuración Central del Pipeline de IA
 Single Source of Truth (SSOT) para Preprocesamiento e Hiperparámetros
=============================================================
"""

# Restricciones Temporales Canónicas (Control en Tiempo Real)
WINDOW_SIZE = 250         # W: 250 ms (250 muestras @ 1 kHz)
STRIDE = 50               # S: 50 ms (50 muestras @ 1 kHz, 20 decisiones/seg)
W = WINDOW_SIZE
S = STRIDE

SAMPLING_RATE_HZ = 1000   # 1 kHz
LATENCY_UPDATE_MS = 50    # Latencia máxima de refresco en tiempo real
NUM_CLASSES = 4
GESTURES = {0: "Reposo", 1: "Palma", 2: "Puño", 3: "Paz"}

# Semilla Global de Reproducibilidad
RANDOM_STATE = 42
SEED = 42

# Diccionario Centralizado de Hiperparámetros Óptimos por Modelo y Músculo
BEST_HYPERPARAMS = {
    "FDS": {
        "RF": {
            "n_estimators": 150,
            "max_depth": 15,
            "min_samples_split": 4,
            "class_weight": "balanced",
            "random_state": 42
        },
        "SVM": {
            "kernel": "rbf",
            "C": 50.0,
            "gamma": 0.1,
            "class_weight": "balanced",
            "random_state": 42
        },
        "CNN-LSTM": {
            "conv_filters": (32, 64),
            "kernel_size": 3,
            "lstm_units": 64,
            "dropout_rate": 0.3,
            "learning_rate": 0.001
        },
        "CNN-TCN": {
            "nb_filters": 32,
            "kernel_size": 3,
            "dilations": (1, 2, 4),
            "dropout_rate": 0.2,
            "learning_rate": 0.001
        }
    },
    "ED": {
        "RF": {
            "n_estimators": 50,
            "max_depth": None,
            "min_samples_split": 4,
            "class_weight": "balanced",
            "random_state": 42
        },
        "SVM": {
            "kernel": "rbf",
            "C": 50.0,
            "gamma": "scale",
            "class_weight": "balanced",
            "random_state": 42
        },
        "CNN-LSTM": {
            "conv_filters": (32, 64),
            "kernel_size": 3,
            "lstm_units": 64,
            "dropout_rate": 0.3,
            "learning_rate": 0.001
        },
        "CNN-TCN": {
            "nb_filters": 32,
            "kernel_size": 3,
            "dilations": (1, 2, 4),
            "dropout_rate": 0.2,
            "learning_rate": 0.001
        }
    }
}

# Compatibilidad con código anterior
BEST_PARAMS = {
    "FDS": {
        "SVM": {"WINDOW": WINDOW_SIZE, "STRIDE": STRIDE},
        "RF": {"WINDOW": WINDOW_SIZE, "STRIDE": STRIDE},
        "CNN-LSTM": {"WINDOW": WINDOW_SIZE, "STRIDE": STRIDE},
        "CNN-TCN": {"WINDOW": WINDOW_SIZE, "STRIDE": STRIDE},
    },
    "ED": {
        "SVM": {"WINDOW": WINDOW_SIZE, "STRIDE": STRIDE},
        "RF": {"WINDOW": WINDOW_SIZE, "STRIDE": STRIDE},
        "CNN-LSTM": {"WINDOW": WINDOW_SIZE, "STRIDE": STRIDE},
        "CNN-TCN": {"WINDOW": WINDOW_SIZE, "STRIDE": STRIDE},
    }
}

def get_best_window_stride(muscle=None, model_name=None):
    """Retorna la tupla (window_size, stride) canónica."""
    return WINDOW_SIZE, STRIDE

def get_model_hyperparams(model_name, muscle="FDS"):
    """
    Retorna un diccionario con los mejores hiperparámetros registrados para el modelo.
    Normaliza el nombre del modelo (ej. 'rf' -> 'RF', 'svm' -> 'SVM', etc.).
    """
    normalized_name = model_name.upper()
    if normalized_name in ["CNN_LSTM", "CNNLSTM"]:
        normalized_name = "CNN-LSTM"
    elif normalized_name in ["CNN_TCN", "CNNTCN"]:
        normalized_name = "CNN-TCN"

    muscle_key = muscle.upper()
    muscle_dict = BEST_HYPERPARAMS.get(muscle_key, BEST_HYPERPARAMS["FDS"])
    return muscle_dict.get(normalized_name, {}).copy()

def update_best_hyperparams(model_name, muscle, new_params, persist=True):
    """
    Actualiza los hiperparámetros óptimos en memoria y opcionalmente los persiste en disco.
    """
    normalized_name = model_name.upper()
    if normalized_name in ["CNN_LSTM", "CNNLSTM"]:
        normalized_name = "CNN-LSTM"
    elif normalized_name in ["CNN_TCN", "CNNTCN"]:
        normalized_name = "CNN-TCN"

    muscle_key = muscle.upper()
    if muscle_key not in BEST_HYPERPARAMS:
        BEST_HYPERPARAMS[muscle_key] = {}
    BEST_HYPERPARAMS[muscle_key][normalized_name] = new_params.copy()
    
    if persist:
        save_best_hyperparams()

def save_best_hyperparams():
    """Guarda el diccionario BEST_HYPERPARAMS en best_hyperparams.json."""
    import json
    from pathlib import Path
    json_path = Path(__file__).resolve().parent / "best_hyperparams.json"
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(BEST_HYPERPARAMS, f, indent=4)
    except Exception as e:
        print(f"⚠️ Error guardando hiperparámetros en {json_path}: {e}")

def load_best_hyperparams():
    """Carga los hiperparámetros guardados si el archivo JSON existe."""
    import json
    from pathlib import Path
    json_path = Path(__file__).resolve().parent / "best_hyperparams.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                for m, models in loaded.items():
                    if m not in BEST_HYPERPARAMS:
                        BEST_HYPERPARAMS[m] = {}
                    BEST_HYPERPARAMS[m].update(models)
        except Exception as e:
            print(f"⚠️ Error cargando hiperparámetros desde {json_path}: {e}")

# Cargar automáticamente al importar si existe
load_best_hyperparams()

