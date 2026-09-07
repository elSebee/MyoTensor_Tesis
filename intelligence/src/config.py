BEST_PARAMS = {
    "FDS": {
        "SVM": {
            "WINDOW": 200,
            "STRIDE": 100,
            "ACCURACY": 80.92
        },
        "RF": {
            "WINDOW": 300,
            "STRIDE": 100,
            "ACCURACY": 79.65
        },
        "CNN-LSTM": {
            "WINDOW": 300,
            "STRIDE": 25,
            "ACCURACY": 85.8
        },
        "CNN-TCN": {
            "WINDOW": 300,
            "STRIDE": 50,
            "ACCURACY": 84.73
        }
    },
    "ED": {
        "SVM": {
            "WINDOW": 300,
            "STRIDE": 50,
            "ACCURACY": 96.14
        },
        "RF": {
            "WINDOW": 300,
            "STRIDE": 50,
            "ACCURACY": 94.64
        },
        "CNN-LSTM": {
            "WINDOW": 300,
            "STRIDE": 25,
            "ACCURACY": 98.33
        },
        "CNN-TCN": {
            "WINDOW": 300,
            "STRIDE": 25,
            "ACCURACY": 98.22
        }
    }
}

def get_best_window_stride(muscle, model_name):
    """Retorna la tupla (window_size, stride) óptima."""
    params = BEST_PARAMS.get(muscle, {}).get(model_name)
    if params is None and model_name == 'CNN':
        params = BEST_PARAMS.get(muscle, {}).get('CNN-LSTM', {})
    elif params is None:
        params = {}
    return params.get('WINDOW', 300), params.get('STRIDE', 50)
