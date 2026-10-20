"""
=============================================================
 cnn_tcn.py — Modelo CNN-TCN para Clasificación sEMG
 Convoluciones causales dilatadas, sin pruning, INT8 y exportación C++
=============================================================
"""

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, SpatialDropout1D, 
    Add, Activation, GlobalAveragePooling1D, Dense
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold
from .. import config

def tcn_residual_block(x, filters: int, kernel_size: int, dilation_rate: int, dropout_rate: float = 0.2, block_id: int = 0):
    """
    Construye un bloque residual TCN con convolución 1D causal dilatada y conexión residual.
    """
    prev_x = x
    
    # 1ra Convolución causal dilatada
    conv1 = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        activation="relu",
        name=f"tcn_conv1d_a_{block_id}_d{dilation_rate}"
    )(x)
    bn1 = BatchNormalization(name=f"tcn_bn_a_{block_id}")(conv1)
    drop1 = SpatialDropout1D(dropout_rate, name=f"tcn_drop_a_{block_id}")(bn1)
    
    # 2da Convolución causal dilatada
    conv2 = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        activation="relu",
        name=f"tcn_conv1d_b_{block_id}_d{dilation_rate}"
    )(drop1)
    bn2 = BatchNormalization(name=f"tcn_bn_b_{block_id}")(conv2)
    drop2 = SpatialDropout1D(dropout_rate, name=f"tcn_drop_b_{block_id}")(bn2)
    
    # Conexión residual (ajuste 1x1 si difiere el número de canales)
    if prev_x.shape[-1] != filters:
        res_connection = Conv1D(
            filters=filters, 
            kernel_size=1, 
            padding="same", 
            name=f"tcn_res_conv1x1_{block_id}"
        )(prev_x)
    else:
        res_connection = prev_x
        
    out = Add(name=f"tcn_add_{block_id}")([res_connection, drop2])
    return Activation("relu", name=f"tcn_out_act_{block_id}")(out)

def build_model(
    window_size: Optional[int] = None, 
    num_channels: int = 1, 
    num_classes: int = 4, 
    nb_filters: int = 32, 
    kernel_size: int = 3, 
    dilations: Tuple[int, ...] = (1, 2, 4, 8), 
    dropout_rate: float = 0.2, 
    learning_rate: float = 1e-3
) -> Model:
    """
    Construye y compila el modelo CNN-TCN (Temporal Convolutional Network).
    100% libre de capas de poda, optimizado para inferencia rápida.
    """
    if window_size is None:
        window_size = config.WINDOW_SIZE
        
    inputs = Input(shape=(window_size, num_channels), name="input_semg")
    x = inputs
    
    # Convolución de entrada inicial
    x = Conv1D(filters=nb_filters, kernel_size=1, padding="same", name="tcn_initial_conv")(x)
    
    # Pila de Bloques Residuales TCN con dilatación creciente
    for i, d in enumerate(dilations):
        x = tcn_residual_block(
            x=x, 
            filters=nb_filters, 
            kernel_size=kernel_size, 
            dilation_rate=d, 
            dropout_rate=dropout_rate, 
            block_id=i+1
        )
        
    # Global Average Pooling temporal
    x = GlobalAveragePooling1D(name="tcn_global_gap")(x)
    
    # Clasificador denso
    x = Dense(32, activation="relu", name="dense_features")(x)
    outputs = Dense(num_classes, activation="softmax", name="output_gestures")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="CNN_TCN_MyoTensor")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    validation_data=None, 
    muscle: str = "FDS",
    epochs: int = 40, 
    batch_size: int = 32, 
    verbose: int = 0, 
    **kwargs
) -> Model:
    """
    Entrena el modelo CNN-TCN con EarlyStopping y dimensionamiento automático.
    Hereda hiperparámetros de config.py si no se pasan en kwargs.
    """
    if X_train.ndim == 2:
        X_train = np.expand_dims(X_train, axis=-1)
        
    default_params = config.get_model_hyperparams("CNN-TCN", muscle=muscle)
    final_params = {**default_params, **kwargs}
    
    window_size = X_train.shape[1]
    num_channels = X_train.shape[-1]
    nb_filters = final_params.get("nb_filters", 32)
    kernel_size = final_params.get("kernel_size", 3)
    dilations = tuple(final_params.get("dilations", (1, 2, 4, 8)))
    dropout_rate = final_params.get("dropout_rate", 0.2)
    learning_rate = final_params.get("learning_rate", 1e-3)
    
    model = build_model(
        window_size=window_size,
        num_channels=num_channels,
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        dilations=dilations,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )
    
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=verbose),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=verbose)
    ]
    
    if validation_data is not None:
        X_val, y_val = validation_data
        if X_val.ndim == 2:
            X_val = np.expand_dims(X_val, axis=-1)
        val_tuple = (X_val, y_val)
        X_train_fit, y_train_fit = X_train, y_train
    else:
        from sklearn.model_selection import train_test_split
        X_train_fit, X_val, y_train_fit, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )
        val_tuple = (X_val, y_val)
        
    model.fit(
        X_train_fit, y_train_fit,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=val_tuple,
        callbacks=callbacks,
        verbose=verbose
    )
    return model

def predict(model: Model, X: np.ndarray) -> np.ndarray:
    """Retorna las predicciones de clase como array 1D."""
    if X.ndim == 2:
        X = np.expand_dims(X, axis=-1)
    probs = model.predict(X, verbose=0)
    return np.argmax(probs, axis=1)

def evaluate(model: Model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Evalúa el modelo y retorna el diccionario de métricas estándar.
    """
    y_pred = predict(model, X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0).tolist()
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_per_class": f1_per_class,
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "confusion_matrix": cm,
        "classification_report": report
    }

def tune_hyperparameters(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    param_grid: Optional[Dict[str, Any]] = None, 
    cv: int = 3, 
    epochs: int = 25, 
    verbose: int = 0
) -> Tuple[Model, Dict[str, Any], float]:
    """
    Búsqueda en malla para CNN-TCN mediante K-Fold estratificado.
    Retorna: (best_model, best_params, best_cv_f1_score)
    """
    if X_train.ndim == 2:
        X_train = np.expand_dims(X_train, axis=-1)
        
    if param_grid is None:
        param_grid = [
            {"nb_filters": 16, "dilations": (1, 2, 4), "dropout_rate": 0.2, "learning_rate": 1e-3},
            {"nb_filters": 32, "dilations": (1, 2, 4), "dropout_rate": 0.2, "learning_rate": 1e-3},
            {"nb_filters": 32, "dilations": (1, 2, 4, 8), "dropout_rate": 0.2, "learning_rate": 1e-3},
            {"nb_filters": 32, "dilations": (1, 2, 4, 8), "dropout_rate": 0.3, "learning_rate": 5e-4},
        ]
    elif isinstance(param_grid, dict):
        import itertools
        keys, values = zip(*param_grid.items())
        param_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    best_score = -1.0
    best_params = {}
    best_model = None
    
    for candidate in param_grid:
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            
            m = train(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs, verbose=0, **candidate)
            res = evaluate(m, X_val, y_val)
            scores.append(res["f1_macro"])
            
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = candidate
            
    # Entrenar modelo final con toda la data
    best_model = train(X_train, y_train, epochs=epochs + 10, verbose=verbose, **best_params)
    return best_model, best_params, best_score

def quantize(model: Model, X_representative: np.ndarray, filename: str = "cnn_tcn_int8.tflite") -> Tuple[bytes, Dict[str, Any]]:
    """
    Cuantiza el modelo CNN-TCN a INT8 mediante Post-Training Quantization.
    """
    if X_representative.ndim == 2:
        X_representative = np.expand_dims(X_representative, axis=-1)
        
    def representative_dataset():
        n_samples = min(300, len(X_representative))
        for i in range(n_samples):
            yield [X_representative[i:i+1].astype(np.float32)]
            
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_bytes = converter.convert()
    
    with open(filename, "wb") as f:
        f.write(tflite_bytes)
        
    info = {
        "model_type": "CNN-TCN",
        "format": "TFLite INT8",
        "size_bytes": len(tflite_bytes),
        "size_kb": len(tflite_bytes) / 1024.0,
        "filename": filename
    }
    return tflite_bytes, info

def export_to_c(tflite_input, filename: str = "model_cnn_tcn.h", array_name: str = "cnn_tcn_tflite") -> str:
    """
    Exporta el binario TFLite a un archivo de cabecera C/C++ (#include <cstdint>)
    con alineación de 16 bytes esencial para microcontroladores ESP32-S3.
    """
    if isinstance(tflite_input, (str, Path)):
        with open(tflite_input, "rb") as f:
            tflite_bytes = f.read()
    else:
        tflite_bytes = tflite_input
        
    hex_values = ", ".join([f"0x{b:02x}" for b in tflite_bytes])
    header_content = f"""#pragma once
#include <cstdint>

// Modelo CNN-TCN Cuantizado a INT8 para TensorFlow Lite Micro
alignas(16) const unsigned char {array_name}[] = {{
    {hex_values}
}};
const unsigned int {array_name}_len = {len(tflite_bytes)};
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(header_content)
    return header_content
