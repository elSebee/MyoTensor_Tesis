import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D, 
    LSTM, Dropout, Dense, SpatialDropout1D
)
import tensorflow_model_optimization as tfmot


def build_model(
    window_size=300, 
    num_channels=1, 
    num_classes=4, 
    conv_filters=(32, 64), 
    kernel_size=3, 
    lstm_units=64, 
    dropout_rate=0.3, 
    learning_rate=1e-3
):
    """
    Construye y compila el modelo 1D-CNN + LSTM para clasificación sEMG.
    Diseñado para ser compatible con TensorFlow Lite Micro (INT8).
    
    Args:
        window_size (int): Tamaño de la ventana temporal de entrada (muestras).
        num_channels (int): Número de canales musculares (1 para monofásico).
        num_classes (int): Cantidad de clases/gestos a clasificar.
        conv_filters (tuple/list): Filtros para las capas Conv1D.
        kernel_size (int): Tamaños de kernel para las convoluciones.
        lstm_units (int): Cantidad de unidades recurrentes en la capa LSTM.
        dropout_rate (float): Tasa de dropout para regularización.
        learning_rate (float): Tasa de aprendizaje del optimizador Adam.
        
    Returns:
        tf.keras.Model: Modelo Keras compilado.
    """
    inputs = Input(shape=(window_size, num_channels), name="input_semg")
    
    x = inputs
    # Bloques Convolucionales (Extracción de características locales)
    for i, filters in enumerate(conv_filters):
        x = Conv1D(
            filters=filters, 
            kernel_size=kernel_size, 
            padding="same", 
            activation="relu",
            name=f"conv1d_{i+1}"
        )(x)
        x = BatchNormalization(name=f"bn_{i+1}")(x)
        x = MaxPooling1D(pool_size=2, name=f"pool_{i+1}")(x)
        x = SpatialDropout1D(dropout_rate, name=f"spatial_dropout_{i+1}")(x)
    
    # Capa Recurrente (Dinámica temporal unidireccional desenrollada para TFLite Micro)
    x = LSTM(lstm_units, return_sequences=False, unroll=True, name="lstm_layer")(x)
    x = Dropout(dropout_rate, name="dropout_dense")(x)
    
    # Capas de Clasificación
    x = Dense(32, activation="relu", name="dense_features")(x)
    outputs = Dense(num_classes, activation="softmax", name="output_gestures")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="CNN_LSTM_MyoTensor")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_with_pruning(model, X_train, y_train, epochs=20, batch_size=32, validation_data=None):
    """
    Aplica poda de pesos (Weight Pruning) durante el entrenamiento para optimizar almacenamiento en ESP32-S3.
    """
    if X_train.ndim == 2:
        X_train = np.expand_dims(X_train, axis=-1)
        
    if validation_data is not None:
        X_val, y_val = validation_data
        if X_val.ndim == 2:
            X_val = np.expand_dims(X_val, axis=-1)
        validation_data = (X_val, y_val)
        
    num_samples = X_train.shape[0]
    end_step = np.ceil(num_samples / batch_size).astype(np.int32) * epochs
    
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.70,
            begin_step=0,
            end_step=end_step)
    }
    
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    model_for_pruning.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
                              
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep()
    ]
    
    model_for_pruning.fit(
        X_train, y_train, 
        batch_size=batch_size, 
        epochs=epochs, 
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1
    )
                          
    model_exported = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    model_exported.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model_exported


def export_tflite_int8(model, X_representative, filename="cnn_lstm_int8.tflite"):
    """
    Cuantiza el modelo a INT8 usando representational dataset y lo exporta a archivo .tflite.
    """
    if X_representative.ndim == 2:
        X_representative = np.expand_dims(X_representative, axis=-1)
        
    def representative_dataset():
        for i in range(min(500, len(X_representative))):
            yield [X_representative[i:i+1].astype(np.float32)]
            
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    with open(filename, 'wb') as f:
        f.write(tflite_model)
        
    return tflite_model
