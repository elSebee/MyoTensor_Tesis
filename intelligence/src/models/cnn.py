import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM, TimeDistributed, Dropout, Input
import tensorflow_model_optimization as tfmot

def build_model(window_size=300):
    """
    Construye la arquitectura MS-CLSTM (Multi-Stream CNN-LSTM).
    """
    # Usar TF_USE_LEGACY_KERAS=1 para asegurar compatibilidad con tfmot
    import os
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    
    model = Sequential([
        Input(shape=(window_size, 1)),
        # CNN Feature Extractor
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Temporal Dynamics
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        
        # Classifier
        Dense(32, activation='relu'),
        Dense(4, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_with_pruning(model, X_train, y_train, epochs=20, batch_size=32, validation_data=None):
    """
    Aplica poda de pesos (Weight Pruning) durante el entrenamiento.
    """
    import os
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    
    # Asegurarnos de que X_train tenga shape (N, W, 1)
    if X_train.ndim == 2:
        X_train = np.expand_dims(X_train, axis=-1)
        
    if validation_data is not None:
        X_val, y_val = validation_data
        if X_val.ndim == 2:
            X_val = np.expand_dims(X_val, axis=-1)
        validation_data = (X_val, y_val)
        
    num_images = X_train.shape[0]
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs
    
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.70,
            begin_step=0,
            end_step=end_step)
    }
    
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    model_for_pruning.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
                              
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir='./logs')
    ]
    
    model_for_pruning.fit(X_train, y_train, 
                          batch_size=batch_size, 
                          epochs=epochs, 
                          validation_data=validation_data,
                          callbacks=callbacks,
                          verbose=1)
                          
    model_exported = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    model_exported.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
    return model_exported

def export_tflite_int8(model, X_representative, filename="model_int8.tflite"):
    """
    Cuantiza el modelo a INT8 usando representational data y lo exporta a un archivo .tflite.
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
