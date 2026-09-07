import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, SpatialDropout1D, 
    Add, Activation, GlobalAveragePooling1D, Dense
)
import tensorflow_model_optimization as tfmot


def tcn_residual_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.2, block_id=0):
    """
    Construye un bloque residual TCN con convolución 1D causal dilatada.
    
    Args:
        x: Tensor de entrada.
        filters (int): Número de filtros convolucionales.
        kernel_size (int): Tamaños de kernel de la convolución.
        dilation_rate (int): Factor de dilatación de la convolución temporal.
        dropout_rate (float): Tasa de espacial dropout.
        block_id (int): Identificador del bloque para nombres de capas.
        
    Returns:
        Tensor de salida del bloque residual.
    """
    prev_x = x
    
    # Primera convolución causal dilatada
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
    
    # Segunda convolución causal dilatada
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
    
    # Conexión residual (ajuste 1x1 si las dimensiones de canales difieren)
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
    window_size=300, 
    num_channels=1, 
    num_classes=4, 
    nb_filters=32, 
    kernel_size=3, 
    dilations=(1, 2, 4, 8), 
    dropout_rate=0.2, 
    learning_rate=1e-3
):
    """
    Construye y compila el modelo CNN-TCN (Temporal Convolutional Network) para sEMG.
    Aprovecha convoluciones causales dilatadas y conexiones residuales.
    
    Args:
        window_size (int): Tamaño de la ventana temporal de entrada (muestras).
        num_channels (int): Número de canales musculares (1 para monofásico).
        num_classes (int): Cantidad de clases/gestos a clasificar.
        nb_filters (int): Número de filtros por bloque TCN.
        kernel_size (int): Tamaños de kernel convolucional.
        dilations (tuple/list): Lista de tasas de dilatación (ej. 1, 2, 4, 8).
        dropout_rate (float): Tasa de dropout.
        learning_rate (float): Tasa de aprendizaje del optimizador.
        
    Returns:
        tf.keras.Model: Modelo Keras compilado.
    """
    inputs = Input(shape=(window_size, num_channels), name="input_semg")
    x = inputs
    
    # Convolución de entrada inicial
    x = Conv1D(filters=nb_filters, kernel_size=1, padding="same", name="tcn_initial_conv")(x)
    
    # Pila de Bloques Residuales TCN con Dilatación Creciente
    for i, d in enumerate(dilations):
        x = tcn_residual_block(
            x=x, 
            filters=nb_filters, 
            kernel_size=kernel_size, 
            dilation_rate=d, 
            dropout_rate=dropout_rate, 
            block_id=i+1
        )
        
    # Global Pooling Temporal (colapsa la dimensión de tiempo sin pérdida espacial)
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


def train_with_pruning(model, X_train, y_train, epochs=20, batch_size=32, validation_data=None):
    """
    Aplica poda de pesos (Weight Pruning) durante el entrenamiento.
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


def export_tflite_int8(model, X_representative, filename="cnn_tcn_int8.tflite"):
    """
    Cuantiza el modelo CNN-TCN a INT8 usando representational dataset y lo exporta a archivo .tflite.
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
