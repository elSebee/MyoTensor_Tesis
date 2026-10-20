#!/usr/bin/env python3
"""
====================================================================================================
 prepare_s01_models.py — Generador Autónomo de Modelos para el Sujeto S01
 
 Prepara de forma determinista y reproducible los modelos necesarios para el Experimento 06:
   1. Régimen 1 (Exp 03 - Poblacional): Entrena y persiste los modelos pooled en logs/models_pooled/
   2. Régimen 2 (Exp 04 - Zero-Shot LOSO): Verifica los modelos existentes en logs/models_loso/
   3. Régimen 3 (Exp 05 - Calibrado 10% S01): Adapta los modelos base con el 10% de S01 en logs/models_calibrated/
====================================================================================================
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Configuración de entorno TensorFlow y paths
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

curr = Path(__file__).resolve().parent
intelligence_dir = None
while curr != curr.parent:
    if (curr / "intelligence" / "src").is_dir():
        intelligence_dir = curr / "intelligence"
        break
    elif (curr / "src").is_dir() and (curr / "src" / "config.py").is_file():
        intelligence_dir = curr
        break
    curr = curr.parent

if intelligence_dir is None:
    raise RuntimeError("No se pudo localizar el directorio raíz 'intelligence'")

if str(intelligence_dir) not in sys.path:
    sys.path.insert(0, str(intelligence_dir))

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src import config, dataset, features
from src.models import rf, svm, cnn_lstm, cnn_tcn

# Fijar semillas globales
np.random.seed(config.RANDOM_STATE)
tf.random.set_seed(config.RANDOM_STATE)

# Directorios de destino
logs_dir = intelligence_dir / "src" / "experimentos" / "logs"
models_pooled_dir = logs_dir / "models_pooled"
models_loso_dir = logs_dir / "models_loso"
models_calib_dir = logs_dir / "models_calibrated"

os.makedirs(models_pooled_dir, exist_ok=True)
os.makedirs(models_calib_dir, exist_ok=True)

MUSCLES = ["ED", "FDS"]
ALL_SUBJECTS = [f"S{i:02d}" for i in range(1, 11)]
TARGET_SUBJECT = "S01"


def adapt_cnn_lstm(model_path, X_cal, y_cal, epochs=20, batch_size=16, lr=2e-4):
    """Fine-Tuning de CNN-LSTM: congela convoluciones, descongela LSTM y denso."""
    model = tf.keras.models.load_model(model_path)
    for layer in model.layers:
        if layer.name in ["lstm_layer", "dropout_dense", "dense_features", "output_gestures"]:
            layer.trainable = True
        else:
            layer.trainable = False
            
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_cal, y_cal, epochs=epochs, batch_size=batch_size, verbose=0)
    return model


def adapt_cnn_tcn(model_path, X_cal, y_cal, epochs=20, batch_size=16, lr=2e-4):
    """Fine-Tuning de CNN-TCN: congela bloques iniciales, descongela último bloque residual y denso."""
    model = tf.keras.models.load_model(model_path)
    for layer in model.layers:
        if "3" in layer.name or "dense" in layer.name or "output" in layer.name or "gap" in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False
            
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_cal, y_cal, epochs=epochs, batch_size=batch_size, verbose=0)
    return model


def prepare_exp03_pooled_models(force=False):
    """Entrena y guarda los modelos poblacionales globales en logs/models_pooled/."""
    print("\n" + "="*85)
    print(" PASO 1/3: PREPARACIÓN DE MODELOS POBLACIONALES (EXPERIMENTO 03)")
    print("="*85)
    
    for muscle in MUSCLES:
        print(f"\n[+] Evaluando necesidad de entrenamiento poblacional para: {muscle}")
        target_files = [
            models_pooled_dir / f"pooled_{muscle}_CNN-TCN.keras",
            models_pooled_dir / f"pooled_{muscle}_CNN-LSTM.keras",
            models_pooled_dir / f"pooled_{muscle}_SVM.joblib",
            models_pooled_dir / f"pooled_{muscle}_RF.joblib",
            models_pooled_dir / f"pooled_{muscle}_scaler.joblib"
        ]
        
        if not force and all(p.is_file() for p in target_files):
            print(f"  ✅ Modelos poblacionales ({muscle}) ya existen en disco. Omitiendo.")
            continue
            
        print(f"  [*] Cargando dataset poblacional agregado para {muscle} (S01-S10)...")
        X_train_raw_list, y_train_list = [], []
        X_test_raw_list, y_test_list   = [], []
        X_train_feat_list, X_test_feat_list = [], []
        
        for s in ALL_SUBJECTS:
            X_tr, y_tr, X_te, y_te = dataset.load_and_segment_dataset(
                subject=s, muscle=muscle,
                window_size=config.WINDOW_SIZE, overlap=config.STRIDE,
                apply_mvc=True, steady_state_only=True
            )
            f_tr = features.extract_features(X_tr, subject=s, muscle=muscle)
            f_te = features.extract_features(X_te, subject=s, muscle=muscle)
            
            X_train_raw_list.append(X_tr)
            y_train_list.append(y_tr)
            X_test_raw_list.append(X_te)
            y_test_list.append(y_te)
            X_train_feat_list.append(f_tr)
            X_test_feat_list.append(f_te)
            
        X_train_raw  = np.concatenate(X_train_raw_list, axis=0)
        y_train      = np.concatenate(y_train_list, axis=0)
        X_test_raw   = np.concatenate(X_test_raw_list, axis=0)
        y_test       = np.concatenate(y_test_list, axis=0)
        
        X_train_feat = np.concatenate(X_train_feat_list, axis=0)
        X_test_feat  = np.concatenate(X_test_feat_list, axis=0)
        
        X_train_feat_sc, X_test_feat_sc, scaler = features.scale_features(X_train_feat, X_test_feat)
        joblib.dump(scaler, models_pooled_dir / f"pooled_{muscle}_scaler.joblib")
        
        # 1. Random Forest
        print(f"  -> Entrenando Random Forest Poblacional ({muscle})...")
        model_rf = rf.train(X_train_feat, y_train, muscle=muscle)
        joblib.dump(model_rf, models_pooled_dir / f"pooled_{muscle}_RF.joblib")
        
        # 2. SVM
        print(f"  -> Entrenando SVM Poblacional ({muscle})...")
        model_svm = svm.train(X_train_feat_sc, y_train, muscle=muscle)
        joblib.dump(model_svm, models_pooled_dir / f"pooled_{muscle}_SVM.joblib")
        
        # 3. CNN-LSTM
        print(f"  -> Entrenando CNN-LSTM Poblacional ({muscle})...")
        model_lstm = cnn_lstm.train(
            X_train_raw, y_train,
            muscle=muscle,
            epochs=40,
            batch_size=64,
            verbose=0
        )
        model_lstm.save(models_pooled_dir / f"pooled_{muscle}_CNN-LSTM.keras")
        
        # 4. CNN-TCN
        print(f"  -> Entrenando CNN-TCN Poblacional ({muscle})...")
        model_tcn = cnn_tcn.train(
            X_train_raw, y_train,
            muscle=muscle,
            epochs=40,
            batch_size=64,
            verbose=0
        )
        model_tcn.save(models_pooled_dir / f"pooled_{muscle}_CNN-TCN.keras")
        print(f"  ✅ Guardados 4 modelos poblacionales de {muscle} en {models_pooled_dir}")


def verify_exp04_loso_models():
    """Verifica la existencia de los modelos LOSO de S01 en logs/models_loso/."""
    print("\n" + "="*85)
    print(" PASO 2/3: VERIFICACIÓN DE MODELOS ZERO-SHOT LOSO (EXPERIMENTO 04)")
    print("="*85)
    
    missing = []
    for muscle in MUSCLES:
        for model_name, ext in [("CNN-TCN", "keras"), ("CNN-LSTM", "keras"), ("SVM", "joblib"), ("RF", "joblib")]:
            p = models_loso_dir / f"loso_{muscle}_{model_name}_{TARGET_SUBJECT}.{ext}"
            if p.is_file():
                print(f"  ✅ Encontrado: {p.name}")
            else:
                missing.append(str(p))
                print(f"  ❌ Faltante: {p.name}")
                
    if missing:
        raise RuntimeError(f"Faltan modelos de LOSO para S01. Ejecuta primero 04_Generalizacion_LOSO.ipynb: {missing}")
    print(f"  🎉 Todos los modelos LOSO de {TARGET_SUBJECT} están verificados en disco.")


def prepare_exp05_calibrated_models(force=False):
    """Adapta con el 10% de S01 y guarda los modelos calibrados en logs/models_calibrated/."""
    print("\n" + "="*85)
    print(" PASO 3/3: ADAPTACIÓN Y CALIBRACIÓN DEL 10% PARA S01 (EXPERIMENTO 05)")
    print("="*85)
    
    for muscle in MUSCLES:
        print(f"\n[+] Procesando calibración rápida para S01 — Músculo: {muscle}")
        target_files = [
            models_calib_dir / f"calib10_{muscle}_CNN-TCN_{TARGET_SUBJECT}.keras",
            models_calib_dir / f"calib10_{muscle}_CNN-LSTM_{TARGET_SUBJECT}.keras",
            models_calib_dir / f"calib10_{muscle}_SVM_{TARGET_SUBJECT}.joblib",
            models_calib_dir / f"calib10_{muscle}_RF_{TARGET_SUBJECT}.joblib",
            models_calib_dir / f"calib10_{muscle}_scaler_{TARGET_SUBJECT}.joblib"
        ]
        
        if not force and all(p.is_file() for p in target_files):
            print(f"  ✅ Modelos calibrados de {TARGET_SUBJECT} ({muscle}) ya existen. Omitiendo.")
            continue
            
        # Carga segmentada de datos de S01
        X_tr_raw, y_tr, X_te_raw, y_te = dataset.load_and_segment_dataset(
            subject=TARGET_SUBJECT, muscle=muscle,
            window_size=config.WINDOW_SIZE, overlap=config.STRIDE,
            apply_mvc=True, steady_state_only=True
        )
        X_raw = np.concatenate([X_tr_raw, X_te_raw], axis=0)
        y = np.concatenate([y_tr, y_te], axis=0)
        X_feat = features.extract_features(X_raw, subject=TARGET_SUBJECT, muscle=muscle)
        
        # Partición idéntica a Exp 05: 10% calibración (~5s/gesto) / 90% test
        idx_calib, idx_test = train_test_split(
            np.arange(len(y)), train_size=0.10, stratify=y, random_state=42
        )
        X_cal_raw, y_cal = X_raw[idx_calib], y[idx_calib]
        X_cal_feat       = X_feat[idx_calib]
        
        # Escalador ajustado sobre el 10% de calibración
        scaler = StandardScaler()
        X_cal_feat_sc = scaler.fit_transform(X_cal_feat)
        joblib.dump(scaler, models_calib_dir / f"calib10_{muscle}_scaler_{TARGET_SUBJECT}.joblib")
        
        # 1. Random Forest (Calibración)
        print(f"  -> Calibrando Random Forest...")
        rf_params = config.get_model_hyperparams("RF", muscle=muscle)
        rf_model = RandomForestClassifier(
            n_estimators=rf_params.get("n_estimators", 100),
            max_depth=rf_params.get("max_depth", 15),
            min_samples_split=rf_params.get("min_samples_split", 4),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_cal_feat, y_cal)
        joblib.dump(rf_model, models_calib_dir / f"calib10_{muscle}_RF_{TARGET_SUBJECT}.joblib")
        
        # 2. SVM (Calibración)
        print(f"  -> Calibrando SVM...")
        svm_params = config.get_model_hyperparams("SVM", muscle=muscle)
        svm_model = SVC(
            C=svm_params.get("C", 50.0),
            kernel=svm_params.get("kernel", "rbf"),
            gamma=svm_params.get("gamma", 0.1 if muscle == "FDS" else "scale"),
            class_weight="balanced",
            random_state=42
        )
        svm_model.fit(X_cal_feat_sc, y_cal)
        joblib.dump(svm_model, models_calib_dir / f"calib10_{muscle}_SVM_{TARGET_SUBJECT}.joblib")
        
        # 3. CNN-LSTM (Fine-Tuning desde modelo base LOSO S01)
        print(f"  -> Calibrando CNN-LSTM (Fine-Tuning)...")
        lstm_base = models_loso_dir / f"loso_{muscle}_CNN-LSTM_{TARGET_SUBJECT}.keras"
        lstm_ft = adapt_cnn_lstm(lstm_base, X_cal_raw, y_cal, epochs=20, batch_size=16)
        lstm_ft.save(models_calib_dir / f"calib10_{muscle}_CNN-LSTM_{TARGET_SUBJECT}.keras")
        
        # 4. CNN-TCN (Fine-Tuning desde modelo base LOSO S01)
        print(f"  -> Calibrando CNN-TCN (Fine-Tuning)...")
        tcn_base = models_loso_dir / f"loso_{muscle}_CNN-TCN_{TARGET_SUBJECT}.keras"
        tcn_ft = adapt_cnn_tcn(tcn_base, X_cal_raw, y_cal, epochs=20, batch_size=16)
        tcn_ft.save(models_calib_dir / f"calib10_{muscle}_CNN-TCN_{TARGET_SUBJECT}.keras")
        
        print(f"  ✅ Guardados 4 modelos calibrados de {muscle} ({TARGET_SUBJECT}) en {models_calib_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepara los modelos de S01 para Exp 06.")
    parser.add_argument("--force", action="store_true", help="Fuerza re-entrenamiento incluso si ya existen.")
    args = parser.parse_args()
    
    t_start = time.time()
    print("="*90)
    print(f" SCRIPT DE PREPARACIÓN DE MODELOS PARA SUJETO {TARGET_SUBJECT} (EXP 03, 04, 05)")
    print("="*90)
    
    prepare_exp03_pooled_models(force=args.force)
    verify_exp04_loso_models()
    prepare_exp05_calibrated_models(force=args.force)
    
    dt = (time.time() - t_start) / 60
    print("\n" + "="*90)
    print(f" 🎉 TODOS LOS MODELOS DE {TARGET_SUBJECT} ESTÁN LISTOS EN DISCO EN {dt:.2f} MINUTOS")
    print("  - logs/models_pooled/      (Exp 03 Poblacional)")
    print("  - logs/models_loso/        (Exp 04 Zero-Shot)")
    print("  - logs/models_calibrated/  (Exp 05 Calibrado 10%)")
    print("="*90 + "\n")


if __name__ == "__main__":
    main()
