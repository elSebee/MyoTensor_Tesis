"""
=============================================================
 rf.py — Módulo de Random Forest para Clasificación sEMG
 Implementación sin poda, con búsqueda de hiperparámetros y exportación a C++
=============================================================
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import micromlgen
from .. import config

def train(X_train: np.ndarray, y_train: np.ndarray, validation_data=None, muscle: str = "FDS", **kwargs) -> RandomForestClassifier:
    """
    Entrena un modelo Random Forest (RF).
    Si no se pasan parámetros en kwargs, carga automáticamente los mejores
    hiperparámetros registrados en config.py para ese músculo.
    """
    default_params = config.get_model_hyperparams("RF", muscle=muscle)
    final_params = {**default_params, **kwargs}
    
    n_estimators = final_params.get("n_estimators", 100)
    max_depth = final_params.get("max_depth", 15)
    min_samples_split = final_params.get("min_samples_split", 4)
    class_weight = final_params.get("class_weight", "balanced")
    random_state = final_params.get("random_state", 42)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def tune_hyperparameters(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    param_grid: Optional[Dict[str, Any]] = None, 
    cv: int = 5,
    verbose: int = 0
) -> Tuple[RandomForestClassifier, Dict[str, Any], float]:
    """
    Optimiza y entrena un modelo Random Forest usando GridSearchCV.
    Retorna: (best_model, best_params, best_cv_f1_score)
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 15, 20],
            'min_samples_split': [2, 4, 8]
        }
    
    rf_base = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=verbose
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, float(grid_search.best_score_)

# Alias de compatibilidad
train_with_gridsearch = tune_hyperparameters

def predict(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """Retorna las predicciones de clase como array 1D."""
    return model.predict(X)

def evaluate(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
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

def quantize(model: RandomForestClassifier, X_representative: Optional[np.ndarray] = None, filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Para modelos de Machine Learning clásico (árboles), la representación se mantiene
    en estructuras C/C++ enteras o de coma flotante estándar.
    """
    return {
        "model_type": "RandomForest",
        "quantization": "Native_Tree_Logic",
        "n_estimators": model.n_estimators,
        "n_classes": len(model.classes_)
    }

def export_to_c(model: RandomForestClassifier, filename: str = "rf_model.h") -> str:
    """
    Exporta el modelo RF a código C++ utilizando micromlgen.
    """
    c_code = micromlgen.port(model)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(c_code)
    return c_code
