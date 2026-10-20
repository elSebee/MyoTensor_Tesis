"""
=============================================================
 svm.py — Módulo de Support Vector Machine para sEMG
 Soporta búsqueda amplia (Linear, RBF), herencia de config y exportación C++
=============================================================
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import micromlgen
from .. import config

def train(X_train: np.ndarray, y_train: np.ndarray, validation_data=None, muscle: str = "FDS", **kwargs) -> SVC:
    """
    Entrena un modelo Support Vector Machine (SVM).
    Si no se especifican hiperparámetros en kwargs, hereda automáticamente
    los mejores valores descubiertos en config.py para ese músculo.
    """
    default_params = config.get_model_hyperparams("SVM", muscle=muscle)
    final_params = {**default_params, **kwargs}
    
    C = final_params.get("C", 10.0)
    kernel = final_params.get("kernel", "rbf")
    gamma = final_params.get("gamma", "scale")
    class_weight = final_params.get("class_weight", "balanced")
    random_state = final_params.get("random_state", 42)
    
    model = SVC(
        C=C, 
        kernel=kernel, 
        gamma=gamma, 
        class_weight=class_weight, 
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def tune_hyperparameters(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    param_grid: Optional[Dict[str, Any]] = None, 
    cv: int = 5,
    verbose: int = 0
) -> Tuple[SVC, Dict[str, Any], float]:
    """
    Optimiza hiperparámetros de SVM explorando kernels lineales y no lineales (RBF).
    Retorna: (best_model, best_params, best_cv_f1_score)
    """
    if param_grid is None:
        param_grid = {
            'C': [0.1, 1.0, 10.0, 100.0],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto', 0.01, 0.1]
        }
    
    svm_base = SVC(class_weight='balanced', random_state=42)
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=svm_base,
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

def predict(model: SVC, X: np.ndarray) -> np.ndarray:
    """Retorna las predicciones de clase como array 1D."""
    return model.predict(X)

def evaluate(model: SVC, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
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

def quantize(model: SVC, X_representative: Optional[np.ndarray] = None, filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Para SVM, la representación en C++ se genera como pesos float estáticos.
    """
    return {
        "model_type": "SVM",
        "kernel": model.kernel,
        "n_support_vectors": int(np.sum(model.n_support_)),
        "classes": len(model.classes_)
    }

def export_to_c(model: SVC, filename: str = "svm_model.h") -> str:
    """
    Exporta el modelo SVM a código C++ utilizando micromlgen.
    """
    c_code = micromlgen.port(model)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(c_code)
    return c_code
