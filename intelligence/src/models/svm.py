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
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import micromlgen
from .. import config
from ..cv import grouped_cv

def _make_pipeline(svc: SVC) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("svc", svc)])

def split_pipeline(model) -> Tuple[Optional[StandardScaler], SVC]:
    """Devuelve (scaler, svc). Si el modelo es un SVC suelto, scaler = None."""
    if isinstance(model, Pipeline):
        return model.named_steps["scaler"], model.named_steps["svc"]
    return None, model

def train(X_train: np.ndarray, y_train: np.ndarray, validation_data=None, muscle: str = "FDS",
          scale: bool = True, **kwargs):
    """
    Entrena un modelo Support Vector Machine (SVM).
    Si no se especifican hiperparámetros en kwargs, hereda automáticamente
    los mejores valores descubiertos en config.py para ese músculo.

    Con scale=True (por defecto) devuelve Pipeline(StandardScaler, SVC): se le entregan
    las características SIN escalar y el Z-score se ajusta solo con X_train.
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
    if scale:
        model = _make_pipeline(model)
    model.fit(X_train, y_train)
    return model

def tune_hyperparameters(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    groups: Optional[np.ndarray] = None,
    param_grid: Optional[Dict[str, Any]] = None, 
    cv: int = 5,
    verbose: int = 0,
    return_cv_results: bool = False
):
    """
    Optimiza hiperparámetros de SVM explorando kernels lineales y no lineales (RBF).

    La validación cruzada es agrupada (StratifiedGroupKFold): `groups` es obligatorio
    (WindowSet.groups('set') intra-sujeto o .groups('subject') inter-sujeto).
    El StandardScaler va dentro del Pipeline, así se ajusta en cada fold sin ver la validación.
    X_train son las características SIN escalar.

    Retorna: (best_model [Pipeline], best_params, best_cv_f1_score[, cv_results_ DataFrame])
    """
    if param_grid is None:
        param_grid = {
            'C': [0.1, 1.0, 10.0, 100.0],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto', 0.01, 0.1]
        }
    
    pipe = _make_pipeline(SVC(class_weight='balanced', random_state=42))
    folds = grouped_cv(y_train, groups, n_splits=cv)
    pipe_grid = {f"svc__{k}": v for k, v in param_grid.items()}
    
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=pipe_grid,
        cv=folds,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=verbose
    )
    grid_search.fit(X_train, y_train)
    best_params = {k.replace("svc__", "", 1): v for k, v in grid_search.best_params_.items()}
    out = (grid_search.best_estimator_, best_params, float(grid_search.best_score_))
    if return_cv_results:
        import pandas as pd
        out = out + (pd.DataFrame(grid_search.cv_results_),)
    return out

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

def quantize(model, X_representative: Optional[np.ndarray] = None, filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Para SVM, la representación en C++ se genera como pesos float estáticos.
    """
    _, model = split_pipeline(model)
    return {
        "model_type": "SVM",
        "kernel": model.kernel,
        "n_support_vectors": int(np.sum(model.n_support_)),
        "classes": len(model.classes_)
    }

def get_scaler_params(model) -> Optional[Dict[str, list]]:
    """Media y desviación del StandardScaler del Pipeline (para model_config.h del firmware)."""
    scaler, _ = split_pipeline(model)
    if scaler is None:
        return None
    return {"mean": scaler.mean_.tolist(), "std": scaler.scale_.tolist()}

def export_to_c(model, filename: str = "svm_model.h") -> str:
    """
    Exporta el modelo SVM a código C++ utilizando micromlgen.
    Si es un Pipeline, exporta solo el SVC; el scaler se obtiene con get_scaler_params().
    """
    _, model = split_pipeline(model)
    c_code = micromlgen.port(model)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(c_code)
    return c_code
