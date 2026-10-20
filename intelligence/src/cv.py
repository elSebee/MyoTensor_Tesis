"""
=============================================================
 cv.py — Validación cruzada agrupada (sin fuga por solapamiento)
=============================================================

Con ventanas de 250 ms y paso de 50 ms (80 % de solapamiento), dos ventanas vecinas
comparten 200 ms de señal. Un K-Fold sobre ventanas sueltas pone ventanas casi idénticas
en entrenamiento y validación, e infla el puntaje. Aquí todo se agrupa:
  - intra-sujeto : groups = set_id         (WindowSet.groups('set'))
  - inter-sujeto : groups = sujeto         (WindowSet.groups('subject'))
"""

from typing import List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def _check_groups(y: np.ndarray, groups) -> np.ndarray:
    if groups is None:
        raise ValueError(
            "Se requiere `groups` (p. ej. WindowSet.groups('set') o .groups('subject')). "
            "La validación por ventana sueltas produce fuga de información con ventanas solapadas."
        )
    groups = np.asarray(groups)
    if len(groups) != len(y):
        raise ValueError(f"len(groups)={len(groups)} distinto de len(y)={len(y)}.")
    return groups


def grouped_cv(y: np.ndarray, groups, n_splits: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Folds estratificados por clase y agrupados (ningún grupo aparece en train y validación).
    Si hay menos grupos que `n_splits`, usa tantos folds como grupos.
    """
    groups = _check_groups(y, groups)
    n_groups = np.unique(groups).size
    if n_groups < 2:
        raise ValueError("Se necesitan al menos 2 grupos para validación cruzada agrupada.")
    k = min(n_splits, n_groups)
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups))


def grouped_val_split(y: np.ndarray, groups, val_fraction: float = 0.15, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separa un subconjunto de validación (para EarlyStopping) por grupos completos.
    Devuelve (train_idx, val_idx).
    """
    groups = _check_groups(y, groups)
    n_groups = np.unique(groups).size
    if n_groups < 2:
        raise ValueError("Se necesitan al menos 2 grupos para separar validación.")
    k = int(np.clip(round(1.0 / val_fraction), 2, n_groups))
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return next(cv.split(np.zeros(len(y)), y, groups))


def assert_no_group_leak(groups, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    groups = np.asarray(groups)
    shared = np.intersect1d(groups[train_idx], groups[test_idx])
    if shared.size:
        raise AssertionError(f"Grupos presentes en train y test: {shared[:10].tolist()}")
