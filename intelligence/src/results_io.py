"""
=============================================================
 results_io.py — Registro trazable de resultados
=============================================================

Toda cifra de la tesis debe salir de un archivo escrito por `save_result()`.
Cada CSV va acompañado de `<nombre>.meta.json` con commit de git, fecha, semilla,
versiones de librerías y configuración de segmentación, para poder reproducirlo.

Los resultados se guardan en experimentos/logs/results/ (versionado en git).
"""

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from . import config

RESULTS_DIR = Path(__file__).resolve().parent / "experimentos" / "logs" / "results"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_state() -> Dict[str, Any]:
    status = _git("status", "--porcelain", "--", "intelligence/src")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # Cambios sin commitear en el código de IA: el resultado no es reproducible desde el commit
        "dirty": bool(status),
        "dirty_files": status.splitlines() if status else [],
    }


def library_versions() -> Dict[str, str]:
    versions = {"python": platform.python_version()}
    # Solo librerías ya importadas: no forzar la carga de TensorFlow
    for name in ("numpy", "pandas", "sklearn", "scipy", "tensorflow", "micromlgen"):
        mod = sys.modules.get(name)
        if mod is not None:
            versions[name] = getattr(mod, "__version__", "unknown")
    return versions


def save_result(
    df: pd.DataFrame,
    name: str,
    experiment: str,
    params: Optional[Dict[str, Any]] = None,
    subdir: Optional[str] = None,
) -> Path:
    """
    Guarda `df` como CSV y su metadata.

    Args:
        name       : nombre base del archivo (sin extensión), p. ej. 'exp05v2_calibracion'.
        experiment : ID del experimento, p. ej. 'Exp05v2'.
        params     : parámetros relevantes (regímenes, k, semillas, mallas, etc.).
        subdir     : subcarpeta dentro de results/ (opcional).
    """
    out_dir = RESULTS_DIR / subdir if subdir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)

    meta = {
        "name": name,
        "experiment": experiment,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "git": git_state(),
        "versions": library_versions(),
        "config": {
            "WINDOW_SIZE": config.WINDOW_SIZE,
            "STRIDE": config.STRIDE,
            "SAMPLING_RATE_HZ": config.SAMPLING_RATE_HZ,
            "RANDOM_STATE": config.RANDOM_STATE,
        },
        "params": params or {},
    }
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    if meta["git"]["dirty"]:
        print(f"⚠️  {name}: hay cambios sin commitear en intelligence/src; el resultado no queda atado a un commit.")
    return csv_path


def load_result(name: str, subdir: Optional[str] = None):
    """Devuelve (DataFrame, metadata)."""
    base = RESULTS_DIR / subdir if subdir else RESULTS_DIR
    df = pd.read_csv(base / f"{name}.csv")
    meta = json.loads((base / f"{name}.meta.json").read_text(encoding="utf-8"))
    return df, meta
