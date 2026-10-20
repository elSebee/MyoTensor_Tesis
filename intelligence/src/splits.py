"""
=============================================================
 splits.py — Particiones canónicas por set (fuente única)
=============================================================

Todas las particiones se definen por SET COMPLETO (`set_id`, 1..10), en orden
cronológico. Ningún experimento debe partir por ventana.

Particiones:
  - intra     : entrenamiento = sets 1–7,   prueba = sets 8–10   (Exp 01–03)
  - calib(k)  : calibración  = sets 1..k,   prueba = sets 4–10   (Exp 05; k ∈ {1,2,3})
                La prueba es FIJA para todo k, así las curvas de calibración son comparables.
                k = 1 ≡ 5 s por gesto (≈ 4 s útiles tras el recorte steady-state).
  - loso      : entrenamiento = los otros sujetos (todos sus sets),
                prueba = sets 4–10 del sujeto excluido (las mismas ventanas que calib(k),
                de modo que 0 % y k % se evalúan sobre exactamente los mismos datos).

Uso como script (genera los JSON de índices que reutiliza el banco HIL del Exp 06):
    python -m src.splits                # configuración canónica
    python -m src.splits --no-steady    # contracción completa
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import config
from .dataset import DEFAULT_SUBJECTS, MUSCLES, WindowSet, load_windows

N_SETS = 10
INTRA_TRAIN_SETS: Tuple[int, ...] = tuple(range(1, 8))
INTRA_TEST_SETS: Tuple[int, ...] = (8, 9, 10)
CALIB_K: Tuple[int, ...] = (1, 2, 3)
CALIB_TEST_SETS: Tuple[int, ...] = tuple(range(4, N_SETS + 1))
LOSO_TEST_SETS: Tuple[int, ...] = CALIB_TEST_SETS

SPLITS_DIR = Path(__file__).resolve().parent / "experimentos" / "logs" / "splits"


def _set_ids(ws_or_sets) -> np.ndarray:
    return ws_or_sets.set_id if isinstance(ws_or_sets, WindowSet) else np.asarray(ws_or_sets)


def mask_sets(ws_or_sets, sets: Sequence[int]) -> np.ndarray:
    """Máscara booleana de las ventanas cuyo set está en `sets`."""
    return np.isin(_set_ids(ws_or_sets), np.asarray(sets))


def assert_disjoint(ws_or_sets, mask_a: np.ndarray, mask_b: np.ndarray, label: str = "") -> None:
    """Falla si dos particiones comparten ventanas o sets."""
    sets = _set_ids(ws_or_sets)
    if np.any(mask_a & mask_b):
        raise AssertionError(f"[{label}] Particiones comparten ventanas.")
    shared = np.intersect1d(sets[mask_a], sets[mask_b])
    if shared.size:
        raise AssertionError(f"[{label}] Particiones comparten sets: {shared.tolist()}")


def intra_split(ws: WindowSet) -> Tuple[np.ndarray, np.ndarray]:
    """(train_mask, test_mask): sets 1–7 vs. 8–10."""
    tr, te = mask_sets(ws, INTRA_TRAIN_SETS), mask_sets(ws, INTRA_TEST_SETS)
    assert_disjoint(ws, tr, te, "intra")
    return tr, te


def calib_sets(k: int) -> Tuple[int, ...]:
    if k not in CALIB_K:
        raise ValueError(f"k={k} no válido; use uno de {CALIB_K} (la prueba fija empieza en el set 4).")
    return tuple(range(1, k + 1))


def calib_split(ws: WindowSet, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """(calib_mask, test_mask): sets 1..k vs. 4–10 (prueba fija)."""
    cal, te = mask_sets(ws, calib_sets(k)), mask_sets(ws, CALIB_TEST_SETS)
    assert_disjoint(ws, cal, te, f"calib k={k}")
    return cal, te


def loso_test_mask(ws: WindowSet) -> np.ndarray:
    """Ventanas de prueba del sujeto excluido en LOSO (sets 4–10)."""
    return mask_sets(ws, LOSO_TEST_SETS)


def loso_folds(subjects: Sequence[str] = DEFAULT_SUBJECTS) -> Iterator[Tuple[int, str, List[str]]]:
    """Itera (fold, sujeto_prueba, sujetos_entrenamiento)."""
    subjects = list(subjects)
    for i, test_subject in enumerate(subjects):
        yield i, test_subject, [s for s in subjects if s != test_subject]


def fingerprint(ws: WindowSet) -> str:
    """Hash de (inicio, etiqueta, set) de cada ventana: verifica que se reconstruyen las mismas ventanas."""
    h = hashlib.sha256()
    for arr in (ws.start.astype(np.int64), ws.y.astype(np.int32), ws.set_id.astype(np.int16)):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:16]


def build_split_record(ws: WindowSet) -> Dict:
    """Índices de ventana (y fila de inicio en el CSV) de cada partición canónica."""
    def part(mask: np.ndarray) -> Dict:
        idx = np.flatnonzero(mask)
        return {"window_idx": idx.tolist(), "start_row": ws.start[idx].tolist(), "n": int(idx.size)}

    tr, te = intra_split(ws)
    record = {
        "subject": str(ws.subject[0]),
        "muscle": ws.muscle,
        "window_config": ws.config,
        "n_windows": len(ws),
        "fingerprint": fingerprint(ws),
        "class_counts": {int(c): int(n) for c, n in zip(*np.unique(ws.y, return_counts=True))},
        "partitions": {
            "intra_train": part(tr),
            "intra_test": part(te),
            "calib_test": part(mask_sets(ws, CALIB_TEST_SETS)),
        },
    }
    for k in CALIB_K:
        cal, _ = calib_split(ws, k)
        record["partitions"][f"calib_k{k}"] = part(cal)
    return record


def split_path(subject: str, muscle: str, tag: str = "steady") -> Path:
    return SPLITS_DIR / tag / f"{muscle}_{subject}.json"


def save_split_file(subject: str, muscle: str, tag: str = "steady", **window_kwargs) -> Path:
    ws = load_windows(subject, muscle, **window_kwargs)
    path = split_path(subject, muscle, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_split_record(ws), separators=(",", ":")), encoding="utf-8")
    return path


def load_split_file(subject: str, muscle: str, tag: str = "steady", ws: Optional[WindowSet] = None) -> Dict:
    """
    Lee el JSON de particiones. Si se entrega `ws`, verifica que las ventanas reconstruidas
    coinciden exactamente (misma huella); si no, la segmentación cambió y hay que regenerar.
    """
    record = json.loads(split_path(subject, muscle, tag).read_text(encoding="utf-8"))
    if ws is not None and fingerprint(ws) != record["fingerprint"]:
        raise RuntimeError(
            f"Las ventanas de {muscle}/{subject} no coinciden con {split_path(subject, muscle, tag)}. "
            "Regenere con `python -m src.splits`."
        )
    return record


def _main() -> None:
    ap = argparse.ArgumentParser(description="Genera los JSON de particiones canónicas por set.")
    ap.add_argument("--no-steady", action="store_true", help="Usar contracción completa (sin recorte).")
    ap.add_argument("--no-mvc", action="store_true", help="Usar señal en unidades del ADC filtrada (sin %%MVC).")
    ap.add_argument("--subjects", nargs="*", default=DEFAULT_SUBJECTS)
    args = ap.parse_args()

    tag = ("full" if args.no_steady else "steady") + ("_raw" if args.no_mvc else "")
    kwargs = dict(window_size=config.WINDOW_SIZE, stride=config.STRIDE,
                  steady_state_only=not args.no_steady, apply_mvc=not args.no_mvc)
    for muscle in MUSCLES:
        for s in args.subjects:
            p = save_split_file(s, muscle, tag=tag, **kwargs)
            rec = json.loads(p.read_text(encoding="utf-8"))
            parts = rec["partitions"]
            print(f"{muscle}/{s}: {rec['n_windows']:5d} ventanas | intra {parts['intra_train']['n']}/"
                  f"{parts['intra_test']['n']} | calib k1/k2/k3 {parts['calib_k1']['n']}/"
                  f"{parts['calib_k2']['n']}/{parts['calib_k3']['n']} | test {parts['calib_test']['n']} -> {p.name}")


if __name__ == "__main__":
    _main()
