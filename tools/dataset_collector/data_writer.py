"""
=============================================================
 data_writer.py — Acumula muestras en RAM y guarda al final
=============================================================
 Registra un callback en UDPReceiver. Solo acumula muestras
 cuando el ProtocolEngine esta en estado GESTURE o REST
 (recording == True).

 Al recibir la señal finished del engine:
   1. Vuelca todos los datos a un CSV
   2. Guarda session_metadata.json
   3. Ejecuta postprocess para agregar restimulus
=============================================================
"""

import json
import os
from datetime import datetime

from postprocess import add_restimulus


class DataWriter:
    """Acumula muestras EMG etiquetadas y las guarda al finalizar."""

    def __init__(self, engine, config: dict, subject_info: dict, output_dir: str):
        self.engine = engine
        self.config = config
        self.subject_info = subject_info
        self.output_dir = output_dir

        # Acumulador en RAM (~13MB para sesion completa — sin problema)
        self.samples = []
        self.session_start = None

    def on_sample(self, sample: dict):
        """Callback invocado desde el hilo UDP por cada muestra.

        Solo registra si el engine esta grabando (GESTURE o REST).
        Lee engine.recording, engine.stimulus, engine.repetition
        que son atributos simples protegidos por el GIL de CPython.
        """
        if not self.engine.recording:
            return

        if self.session_start is None:
            self.session_start = datetime.now()

        self.samples.append((
            int(sample["timestamp_us"]),
            int(sample["raw"]),
            sample["centered"],
            sample["filtered"],
            sample["voltage_v"],
            self.engine.stimulus,
            self.engine.repetition,
        ))

    def save(self) -> tuple:
        """Volcar datos a disco. Retorna (csv_path, meta_path).

        Llamar desde el hilo principal (señal finished del engine).
        """
        if not self.samples:
            print("[WARN] No hay muestras para guardar.")
            return None, None

        subject_id = self.subject_info.get("subject_id", "S00")
        subdir = os.path.join(self.output_dir, subject_id)
        os.makedirs(subdir, exist_ok=True)

        ts = self.session_start.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(subdir, f"session_{ts}.csv")
        meta_path = os.path.join(subdir, f"session_{ts}_metadata.json")

        # ── Escribir CSV ─────────────────────────────────────
        header = "timestamp_us,raw,centered,filtered,voltage_v,stimulus,repetition\n"
        with open(csv_path, "w") as f:
            f.write(header)
            for s in self.samples:
                f.write(f"{s[0]},{s[1]},{s[2]:.2f},{s[3]:.2f},"
                        f"{s[4]:.4f},{s[5]},{s[6]}\n")

        n = len(self.samples)
        print(f"[OK] CSV guardado: {csv_path} ({n} muestras, {n/1000:.1f}s)")

        # ── Escribir metadata ────────────────────────────────
        meta = {
            **self.subject_info,
            "date": self.session_start.isoformat(),
            "fs_hz": 1000,
            "total_samples": n,
            "duration_s": round(n / 1000.0, 2),
            "gestures": [g["name"] for g in self.config["gestures"]],
            "gesture_ids": {g["name"]: g["id"] for g in self.config["gestures"]},
            "repetitions": self.config["protocol"]["repetitions"],
            "gesture_duration_s": self.config["protocol"]["gesture_duration_s"],
            "rest_duration_s": self.config["protocol"]["rest_duration_s"],
            "reaction_time_offset_ms": self.config["protocol"]["reaction_time_ms"],
            "csv_file": os.path.basename(csv_path),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"[OK] Metadata guardado: {meta_path}")

        # ── Post-proceso: agregar restimulus ──────────────────
        rt = self.config["protocol"]["reaction_time_ms"]
        add_restimulus(csv_path, reaction_time_ms=rt)

        return csv_path, meta_path

    def reset(self):
        """Limpiar para una nueva sesion."""
        self.samples.clear()
        self.session_start = None

