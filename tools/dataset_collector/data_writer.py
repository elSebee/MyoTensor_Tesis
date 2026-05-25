"""
=============================================================
 data_writer.py — Acumula muestras y guarda al final
=============================================================
 Registra un callback en UDPReceiver. Solo acumula muestras
 cuando el ProtocolEngine esta en estado GESTURE o REST
 (recording == True).

 Al recibir la señal finished del engine:
   1. Vuelca todos los datos a un CSV
   2. Guarda session_metadata.json (incluye calibracion)
   3. Ejecuta run_postprocess para agregar restimulus y valid_flag

 Columnas del CSV (protocolo binario):
   timestamp_us, filtered, emg_norm,
   stimulus, set_id, repetition_id

 NOTA: raw, centered y voltage_v ya no llegan del firmware
 (protocolo binario optimizado). emg_norm se calcula en post.
=============================================================
"""

import json
import os
from datetime import datetime

from preprocess import run_preprocess


class DataWriter:
    """Acumula muestras EMG etiquetadas y las guarda al finalizar."""

    def __init__(self, engine, config: dict, subject_info: dict, output_dir: str):
        self.engine      = engine
        self.config      = config
        self.subject_info = subject_info
        self.output_dir  = output_dir

        # Resultado de calibracion (se setea via set_calibration antes de grabar)
        self.calibration: dict = {}

        # Acumulador en RAM
        self.samples: list = []
        self.session_start = None

    # ── API publica ────────────────────────────────────────────

    def set_calibration(self, result: dict):
        """Recibir resultado de calibracion desde el Calibrator.

        Llamar antes de que empiece el protocolo de adquisicion.
        """
        self.calibration = result or {}
        mvc = self.calibration.get("mvc_voltage_v", 1.0)
        thr = self.calibration.get("onset_threshold_v", 0.0)
        print(f"[WRITER] Calibracion recibida — MVC: {mvc:.4f}V | Umbral: {thr:.4f}V")

    def on_batch(self, timestamps, filtered_arr):
        """Callback invocado desde el hilo UDP por cada batch de muestras.

        Recibe arrays numpy con N muestras (tipicamente N=50).
        Solo registra si el engine esta grabando (GESTURE o REST).
        """
        if not self.engine.recording:
            return

        if self.session_start is None:
            self.session_start = datetime.now()

        mvc_v = self.calibration.get("mvc_voltage_v", 0.0)

        for ts, filtered in zip(timestamps, filtered_arr):
            # emg_norm: señal filtrada normalizada por el pico MVC.
            # Si no hay calibracion, emg_norm == filtered (sin escala).
            emg_norm = float(filtered) / mvc_v if mvc_v > 1e-6 else float(filtered)

            self.samples.append((
                int(ts),
                float(filtered),
                emg_norm,
                self.engine.stimulus,
                self.engine.repetition_id,
            ))

    def save(self) -> tuple:
        """Volcar datos a disco. Retorna (csv_path, meta_path).

        Llamar desde el hilo principal (señal finished del engine).
        """
        if not self.samples:
            print("[WARN] No hay muestras para guardar.")
            return None, None

        subject_id = self.subject_info.get("subject_id", "S00")
        subdir     = os.path.join(self.output_dir, subject_id)
        os.makedirs(subdir, exist_ok=True)

        ts        = self.session_start.strftime("%Y%m%d_%H%M%S")
        csv_path  = os.path.join(subdir, f"session_{ts}.csv")
        meta_path = os.path.join(subdir, f"session_{ts}_metadata.json")

        # ── Escribir CSV ────────────────────────────────────
        header = "timestamp_us,filtered,emg_norm,stimulus,repetition_id\n"
        with open(csv_path, "w") as f:
            f.write(header)
            for s in self.samples:
                f.write(
                    f"{s[0]},{s[1]:.4f},{s[2]:.6f},{s[3]},{s[4]}\n"
                )

        n = len(self.samples)
        print(f"[OK] CSV guardado: {csv_path} ({n} muestras, {n/1000:.1f}s)")

        # ── Escribir metadata ─────────────────────────────────
        onset_cfg = self.config.get("onset", {})
        meta = {
            **self.subject_info,
            "date":               self.session_start.isoformat(),
            "fs_hz":              1000,
            "total_samples":      n,
            "duration_s":         round(n / 1000.0, 2),
            "gestures":           [g["name"] for g in self.engine._base_gestures],
            "gesture_ids":        {g["name"]: g["id"] for g in self.engine._base_gestures},
            "n_sets":             self.config["protocol"].get("n_sets", 10),
            "gesture_duration_s": self.config["protocol"]["gesture_duration_s"],
            "rest_duration_s":    self.config["protocol"]["rest_duration_s"],
            "prep_duration_s":    self.config["protocol"].get("prep_duration_s", 2.0),
            "randomized_order":   True,
            "calibration":        self.calibration,
            "onset_config":       {
                "rms_window_ms":    onset_cfg.get("rms_window_ms", 100),
                "threshold_factor": onset_cfg.get("threshold_factor", 3.0),
                "min_active_ms":    onset_cfg.get("min_active_ms", 200),
            },
            "csv_file": os.path.basename(csv_path),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"[OK] Metadata guardado: {meta_path}")

        # ── Pre-proceso: restimulus dinamico + valid_flag ────
        run_preprocess(csv_path, meta_path)

        return csv_path, meta_path

    def reset(self):
        """Limpiar para una nueva sesion."""
        self.samples.clear()
        self.session_start = None
        self.calibration   = {}
