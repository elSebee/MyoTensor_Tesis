"""
=============================================================
 calibrator.py — Acumulador pasivo de datos de calibracion
=============================================================
 Con el nuevo ProtocolEngine, la temporización de las fases
 de calibración la gestiona el engine. El Calibrator solo:

   1. Acumula muestras en on_batch() leyendo engine.cal_phase
   2. Calcula noise stats y MVC peak cuando el engine emite
      calibration_done (transicion MVC → primer GESTURE_PREP)
   3. Emite finished(dict) con el resultado de calibracion

 Propiedades thread-safe (GIL safe):
   Ninguna — solo acumula en listas via on_batch()
   (el engine.cal_phase es la propiedad thread-safe del engine)

 Señales:
   finished(object) — calibration_result dict
=============================================================
"""

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal


class Calibrator(QObject):
    """Acumulador de datos para las fases de calibracion del engine."""

    finished = pyqtSignal(object)  # calibration_result dict

    def __init__(self, engine, config: dict):
        super().__init__()
        self._engine = engine
        onset_cfg    = config.get("onset", {})
        cal_cfg      = config.get("calibration", {})

        self._threshold_factor = float(onset_cfg.get("threshold_factor", 3.0))
        self._rms_window_ms    = int(onset_cfg.get("rms_window_ms", 100))
        self._mvc_dur          = float(cal_cfg.get("mvc_duration_s", 3.0))

        self._noise_samples: list[float] = []
        self._mvc_samples:   list[float] = []

        # El engine avisa cuando termina la calibracion
        engine.calibration_done.connect(self._finalize)

    # ── API publica ────────────────────────────────────────────

    def on_batch(self, timestamps, filtered_arr):
        """Callback desde hilo UDP. Recibe arrays numpy con N muestras."""
        cal_phase = self._engine.cal_phase
        if cal_phase is None:
            return
        for val in filtered_arr:
            if cal_phase == "noise":
                self._noise_samples.append(float(val))
            elif cal_phase == "mvc":
                self._mvc_samples.append(float(val))

    @property
    def active(self) -> bool:
        """True si hay una calibracion en curso (engine en fase cal)."""
        return self._engine.cal_phase is not None

    def reset(self):
        """Limpiar datos para una nueva sesion."""
        self._noise_samples.clear()
        self._mvc_samples.clear()

    # ── Computacion ────────────────────────────────────────────

    def _finalize(self):
        """Llamado por engine.calibration_done. Calcula y emite resultado."""
        noise = np.array(self._noise_samples, dtype=np.float64)
        mvc   = np.array(self._mvc_samples,   dtype=np.float64)

        noise_mean = float(np.mean(noise)) if len(noise) > 0 else 0.0
        noise_std  = float(np.std(noise))  if len(noise) > 0 else 0.01
        onset_threshold_v = noise_mean + self._threshold_factor * noise_std

        if len(mvc) > 0:
            fs_approx = len(mvc) / self._mvc_dur
            win_size  = max(1, int(self._rms_window_ms * fs_approx / 1000))
            step      = max(1, win_size // 2)
            rms_vals  = [
                float(np.sqrt(np.mean(mvc[i : i + win_size] ** 2)))
                for i in range(0, len(mvc) - win_size, step)
            ]
            mvc_voltage_v = float(np.percentile(rms_vals, 99)) if rms_vals else float(np.max(np.abs(mvc)))
        else:
            mvc_voltage_v = 1.0

        if mvc_voltage_v < 1e-6:
            mvc_voltage_v = 1.0

        result = {
            "noise_mean":        noise_mean,
            "noise_std":         noise_std,
            "onset_threshold_v": onset_threshold_v,
            "mvc_voltage_v":     mvc_voltage_v,
            "noise_samples":     int(len(noise)),
            "mvc_samples":       int(len(mvc)),
        }

        print(f"[CAL] noise: mean={noise_mean:.4f}V  std={noise_std:.4f}V")
        print(f"[CAL] umbral onset: {onset_threshold_v:.4f}V")
        print(f"[CAL] MVC peak RMS: {mvc_voltage_v:.4f}V")

        self.finished.emit(result)
