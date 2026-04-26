"""
gui/screen_calibration.py — Pantalla 1: Calibración MVC + ruido basal.

Con el nuevo ProtocolEngine, la temporización viene del engine.
Esta pantalla solo reacciona a señales:

  engine.state_changed → muestra instruccion y fase correcta
  engine.progress      → actualiza barra de progreso
  calibrator.finished  → muestra badge con resultados

No emite señales de navegacion — el MainWindow navega basado
en engine.state_changed (cuando llega gesture_prep cambia de pantalla).
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QProgressBar, QFrame,
)

from .styles import COLOR_TEXT, COLOR_TEXT_MUTED, COLOR_CALIBRATING
from .plot_widget import create_plot_widget, update_plot


class CalibrationScreen(QWidget):
    """Pantalla de calibracion: reacciona a señales del engine y calibrator."""

    def __init__(self, engine, calibrator, parent=None):
        super().__init__(parent)
        self._build_ui()

        # Conectar al engine para estado y progreso
        engine.state_changed.connect(self._on_state_changed)
        engine.progress.connect(self._on_progress)

        # Conectar al calibrator para el badge de resultado
        calibrator.finished.connect(self._on_calibration_result)

        self._in_cal_phase = False

    # ── Construccion de UI ─────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 32, 40, 20)
        layout.setSpacing(14)

        self.lbl_phase = QLabel("CALIBRACIÓN")
        self.lbl_phase.setFont(QFont("Inter", 13, QFont.Bold))
        self.lbl_phase.setAlignment(Qt.AlignCenter)
        self.lbl_phase.setStyleSheet(f"color: {COLOR_CALIBRATING}; letter-spacing: 3px;")
        layout.addWidget(self.lbl_phase)

        self.lbl_instruction = QLabel("Iniciando calibración...")
        self.lbl_instruction.setFont(QFont("Inter", 42, QFont.Bold))
        self.lbl_instruction.setAlignment(Qt.AlignCenter)
        self.lbl_instruction.setStyleSheet(f"color: {COLOR_TEXT};")
        self.lbl_instruction.setWordWrap(True)
        layout.addWidget(self.lbl_instruction)

        self.lbl_description = QLabel("")
        self.lbl_description.setFont(QFont("Inter", 14))
        self.lbl_description.setAlignment(Qt.AlignCenter)
        self.lbl_description.setStyleSheet(f"color: {COLOR_TEXT_MUTED};")
        self.lbl_description.setWordWrap(True)
        layout.addWidget(self.lbl_description)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("calibBar")
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("")
        layout.addWidget(self.progress_bar)

        self.lbl_result = QLabel("")
        self.lbl_result.setFont(QFont("Inter", 14))
        self.lbl_result.setAlignment(Qt.AlignCenter)
        self.lbl_result.setStyleSheet(
            "color: #059669; background: #f0fdf4; border-radius: 10px; padding: 10px;"
        )
        self.lbl_result.setVisible(False)
        layout.addWidget(self.lbl_result)

        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.HLine)
        layout.addWidget(sep)

        from .styles import PROTOCOL_WINDOW_S
        self._plot = create_plot_widget(
            title="Señal EMG en tiempo real",
            window_s=PROTOCOL_WINDOW_S,
            height=130,
        )
        layout.addWidget(self._plot["widget"])

    # ── Slots publicos ─────────────────────────────────────────

    def update_plot(self, data, total_samples: int = 0) -> None:
        update_plot(self._plot, data, total_samples)

    def reset(self):
        """Preparar pantalla para una nueva sesion."""
        self.lbl_result.setVisible(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")
        self._in_cal_phase = False

    # ── Slots privados — conectados en __init__ ────────────────

    def _on_state_changed(self, state_name: str, info: dict):
        """Actualizar UI solo durante fases de calibracion."""
        if state_name == "calibration_noise":
            self._in_cal_phase = True
            self.lbl_phase.setText("FASE 1 / 2  →  RUIDO BASAL")
            self.lbl_instruction.setText("REPOSO ABSOLUTO")
            self.lbl_instruction.setStyleSheet(f"color: {COLOR_CALIBRATING};")
            self.lbl_description.setText(
                "Mantén el brazo completamente relajado. No te muevas."
            )
            self.progress_bar.setObjectName("calibBar")
            self.progress_bar.setValue(0)
            self._refresh_style()

        elif state_name == "calibration_mvc":
            self._in_cal_phase = True
            self.lbl_phase.setText("FASE 2 / 2  →  CONTRACCIÓN MÁXIMA")
            self.lbl_instruction.setText("¡CONTRAE AL MÁXIMO!")
            self.lbl_instruction.setStyleSheet("color: #dc2626;")
            self.lbl_description.setText(
                "Aplica la mayor fuerza posible y mantenla."
            )
            self.progress_bar.setObjectName("mvcBar")
            self.progress_bar.setValue(0)
            self._refresh_style()

        else:
            self._in_cal_phase = False

    def _on_progress(self, elapsed: float, total: float):
        """Actualizar barra solo durante fases de calibracion."""
        if not self._in_cal_phase or total <= 0:
            return
        pct = min(elapsed / total, 1.0)
        remaining = max(0.0, total - elapsed)
        self.progress_bar.setValue(int(pct * 1000))
        self.progress_bar.setFormat(f"{remaining:.1f}s")

    def _on_calibration_result(self, result: dict):
        mvc = result.get("mvc_voltage_v", 0.0)
        thr = result.get("onset_threshold_v", 0.0)
        std = result.get("noise_std", 0.0)
        self.lbl_phase.setText("✓  CALIBRACIÓN COMPLETADA")
        self.lbl_instruction.setText("¡Listo!")
        self.lbl_instruction.setStyleSheet("color: #059669;")
        self.lbl_description.setText("Iniciando protocolo de adquisición...")
        self.lbl_result.setText(
            f"✓  MVC: {mvc:.3f} V   │   Umbral onset: {thr:.3f} V   │   σ ruido: {std:.4f} V"
        )
        self.lbl_result.setVisible(True)
        self.progress_bar.setValue(1000)

    def _refresh_style(self):
        self.progress_bar.style().unpolish(self.progress_bar)
        self.progress_bar.style().polish(self.progress_bar)
