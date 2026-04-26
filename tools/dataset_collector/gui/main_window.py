"""
gui/main_window.py — Orquestador principal (QMainWindow).

Responsabilidades:
  - Contener el QStackedWidget con las 4 pantallas
  - Navegar entre pantallas segun engine.state_changed
  - Gestionar el timer de actualizacion de graficos (30 FPS)
  - Conectar calibrator.finished → writer.set_calibration
  - Confirmar cierre si hay adquisicion en curso

Flujo de pantallas (manejado por engine.state_changed):
  0: MonitorScreen      → usuario presiona "Iniciar Adquisicion"
  1: CalibrationScreen  → calibration_noise, calibration_mvc
  2: ProtocolScreen     → gesture_prep, gesture_active, rest
  3: DoneScreen         → engine.finished

La navegacion es automatica — el engine dirige todo.
"""

import time

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMainWindow, QStackedWidget, QMessageBox

from .styles import MAIN_STYLE, UPDATE_MS
from .screen_monitor     import MonitorScreen
from .screen_calibration import CalibrationScreen
from .screen_protocol    import ProtocolScreen
from .screen_done        import DoneScreen

# Estados que corresponden a cada pantalla
_CAL_STATES      = {"calibration_noise", "calibration_mvc"}
_PROTOCOL_STATES = {"gesture_prep", "gesture_active", "rest"}


class MainWindow(QMainWindow):
    """Ventana principal — orquestador del Dataset Collector."""

    IDX_MONITOR     = 0
    IDX_CALIBRATION = 1
    IDX_PROTOCOL    = 2
    IDX_DONE        = 3

    def __init__(self, buf, engine, writer, calibrator, config: dict, assets_dir: str):
        super().__init__()
        self.buf        = buf
        self.engine     = engine
        self.writer     = writer
        self.calibrator = calibrator

        self.setWindowTitle("MyoTensor — Recolector")
        self.setMinimumSize(900, 700)
        self.resize(1100, 800)
        self.setStyleSheet(MAIN_STYLE)

        # ── 4 pantallas ───────────────────────────────────────
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.screen_monitor     = MonitorScreen()
        self.screen_calibration = CalibrationScreen(engine, calibrator)
        self.screen_protocol    = ProtocolScreen(assets_dir)
        self.screen_done        = DoneScreen()

        self.stack.addWidget(self.screen_monitor)      # 0
        self.stack.addWidget(self.screen_calibration)  # 1
        self.stack.addWidget(self.screen_protocol)     # 2
        self.stack.addWidget(self.screen_done)         # 3

        self.stack.setCurrentIndex(self.IDX_MONITOR)

        # ── Señales de navegacion ─────────────────────────────
        self.screen_monitor.start_requested.connect(self._on_start_requested)
        self.screen_done.new_session_requested.connect(self._on_new_session)

        # ── Engine: navegar + actualizar ProtocolScreen ───────
        engine.state_changed.connect(self._on_engine_state_changed)
        engine.state_changed.connect(self.screen_protocol.on_state_changed)
        engine.progress.connect(self.screen_protocol.on_progress)
        engine.finished.connect(self._on_protocol_finished)

        # ── Calibrator: escribir resultado en writer ──────────
        calibrator.finished.connect(self._on_calibration_result)

        # ── Timer de graficos 30 FPS ──────────────────────────
        self._plot_timer = QTimer()
        self._plot_timer.timeout.connect(self._update_plots)
        self._plot_timer.start(UPDATE_MS)
        self._t_start = time.time()

    # ── Navegacion ─────────────────────────────────────────────

    def _on_start_requested(self, subject_info: dict):
        """El usuario presiono Iniciar — limpiar y arrancar engine."""
        self.writer.subject_info = subject_info
        self.writer.reset()
        self.calibrator.reset()
        self.screen_calibration.reset()
        self.engine.start()
        # La pantalla cambiara via _on_engine_state_changed al llegar
        # el primer estado calibration_noise

    def _on_engine_state_changed(self, state_name: str, info: dict):
        """Navegar entre CalibrationScreen y ProtocolScreen segun el estado."""
        if state_name in _CAL_STATES:
            if self.stack.currentIndex() != self.IDX_CALIBRATION:
                self.stack.setCurrentIndex(self.IDX_CALIBRATION)

        elif state_name in _PROTOCOL_STATES:
            if self.stack.currentIndex() != self.IDX_PROTOCOL:
                self.stack.setCurrentIndex(self.IDX_PROTOCOL)

    def _on_calibration_result(self, result: dict):
        """Guardar resultado de calibracion en el DataWriter."""
        self.writer.set_calibration(result)

    def _on_protocol_finished(self):
        """Todos los Sets completados — guardar y mostrar resumen."""
        csv_path, meta_path = self.writer.save()
        n       = len(self.writer.samples)
        subject = self.writer.subject_info.get("subject_id", "?")
        self.screen_done.set_result(n, csv_path, meta_path, subject)
        self.stack.setCurrentIndex(self.IDX_DONE)

    def _on_new_session(self):
        self.stack.setCurrentIndex(self.IDX_MONITOR)

    # ── Actualizacion de graficos ──────────────────────────────

    def _update_plots(self):
        idx = self.stack.currentIndex()
        n   = self.buf.total_samples   # para skip de frames sin datos nuevos
        if idx == self.IDX_MONITOR:
            filtered, = self.buf.snapshot("filtered")
            self.screen_monitor.update_plot(filtered, n)
            elapsed = time.time() - self._t_start
            rate    = self.buf.total_samples / elapsed if elapsed > 1.0 else 0.0
            self.screen_monitor.update_status(rate, self.buf.total_samples, self.buf.drops)
        elif idx == self.IDX_CALIBRATION:
            filtered, = self.buf.snapshot("filtered")
            self.screen_calibration.update_plot(filtered, n)
        elif idx == self.IDX_PROTOCOL:
            filtered, = self.buf.snapshot("filtered")
            self.screen_protocol.update_plot(filtered, n)

    # ── Cierre seguro ──────────────────────────────────────────

    def closeEvent(self, event):
        is_busy = self.engine.recording or self.calibrator.active
        if is_busy:
            reply = QMessageBox.question(
                self,
                "Operación en curso",
                "Hay una adquisición o calibración en curso.\n"
                "¿Seguro que quieres cerrar? Se perderán los datos no guardados.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
        event.accept()
