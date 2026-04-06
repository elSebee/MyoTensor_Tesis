"""
=============================================================
 gui.py — Interfaz grafica PyQt5 | Dataset Collector
=============================================================
 Dos pantallas con QStackedWidget:

   Pantalla 0 — Monitoreo:
     Formulario del sujeto + grafico EMG + boton de inicio

   Pantalla 1 — Protocolo:
     Instruccion grande + GIF del gesto + barra de progreso
     + grafico EMG pequeño al pie

   Pantalla 2 — Completado:
     Resumen de la sesion + ruta del archivo guardado
=============================================================
"""

import os
import sys
import time

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QFont, QMovie, QIcon, QColor
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QStackedWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QPushButton, QProgressBar, QFrame, QMessageBox,
    QSizePolicy, QApplication,
)

from udp_receiver import DataBuffer

# ── Constantes de visualizacion ─────────────────────────────
FS_HZ = 1000
MONITOR_WINDOW_S = 5.0
PROTOCOL_WINDOW_S = 3.0
UPDATE_MS = 33  # ~30 FPS


# ── Estilos CSS ──────────────────────────────────────────────
MAIN_STYLE = """
QMainWindow, QWidget {
    background-color: #FDFDFD;
    font-family: "Inter", "Noto Sans", "Segoe UI", "Helvetica Neue", sans-serif;
}

QLabel {
    color: #1a1a2e;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 14px;
    background-color: #f8fafc;
    color: #1a1a2e;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #2563eb;
}

QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}

QPushButton#startButton {
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 16px 48px;
    font-size: 18px;
    font-weight: bold;
}

QPushButton#startButton:hover {
    background-color: #1d4ed8;
}

QPushButton#startButton:disabled {
    background-color: #94a3b8;
}

QPushButton#newSessionButton {
    background-color: #059669;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px 40px;
    font-size: 16px;
    font-weight: bold;
}

QPushButton#newSessionButton:hover {
    background-color: #047857;
}

QProgressBar {
    border: none;
    border-radius: 10px;
    background-color: #e2e8f0;
    text-align: center;
    height: 22px;
    font-size: 12px;
    color: #64748b;
}

QProgressBar::chunk {
    background-color: #2563eb;
    border-radius: 10px;
}

QFrame#separator {
    background-color: #e2e8f0;
    max-height: 1px;
}
"""


class MainWindow(QMainWindow):
    """Ventana principal del Dataset Collector."""

    def __init__(self, buf: DataBuffer, engine, writer, config: dict,
                 assets_dir: str):
        super().__init__()
        self.buf = buf
        self.engine = engine
        self.writer = writer
        self.config = config
        self.assets_dir = assets_dir

        self.setWindowTitle("MyoTensor — Recolector")
        self.setMinimumSize(900, 700)
        self.resize(1100, 800)
        self.setStyleSheet(MAIN_STYLE)

        # ── Stacked widget ───────────────────────────────────
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self._build_monitor_screen()    # index 0
        self._build_protocol_screen()   # index 1
        self._build_done_screen()       # index 2

        self.stack.setCurrentIndex(0)

        # ── Timer de actualizacion del grafico ────────────────
        self._plot_timer = QTimer()
        self._plot_timer.timeout.connect(self._update_plots)
        self._plot_timer.start(UPDATE_MS)
        self._t_start = time.time()
        self._current_state = "idle"   # para que _on_progress sepa la direccion de la barra

        # ── Conectar señales del engine ───────────────────────
        self.engine.state_changed.connect(self._on_state_changed)
        self.engine.progress.connect(self._on_progress)
        self.engine.finished.connect(self._on_finished)

        # ── GIF movie (para protocolo) ────────────────────────
        self._current_movie = None

    # ==========================================================
    #  PANTALLA 0 — MONITOREO
    # ==========================================================

    def _build_monitor_screen(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(16)

        # ── Titulo ────────────────────────────────────────────
        title = QLabel("MyoTensor — Recolector Cachipún")
        title.setFont(QFont("Inter", 20, QFont.Bold))
        title.setAlignment(Qt.AlignLeft)
        layout.addWidget(title)

        subtitle = QLabel("Completa los datos del sujeto y verifica la señal EMG antes de iniciar el protocolo.")
        subtitle.setFont(QFont("Inter", 11))
        subtitle.setStyleSheet("color: #64748b; margin-bottom: 8px;")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # ── Formulario del sujeto ─────────────────────────────
        form_frame = QFrame()
        form_frame.setStyleSheet("QFrame { background-color: #f8fafc; border-radius: 12px; padding: 16px; }")
        form_grid = QGridLayout(form_frame)
        form_grid.setHorizontalSpacing(16)
        form_grid.setVerticalSpacing(10)

        lbl_style = "font-size: 12px; font-weight: bold; color: #475569;"

        # Fila 1
        lbl_id = QLabel("ID Sujeto")
        lbl_id.setStyleSheet(lbl_style)
        self.input_id = QLineEdit("S01")
        self.input_id.setMaximumWidth(120)

        lbl_age = QLabel("Edad")
        lbl_age.setStyleSheet(lbl_style)
        self.input_age = QSpinBox()
        self.input_age.setRange(10, 90)
        self.input_age.setValue(23)
        self.input_age.setMaximumWidth(100)

        lbl_gender = QLabel("Género")
        lbl_gender.setStyleSheet(lbl_style)
        self.input_gender = QComboBox()
        self.input_gender.addItems(["Masculino", "Femenino"])
        self.input_gender.setMaximumWidth(140)

        lbl_lat = QLabel("Lateralidad")
        lbl_lat.setStyleSheet(lbl_style)
        self.input_laterality = QComboBox()
        self.input_laterality.addItems(["Diestro", "Zurdo"])
        self.input_laterality.setMaximumWidth(140)

        form_grid.addWidget(lbl_id, 0, 0)
        form_grid.addWidget(self.input_id, 1, 0)
        form_grid.addWidget(lbl_age, 0, 1)
        form_grid.addWidget(self.input_age, 1, 1)
        form_grid.addWidget(lbl_gender, 0, 2)
        form_grid.addWidget(self.input_gender, 1, 2)
        form_grid.addWidget(lbl_lat, 0, 3)
        form_grid.addWidget(self.input_laterality, 1, 3)

        # Fila 2
        lbl_circ = QLabel("Circunferencia antebrazo (cm)")
        lbl_circ.setStyleSheet(lbl_style)
        self.input_circumference = QDoubleSpinBox()
        self.input_circumference.setRange(10.0, 50.0)
        self.input_circumference.setValue(25.0)
        self.input_circumference.setDecimals(1)
        self.input_circumference.setSuffix(" cm")
        self.input_circumference.setMaximumWidth(140)

        lbl_muscle = QLabel("Músculo")
        lbl_muscle.setStyleSheet(lbl_style)
        self.input_muscle = QLineEdit("")
        self.input_muscle.setMaximumWidth(260)

        form_grid.addWidget(lbl_circ, 2, 0, 1, 2)
        form_grid.addWidget(self.input_circumference, 3, 0)
        form_grid.addWidget(lbl_muscle, 2, 2, 1, 2)
        form_grid.addWidget(self.input_muscle, 3, 2, 1, 2)

        layout.addWidget(form_frame)

        # ── Grafico EMG ──────────────────────────────────────
        self.monitor_plot = self._create_plot_widget(
            title="Señal EMG filtrada — últimos 5 segundos",
            window_s=MONITOR_WINDOW_S,
            height=280,
        )
        layout.addWidget(self.monitor_plot["widget"])

        # ── Status bar ─────────────────────────────────────
        status_layout = QHBoxLayout()
        self.lbl_status = QLabel("⏳ Esperando datos del ESP32...")
        self.lbl_status.setFont(QFont("Inter", 11))
        self.lbl_status.setStyleSheet("color: #64748b;")
        status_layout.addWidget(self.lbl_status)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        # ── Boton de inicio ──────────────────────────────────
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_start = QPushButton("▶  Iniciar Adquisición")
        self.btn_start.setObjectName("startButton")
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.clicked.connect(self._on_start_clicked)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.stack.addWidget(page)

    # ==========================================================
    #  PANTALLA 1 — PROTOCOLO
    # ==========================================================

    def _build_protocol_screen(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 32, 40, 20)
        layout.setSpacing(12)

        # ── Instruccion grande ─────────────────────────────────
        self.lbl_instruction = QLabel("PREPARANDO...")
        self.lbl_instruction.setFont(QFont("Inter", 52, QFont.Bold))
        self.lbl_instruction.setAlignment(Qt.AlignCenter)
        self.lbl_instruction.setStyleSheet("color: #1a1a2e;")
        self.lbl_instruction.setWordWrap(True)
        layout.addWidget(self.lbl_instruction)

        # ── Imagen / GIF del gesto ─────────────────────────────
        self.lbl_gif = QLabel()
        self.lbl_gif.setAlignment(Qt.AlignCenter)
        self.lbl_gif.setFixedHeight(300)
        self.lbl_gif.setStyleSheet("background: transparent;")
        layout.addWidget(self.lbl_gif)

        # ── Barra de progreso ──────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v")
        layout.addWidget(self.progress_bar)

        # ── Info de repeticion / gesto ─────────────────────────
        self.lbl_rep_info = QLabel("")
        self.lbl_rep_info.setFont(QFont("Inter", 14))
        self.lbl_rep_info.setAlignment(Qt.AlignCenter)
        self.lbl_rep_info.setStyleSheet("color: #64748b;")
        layout.addWidget(self.lbl_rep_info)

        # ── Separador ─────────────────────────────────────────
        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.HLine)
        layout.addWidget(sep)

        # ── Grafico EMG pequeño ───────────────────────────────
        self.protocol_plot = self._create_plot_widget(
            title="Señal EMG en tiempo real",
            window_s=PROTOCOL_WINDOW_S,
            height=140,
        )
        layout.addWidget(self.protocol_plot["widget"])

        self.stack.addWidget(page)

    # ==========================================================
    #  PANTALLA 2 — COMPLETADO
    # ==========================================================

    def _build_done_screen(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(60, 60, 60, 60)
        layout.setSpacing(20)

        layout.addStretch()

        lbl_check = QLabel("✓")
        lbl_check.setFont(QFont("Inter", 64))
        lbl_check.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl_check)

        self.lbl_done_title = QLabel("¡Sesión completada!")
        self.lbl_done_title.setFont(QFont("Inter", 32, QFont.Bold))
        self.lbl_done_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_done_title)

        self.lbl_done_info = QLabel("")
        self.lbl_done_info.setFont(QFont("Inter", 13))
        self.lbl_done_info.setAlignment(Qt.AlignCenter)
        self.lbl_done_info.setStyleSheet("color: #475569;")
        self.lbl_done_info.setWordWrap(True)
        layout.addWidget(self.lbl_done_info)

        layout.addStretch()

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_new_session = QPushButton("Nueva sesión")
        self.btn_new_session.setObjectName("newSessionButton")
        self.btn_new_session.setCursor(Qt.PointingHandCursor)
        self.btn_new_session.clicked.connect(self._on_new_session)
        btn_layout.addWidget(self.btn_new_session)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.stack.addWidget(page)

    # ==========================================================
    #  HELPERS
    # ==========================================================

    def _create_plot_widget(self, title: str, window_s: float, height: int) -> dict:
        """Crea un PlotWidget de pyqtgraph con estilo claro."""
        pw = pg.PlotWidget()
        pw.setBackground("#FDFDFD")
        pw.setFixedHeight(height)
        pw.setTitle(f"<span style='font-size:10pt;color:#475569'>{title}</span>")
        pw.setLabel("left", "amplitud", color="#94a3b8")
        pw.setLabel("bottom", "tiempo (s)", color="#94a3b8")
        pw.showGrid(x=True, y=True, alpha=0.15)
        pw.setMenuEnabled(False)
        pw.getAxis("bottom").setPen(pg.mkPen("#cbd5e1"))
        pw.getAxis("left").setPen(pg.mkPen("#cbd5e1"))

        pen = pg.mkPen("#2563eb", width=1.5)
        curve = pw.plot(pen=pen)

        n = int(FS_HZ * window_s)
        t = np.linspace(-window_s, 0, n)

        return {"widget": pw, "curve": curve, "t": t, "n": n, "window_s": window_s}

    def _update_plots(self):
        """Actualiza los graficos EMG a 30 FPS."""
        filtered, = self.buf.snapshot("filtered")

        # Monitor plot (pantalla 0)
        if self.stack.currentIndex() == 0:
            n = self.monitor_plot["n"]
            data = filtered[-n:]
            if len(data) == n:
                self.monitor_plot["curve"].setData(self.monitor_plot["t"], data)

            # Actualizar status
            elapsed = time.time() - self._t_start
            rate = self.buf.total_samples / elapsed if elapsed > 1.0 else 0
            status_icon = "🟢" if rate > 900 else "🟡" if rate > 0 else "🔴"
            self.lbl_status.setText(
                f"{status_icon}  Rate: {rate:.0f} Hz   |   "
                f"Muestras: {self.buf.total_samples}   |   "
                f"Drops: {self.buf.drops}"
            )

        # Protocol plot (pantalla 1)
        elif self.stack.currentIndex() == 1:
            n = self.protocol_plot["n"]
            data = filtered[-n:]
            if len(data) == n:
                self.protocol_plot["curve"].setData(self.protocol_plot["t"], data)

    # ==========================================================
    #  ACCIONES DEL USUARIO
    # ==========================================================

    def _get_subject_info(self) -> dict:
        """Lee el formulario y retorna los datos del sujeto."""
        gender_map = {"Masculino": "m", "Femenino": "f"}
        lat_map = {"Diestro": "r", "Zurdo": "l"}

        return {
            "subject_id": self.input_id.text().strip(),
            "age": self.input_age.value(),
            "gender": gender_map.get(self.input_gender.currentText(), "m"),
            "laterality": lat_map.get(self.input_laterality.currentText(), "r"),
            "forearm_circumference_cm": self.input_circumference.value(),
            "muscle": self.input_muscle.text().strip(),
        }

    def _on_start_clicked(self):
        """Validar formulario e iniciar protocolo."""
        info = self._get_subject_info()

        if not info["subject_id"]:
            QMessageBox.warning(self, "Dato faltante",
                                "Ingresa un ID de sujeto antes de iniciar.")
            return

        if not info["muscle"]:
            QMessageBox.warning(self, "Dato faltante",
                                "Ingresa el nombre del músculo.")
            return

        # Configurar writer con datos del sujeto
        self.writer.subject_info = info
        self.writer.reset()

        # Cambiar a pantalla de protocolo
        self.stack.setCurrentIndex(1)
        self.engine.start()

    def _on_new_session(self):
        """Volver a pantalla de monitoreo para nueva sesion."""
        self.stack.setCurrentIndex(0)

    # ==========================================================
    #  SEÑALES DEL ENGINE
    # ==========================================================

    def _on_state_changed(self, state_name: str, info: dict):
        """Actualizar la UI segun el estado del protocolo."""
        if state_name == "countdown":
            self._current_state = "countdown"
            cd = info.get("countdown", 0)
            self.lbl_instruction.setText(str(cd))
            self.lbl_instruction.setFont(QFont("Inter", 120, QFont.Bold))
            self.lbl_instruction.setStyleSheet("color: #2563eb;")
            self.lbl_gif.clear()
            self.lbl_rep_info.setText("Preparando...")
            # NO tocar la barra aqui — _on_progress la maneja sin parpadeo
            self._set_protocol_bg("#FDFDFD")

        elif state_name == "gesture":
            self._current_state = "gesture"
            gesture = info["gesture"]
            label = gesture["label"]
            rep = info["repetition"]
            total_reps = info["total_reps"]
            g_idx = info["gesture_index"] + 1
            g_total = info["total_gestures"]

            self.lbl_instruction.setText(label)
            self.lbl_instruction.setFont(QFont("Inter", 48, QFont.Bold))
            self.lbl_instruction.setStyleSheet("color: #1a1a2e;")
            self.lbl_rep_info.setText(
                f"Repetición {rep} / {total_reps}     ·     "
                f"Gesto {g_idx} / {g_total}  ({gesture['name']})"
            )
            self._set_protocol_bg("#FDFDFD")
            self._load_gif(gesture.get("image", ""))

        elif state_name == "rest":
            self._current_state = "rest"
            rep = info["repetition"]
            total_reps = info["total_reps"]

            self.lbl_instruction.setText("DESCANSA")
            self.lbl_instruction.setFont(QFont("Inter", 52, QFont.Bold))
            self.lbl_instruction.setStyleSheet("color: #64748b;")
            self.lbl_rep_info.setText(f"Repetición {rep} / {total_reps}")
            self._set_protocol_bg("#f1f5f9")
            self._stop_gif()

    def _on_progress(self, elapsed: float, total: float):
        """Actualizar barra de progreso.

        - Durante countdown: barra DESCIENDE (llena → vacia = 3s → 0s)
        - Durante gesture/rest: barra ASCIENDE (vacia → llena)
        """
        if total <= 0:
            return
        remaining = max(0.0, total - elapsed)

        if self._current_state == "countdown":
            # Barra descendente: empieza llena, termina vacia
            pct = remaining / total
            self.progress_bar.setFormat(f"{int(remaining + 0.99)}s")
        else:
            # Barra ascendente: empieza vacia, termina llena
            pct = min(elapsed / total, 1.0)
            self.progress_bar.setFormat(f"{remaining:.1f}s")

        self.progress_bar.setValue(int(pct * 1000))

    def _on_finished(self):
        """Protocolo completado — guardar y mostrar resumen."""
        csv_path, meta_path = self.writer.save()

        n = len(self.writer.samples)
        dur = n / 1000.0 if n else 0
        subject = self.writer.subject_info.get("subject_id", "?")

        info_text = (
            f"Sujeto: {subject}\n"
            f"Muestras grabadas: {n:,}\n"
            f"Duración: {dur:.1f} segundos\n\n"
        )
        if csv_path:
            info_text += f"CSV: {os.path.abspath(csv_path)}\n"
        if meta_path:
            info_text += f"Metadata: {os.path.abspath(meta_path)}"

        self.lbl_done_info.setText(info_text)
        self.stack.setCurrentIndex(2)

    # ── Helpers de UI ──────────────────────────────────────────

    def _set_protocol_bg(self, color: str):
        """Cambiar color de fondo de la pantalla de protocolo."""
        page = self.stack.widget(1)
        page.setStyleSheet(f"QWidget {{ background-color: {color}; }}")

    def _load_gif(self, image_name: str):
        """Cargar un GIF/imagen en el label del protocolo."""
        self._stop_gif()

        if not image_name:
            self.lbl_gif.setText("🖐")
            self.lbl_gif.setFont(QFont("Inter", 80))
            return

        path = os.path.join(self.assets_dir, image_name)
        if not os.path.exists(path):
            # Placeholder si no existe el archivo
            name = image_name.replace(".gif", "").replace(".png", "")
            self.lbl_gif.setText(f"[ {name} ]")
            self.lbl_gif.setFont(QFont("Inter", 24))
            self.lbl_gif.setStyleSheet("color: #94a3b8;")
            return

        movie = QMovie(path)
        movie.setScaledSize(QSize(300, 300))
        self.lbl_gif.setMovie(movie)
        movie.start()
        self._current_movie = movie

    def _stop_gif(self):
        """Detener animacion GIF si existe."""
        if self._current_movie:
            self._current_movie.stop()
            self._current_movie = None
        self.lbl_gif.clear()

    # ── Cierre seguro ──────────────────────────────────────────

    def closeEvent(self, event):
        """Confirmar cierre si hay una adquisicion en curso."""
        if self.engine.recording:
            reply = QMessageBox.question(
                self, "Adquisición en curso",
                "Hay una adquisición en curso. ¿Seguro que quieres cerrar?\n"
                "Se perderán los datos no guardados.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return

        event.accept()
