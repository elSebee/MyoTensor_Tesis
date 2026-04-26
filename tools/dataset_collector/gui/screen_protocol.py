"""
gui/screen_protocol.py — Pantalla 2: Protocolo de adquisicion.

Muestra la instruccion del gesto activo, el GIF de referencia,
la barra de progreso del tiempo restante y el grafico EMG en vivo.

Recibe las señales del ProtocolEngine via slots publicos
(on_state_changed, on_progress) que son conectados por MainWindow.
No tiene referencia directa al engine — solo reacciona a señales.
"""

import os

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QMovie
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QProgressBar, QFrame,
)

from .styles import (
    COLOR_TEXT, COLOR_TEXT_MUTED, COLOR_PRIMARY, COLOR_BG, COLOR_BG_MUTED,
    PROTOCOL_WINDOW_S,
)
from .plot_widget import create_plot_widget, update_plot


class ProtocolScreen(QWidget):
    """Pantalla de ejecucion del protocolo de gestos."""

    def __init__(self, assets_dir: str, parent=None):
        super().__init__(parent)
        self._assets_dir    = assets_dir
        self._current_movie = None
        self._current_state = "idle"

        self._build_ui()

    # ── Construccion de UI ─────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 32, 40, 20)
        layout.setSpacing(12)

        # ── Instruccion grande ────────────────────────────────
        self.lbl_instruction = QLabel("PREPARANDO...")
        self.lbl_instruction.setFont(QFont("Inter", 52, QFont.Bold))
        self.lbl_instruction.setAlignment(Qt.AlignCenter)
        self.lbl_instruction.setStyleSheet(f"color: {COLOR_TEXT};")
        self.lbl_instruction.setWordWrap(True)
        layout.addWidget(self.lbl_instruction)

        # ── GIF del gesto ─────────────────────────────────────
        self.lbl_gif = QLabel()
        self.lbl_gif.setAlignment(Qt.AlignCenter)
        self.lbl_gif.setFixedHeight(260)
        self.lbl_gif.setStyleSheet("background: transparent;")
        layout.addWidget(self.lbl_gif)

        # ── Barra de progreso ─────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("")
        layout.addWidget(self.progress_bar)

        # ── Info de repeticion / gesto ────────────────────────
        self.lbl_rep_info = QLabel("")
        self.lbl_rep_info.setFont(QFont("Inter", 14))
        self.lbl_rep_info.setAlignment(Qt.AlignCenter)
        self.lbl_rep_info.setStyleSheet(f"color: {COLOR_TEXT_MUTED};")
        layout.addWidget(self.lbl_rep_info)

        # ── Siguiente gesto (visible solo en REST) ────────────
        self.lbl_next_gesture = QLabel("")
        self.lbl_next_gesture.setFont(QFont("Inter", 18, QFont.Bold))
        self.lbl_next_gesture.setAlignment(Qt.AlignCenter)
        self.lbl_next_gesture.setStyleSheet("color: #60a5fa;")
        self.lbl_next_gesture.setVisible(False)
        layout.addWidget(self.lbl_next_gesture)

        # ── Separador ─────────────────────────────────────────
        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.HLine)
        layout.addWidget(sep)

        # ── Grafico EMG pequeño ───────────────────────────────
        self._plot = create_plot_widget(
            title="Señal EMG en tiempo real",
            window_s=PROTOCOL_WINDOW_S,
            height=130,
        )
        layout.addWidget(self._plot["widget"])

    # ── Slots publicos (conectados por MainWindow) ─────────────

    def update_plot(self, data) -> None:
        update_plot(self._plot, data)

    def on_state_changed(self, state_name: str, info: dict):
        """Actualizar UI segun el estado del protocolo."""
        self._current_state = state_name

        if state_name == "transition":
            next_gesture = info.get("next_gesture")
            self.lbl_instruction.setText("¡LISTO!")
            self.lbl_instruction.setFont(QFont("Inter", 56, QFont.Bold))
            self.lbl_instruction.setStyleSheet("color: #22c55e;")
            self.lbl_rep_info.setText("Calibración completada — comenzando sesión")
            if next_gesture:
                self.lbl_next_gesture.setText(f"Primer gesto: {next_gesture['label']}")
                self.lbl_next_gesture.setVisible(True)
            self._set_bg(COLOR_BG)
            self._stop_gif()

        elif state_name == "gesture_active":
            gesture    = info["gesture"]
            set_id     = info["set_id"]
            n_sets     = info["n_sets"]
            g_idx      = info["gesture_index"] + 1
            n_gestures = info["n_gestures"]
            rep_id     = info["repetition_id"]

            self.lbl_instruction.setText(gesture["label"])
            self.lbl_instruction.setFont(QFont("Inter", 48, QFont.Bold))
            self.lbl_instruction.setStyleSheet(f"color: {COLOR_TEXT};")
            self.lbl_rep_info.setText(
                f"Set {set_id}/{n_sets}   ·   "
                f"Rep {rep_id}   ·   "
                f"Gesto {g_idx}/{n_gestures}  ({gesture['name']})"
            )
            self.lbl_next_gesture.setVisible(False)
            self._set_bg(COLOR_BG)
            self._load_gif(gesture.get("image", ""))

        elif state_name == "rest":
            set_id       = info["set_id"]
            n_sets       = info["n_sets"]
            rep_id       = info["repetition_id"]
            next_gesture = info.get("next_gesture")

            self.lbl_instruction.setText("DESCANSA")
            self.lbl_instruction.setFont(QFont("Inter", 52, QFont.Bold))
            self.lbl_instruction.setStyleSheet(f"color: {COLOR_TEXT_MUTED};")
            self.lbl_rep_info.setText(f"Set {set_id}/{n_sets}   ·   Rep {rep_id}")

            # Mostrar el siguiente gesto (siempre disponible en ciclo REST-first)
            if next_gesture:
                self.lbl_next_gesture.setText(f"Siguiente: {next_gesture['label']}")
                self.lbl_next_gesture.setVisible(True)

            self._set_bg(COLOR_BG_MUTED)
            self._stop_gif()

    def on_progress(self, elapsed: float, total: float):
        """Barra ascendente para prep/activo/descanso."""
        if total <= 0:
            return
        remaining = max(0.0, total - elapsed)
        pct = min(elapsed / total, 1.0)
        self.progress_bar.setFormat(f"{remaining:.1f}s")
        self.progress_bar.setValue(int(pct * 1000))

    # ── Helpers ───────────────────────────────────────────────

    def _set_bg(self, color: str):
        self.setStyleSheet(f"QWidget {{ background-color: {color}; }}")

    def _load_gif(self, image_name: str):
        self._stop_gif()
        if not image_name:
            self.lbl_gif.setText("🖐")
            self.lbl_gif.setFont(QFont("Inter", 80))
            return

        path = os.path.join(self._assets_dir, image_name)
        if not os.path.exists(path):
            name = image_name.replace(".gif", "").replace(".png", "")
            self.lbl_gif.setText(f"[ {name} ]")
            self.lbl_gif.setFont(QFont("Inter", 24))
            self.lbl_gif.setStyleSheet(f"color: {COLOR_TEXT_MUTED};")
            return

        movie = QMovie(path)
        movie.setScaledSize(QSize(260, 260))
        self.lbl_gif.setMovie(movie)
        movie.start()
        self._current_movie = movie

    def _stop_gif(self):
        if self._current_movie:
            self._current_movie.stop()
            self._current_movie = None
        self.lbl_gif.clear()
