"""
gui/screen_done.py — Pantalla 3: Resumen de sesion completada.

Muestra el icono de check, la cantidad de muestras grabadas,
las rutas de los archivos generados y un boton para iniciar
una nueva sesion.

Emite new_session_requested() cuando el usuario quiere repetir.
"""

import os

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
)

from .styles import COLOR_TEXT_MUTED


class DoneScreen(QWidget):
    """Pantalla final: resumen de sesion y navegacion."""

    new_session_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(60, 60, 60, 60)
        layout.setSpacing(20)

        layout.addStretch()

        # Check icon
        lbl_check = QLabel("✓")
        lbl_check.setFont(QFont("Inter", 64))
        lbl_check.setAlignment(Qt.AlignCenter)
        lbl_check.setStyleSheet("color: #059669;")
        layout.addWidget(lbl_check)

        # Titulo
        lbl_title = QLabel("¡Sesión completada!")
        lbl_title.setFont(QFont("Inter", 32, QFont.Bold))
        lbl_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl_title)

        # Info de la sesion
        self.lbl_info = QLabel("")
        self.lbl_info.setFont(QFont("Inter", 13))
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setStyleSheet(f"color: {COLOR_TEXT_MUTED};")
        self.lbl_info.setWordWrap(True)
        layout.addWidget(self.lbl_info)

        layout.addStretch()

        # Boton nueva sesion
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn = QPushButton("Nueva sesión")
        btn.setObjectName("newSessionButton")
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(self.new_session_requested.emit)
        btn_layout.addWidget(btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def set_result(
        self,
        n_samples: int,
        csv_path: str | None,
        meta_path: str | None,
        subject_id: str,
    ) -> None:
        """Actualizar la pantalla con el resumen de la sesion guardada."""
        dur = n_samples / 1000.0 if n_samples else 0.0

        lines = [
            f"Sujeto: {subject_id}",
            f"Muestras grabadas: {n_samples:,}",
            f"Duración: {dur:.1f} segundos",
            "",
        ]
        if csv_path:
            lines.append(f"CSV: {os.path.abspath(csv_path)}")
        if meta_path:
            lines.append(f"Metadata: {os.path.abspath(meta_path)}")

        self.lbl_info.setText("\n".join(lines))
