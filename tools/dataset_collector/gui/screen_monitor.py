"""
gui/screen_monitor.py — Pantalla 0: Monitoreo y datos del sujeto.

Muestra:
  - Formulario de datos del sujeto (ID, edad, género, lateralidad,
    circunferencia del antebrazo, músculo)
  - Gráfico EMG en tiempo real (últimos 5 segundos)
  - Barra de estado con tasa de muestreo, muestras y drops
  - Botón "Iniciar Adquisición" que valida el formulario

Emite la señal start_requested(dict) con los datos del sujeto
cuando el usuario presiona el botón y la validacion pasa.
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QPushButton, QFrame, QMessageBox,
)

from .styles import (
    COLOR_TEXT_MUTED, COLOR_TEXT_LABEL, MONITOR_WINDOW_S,
)
from .plot_widget import create_plot_widget, update_plot


class MonitorScreen(QWidget):
    """Pantalla de monitoreo y captura de datos del sujeto."""

    # Emitida cuando el usuario presiona Iniciar y la validacion pasa.
    # Payload: dict con datos del sujeto (subject_id, age, etc.)
    start_requested = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # ── Construccion de UI ─────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(16)

        # ── Titulo ─────────────────────────────────────────────
        title = QLabel("MyoTensor — Recolector")
        title.setFont(QFont("Inter", 20, QFont.Bold))
        title.setAlignment(Qt.AlignLeft)
        layout.addWidget(title)

        subtitle = QLabel(
            "Completa los datos del sujeto y verifica la señal EMG "
            "antes de iniciar la calibración."
        )
        subtitle.setFont(QFont("Inter", 11))
        subtitle.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; margin-bottom: 8px;")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # ── Formulario del sujeto ─────────────────────────────
        layout.addWidget(self._build_form())

        # ── Grafico EMG ──────────────────────────────────────
        self._plot = create_plot_widget(
            title="Señal EMG filtrada — últimos 5 segundos",
            window_s=MONITOR_WINDOW_S,
            height=240,
        )
        layout.addWidget(self._plot["widget"])

        # ── Status bar ────────────────────────────────────────
        self.lbl_status = QLabel("⏳ Esperando datos del ESP32...")
        self.lbl_status.setFont(QFont("Inter", 11))
        self.lbl_status.setStyleSheet(f"color: {COLOR_TEXT_MUTED};")
        layout.addWidget(self.lbl_status)

        # ── Boton inicio ──────────────────────────────────────
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_start = QPushButton("▶  Iniciar Adquisición")
        self.btn_start.setObjectName("startButton")
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.clicked.connect(self._on_start_clicked)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _build_form(self) -> QFrame:
        form_frame = QFrame()
        form_frame.setStyleSheet(
            "QFrame { background-color: #f8fafc; border-radius: 12px; padding: 16px; }"
        )
        grid = QGridLayout(form_frame)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(10)

        lbl_style = f"font-size: 12px; font-weight: bold; color: {COLOR_TEXT_LABEL};"

        def lbl(text: str) -> QLabel:
            l = QLabel(text)
            l.setStyleSheet(lbl_style)
            return l

        # Fila 0-1: ID, Edad, Género, Lateralidad
        self.input_id = QLineEdit("S01")
        self.input_id.setMaximumWidth(120)

        self.input_age = QSpinBox()
        self.input_age.setRange(10, 90)
        self.input_age.setValue(23)
        self.input_age.setMaximumWidth(100)

        self.input_gender = QComboBox()
        self.input_gender.addItems(["Masculino", "Femenino"])
        self.input_gender.setMaximumWidth(140)

        self.input_laterality = QComboBox()
        self.input_laterality.addItems(["Diestro", "Zurdo"])
        self.input_laterality.setMaximumWidth(140)

        grid.addWidget(lbl("ID Sujeto"),     0, 0)
        grid.addWidget(self.input_id,         1, 0)
        grid.addWidget(lbl("Edad"),           0, 1)
        grid.addWidget(self.input_age,        1, 1)
        grid.addWidget(lbl("Género"),         0, 2)
        grid.addWidget(self.input_gender,     1, 2)
        grid.addWidget(lbl("Lateralidad"),    0, 3)
        grid.addWidget(self.input_laterality, 1, 3)

        # Fila 2-3: Circunferencia, Músculo
        self.input_circumference = QDoubleSpinBox()
        self.input_circumference.setRange(10.0, 50.0)
        self.input_circumference.setValue(25.0)
        self.input_circumference.setDecimals(1)
        self.input_circumference.setSuffix(" cm")
        self.input_circumference.setMaximumWidth(140)

        self.input_muscle = QLineEdit("")
        self.input_muscle.setMaximumWidth(260)

        grid.addWidget(lbl("Circunferencia antebrazo (cm)"), 2, 0, 1, 2)
        grid.addWidget(self.input_circumference,              3, 0)
        grid.addWidget(lbl("Músculo"),                        2, 2, 1, 2)
        grid.addWidget(self.input_muscle,                     3, 2, 1, 2)

        return form_frame

    # ── Slots publicos (llamados por MainWindow) ───────────────

    def update_plot(self, data, total_samples: int = 0) -> None:
        """Actualizar grafico con los ultimos datos del buffer."""
        update_plot(self._plot, data, total_samples)

    def update_status(self, rate: float, total_samples: int, drops: int) -> None:
        """Actualizar barra de estado."""
        if total_samples == 0:
            self.lbl_status.setText("⏳ Esperando datos del ESP32...")
            return
        icon = "🟢" if rate > 900 else "🟡" if rate > 0 else "🔴"
        self.lbl_status.setText(
            f"{icon}  Rate: {rate:.0f} Hz   |   "
            f"Muestras: {total_samples}   |   "
            f"Drops: {drops}"
        )

    # ── Logica interna ─────────────────────────────────────────

    def _get_subject_info(self) -> dict:
        gender_map = {"Masculino": "m", "Femenino": "f"}
        lat_map    = {"Diestro": "r", "Zurdo": "l"}
        return {
            "subject_id":            self.input_id.text().strip(),
            "age":                   self.input_age.value(),
            "gender":                gender_map.get(self.input_gender.currentText(), "m"),
            "laterality":            lat_map.get(self.input_laterality.currentText(), "r"),
            "forearm_circumference_cm": self.input_circumference.value(),
            "muscle":                self.input_muscle.text().strip(),
        }

    def _on_start_clicked(self):
        info = self._get_subject_info()

        if not info["subject_id"]:
            QMessageBox.warning(self, "Dato faltante",
                                "Ingresa un ID de sujeto antes de iniciar.")
            return

        if not info["muscle"]:
            QMessageBox.warning(self, "Dato faltante",
                                "Ingresa el nombre del músculo.")
            return

        self.start_requested.emit(info)
