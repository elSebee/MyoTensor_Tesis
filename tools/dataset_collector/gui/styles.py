"""
gui/styles.py — Constantes de visualizacion y estilos CSS.

Centraliza todos los tokens de diseño para que los modulos
de pantalla no tengan valores hardcodeados.
"""

# ── Constantes de adquisicion y visualizacion ──────────────────
FS_HZ             = 1000    # frecuencia de muestreo (Hz)
MONITOR_WINDOW_S  = 5.0     # segundos visible en pantalla de monitoreo
PROTOCOL_WINDOW_S = 3.0     # segundos visible durante el protocolo
UPDATE_MS         = 33      # intervalo de refresco del grafico (~30 FPS)

# ── Paleta de colores ──────────────────────────────────────────
COLOR_BG          = "#FDFDFD"
COLOR_BG_SUBTLE   = "#f8fafc"
COLOR_BG_MUTED    = "#f1f5f9"
COLOR_BORDER      = "#e2e8f0"
COLOR_TEXT        = "#1a1a2e"
COLOR_TEXT_MUTED  = "#64748b"
COLOR_TEXT_LABEL  = "#475569"
COLOR_PRIMARY     = "#2563eb"
COLOR_PRIMARY_HOV = "#1d4ed8"
COLOR_SUCCESS     = "#059669"
COLOR_SUCCESS_HOV = "#047857"
COLOR_WARNING     = "#d97706"
COLOR_DISABLED    = "#94a3b8"
COLOR_PLOT_LINE   = "#2563eb"
COLOR_PLOT_GRID   = "#cbd5e1"
COLOR_CALIBRATING = "#7c3aed"   # violeta — fase de calibracion
COLOR_MVC         = "#dc2626"   # rojo — fase de MVC


# ── Hoja de estilos global ─────────────────────────────────────
MAIN_STYLE = f"""
QMainWindow, QWidget {{
    background-color: {COLOR_BG};
    font-family: "Inter", "Noto Sans", "Segoe UI", "Helvetica Neue", sans-serif;
}}

QLabel {{
    color: {COLOR_TEXT};
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    border: 2px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 14px;
    background-color: {COLOR_BG_SUBTLE};
    color: {COLOR_TEXT};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {COLOR_PRIMARY};
}}

QComboBox::drop-down {{
    border: none;
    padding-right: 8px;
}}

QPushButton#startButton {{
    background-color: {COLOR_PRIMARY};
    color: white;
    border: none;
    border-radius: 12px;
    padding: 16px 48px;
    font-size: 18px;
    font-weight: bold;
}}

QPushButton#startButton:hover {{
    background-color: {COLOR_PRIMARY_HOV};
}}

QPushButton#startButton:disabled {{
    background-color: {COLOR_DISABLED};
}}

QPushButton#calibrateButton {{
    background-color: {COLOR_CALIBRATING};
    color: white;
    border: none;
    border-radius: 12px;
    padding: 16px 48px;
    font-size: 18px;
    font-weight: bold;
}}

QPushButton#calibrateButton:hover {{
    background-color: #6d28d9;
}}

QPushButton#calibrateButton:disabled {{
    background-color: {COLOR_DISABLED};
}}

QPushButton#newSessionButton {{
    background-color: {COLOR_SUCCESS};
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px 40px;
    font-size: 16px;
    font-weight: bold;
}}

QPushButton#newSessionButton:hover {{
    background-color: {COLOR_SUCCESS_HOV};
}}

QProgressBar {{
    border: none;
    border-radius: 10px;
    background-color: {COLOR_BORDER};
    text-align: center;
    height: 22px;
    font-size: 12px;
    color: {COLOR_TEXT_MUTED};
}}

QProgressBar::chunk {{
    background-color: {COLOR_PRIMARY};
    border-radius: 10px;
}}

QProgressBar#calibBar::chunk {{
    background-color: {COLOR_CALIBRATING};
    border-radius: 10px;
}}

QProgressBar#mvcBar::chunk {{
    background-color: {COLOR_MVC};
    border-radius: 10px;
}}

QFrame#separator {{
    background-color: {COLOR_BORDER};
    max-height: 1px;
}}
"""
