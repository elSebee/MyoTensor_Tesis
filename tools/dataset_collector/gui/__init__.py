"""
gui/__init__.py — Paquete de interfaz grafica del Dataset Collector.

Re-exporta los simbolos que main.py necesita importar directamente
desde 'gui', manteniendo la misma interfaz que el antiguo gui.py.
"""

from .main_window import MainWindow
from .styles import FS_HZ, MONITOR_WINDOW_S

__all__ = ["MainWindow", "FS_HZ", "MONITOR_WINDOW_S"]
