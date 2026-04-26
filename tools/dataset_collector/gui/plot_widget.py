"""
gui/plot_widget.py — Factory de PlotWidget reutilizable.

Centraliza la creacion de graficos pyqtgraph con el estilo
del Dataset Collector, evitando duplicacion entre pantallas.

Renderizado basado en tiempo real:
  El plot avanza con el reloj del sistema (time.monotonic()),
  no con el timestamp del paquete UDP. Esto elimina el efecto
  "congela y salta" causado por el aliasing entre la tasa de
  llegada de batches y el timer del GUI.
"""

import time

import numpy as np
import pyqtgraph as pg

from .styles import FS_HZ, COLOR_BG, COLOR_TEXT_MUTED, COLOR_PLOT_LINE, COLOR_PLOT_GRID


def create_plot_widget(title: str, window_s: float, height: int) -> dict:
    """Crea un PlotWidget de pyqtgraph con estilo consistente.

    Args:
        title:    Titulo del grafico (acepta HTML).
        window_s: Duracion de la ventana temporal visible (segundos).
        height:   Altura fija del widget en pixeles.

    Returns:
        Diccionario con las claves:
          - widget   (pg.PlotWidget): el widget listo para agregar al layout
          - curve    (PlotDataItem):  curva para llamar setData()
          - t        (np.ndarray):    eje temporal precalculado
          - n        (int):           numero de muestras en la ventana
          - window_s (float):         duracion de la ventana
          - _t_last  (float):         timestamp de la ultima renderizacion
    """
    pw = pg.PlotWidget()
    pw.setBackground(COLOR_BG)
    pw.setFixedHeight(height)
    pw.setTitle(f"<span style='font-size:10pt;color:{COLOR_TEXT_MUTED}'>{title}</span>")
    pw.setLabel("left",   "amplitud", color=COLOR_TEXT_MUTED)
    pw.setLabel("bottom", "tiempo (s)", color=COLOR_TEXT_MUTED)
    pw.showGrid(x=True, y=True, alpha=0.15)
    pw.setMenuEnabled(False)
    pw.getAxis("bottom").setPen(pg.mkPen(COLOR_PLOT_GRID))
    pw.getAxis("left").setPen(pg.mkPen(COLOR_PLOT_GRID))

    pen   = pg.mkPen(COLOR_PLOT_LINE, width=1.5)
    curve = pw.plot(pen=pen)

    n = int(FS_HZ * window_s)
    t = np.linspace(-window_s, 0, n)

    return {
        "widget":   pw,
        "curve":    curve,
        "t":        t,
        "n":        n,
        "window_s": window_s,
        "_t_last":  time.monotonic(),   # para scroll continuo basado en reloj
        "_last_total": 0,               # ultima cantidad de muestras vistas
    }


def update_plot(plot_dict: dict, data: np.ndarray, total_samples: int = 0) -> None:
    """Actualiza la curva de un PlotWidget con nuevos datos.

    Scroll suave basado en tiempo real:
      Solo redibuja si llegaron nuevas muestras desde la ultima
      llamada. Esto evita frames donde el plot se muestra estatico
      (congela) alternados con frames donde salta varias muestras.

    Args:
        plot_dict:     Diccionario retornado por create_plot_widget.
        data:          Array de la señal (ultimas muestras del buffer).
        total_samples: Total acumulado de muestras (de DataBuffer).
                       Omitir para comportamiento legacy (siempre redibuja).
    """
    n = plot_dict["n"]
    tail = data[-n:]
    if len(tail) < n:
        return  # buffer no llenado aun — evitar artefactos al inicio

    # Optimizacion: solo redibujar cuando llegaron datos nuevos.
    # Com el deque es circular, comparar total_samples es O(1).
    if total_samples > 0:
        if total_samples == plot_dict["_last_total"]:
            return   # sin datos nuevos — no redibujar (evita frames congelados)
        plot_dict["_last_total"] = total_samples

    plot_dict["curve"].setData(plot_dict["t"], tail)
