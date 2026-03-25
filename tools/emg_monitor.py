#!/usr/bin/env python3
"""
=============================================================
 emg_monitor.py — Monitor EMG en tiempo real | MyoTensor
=============================================================
 Recibe datos CSV del ESP32-S3 via Serial @ 921600 baud
 y los grafica en tiempo real con pyqtgraph.

 Formato Serial esperado del firmware:
   timestamp_us,raw,centered,filtered,rectified,voltage_v

 Uso:
   python emg_monitor.py                  # auto-detecta puerto
   python emg_monitor.py --port /dev/ttyACM0

 Dependencias:
   pip install pyqtgraph pyserial PyQt5 numpy
=============================================================
"""

import sys
import argparse
import threading
import time
from collections import deque

import serial
import serial.tools.list_ports
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

# ─────────────────────────────────────────────
#  Constantes de visualizacion
# ─────────────────────────────────────────────
BAUD_RATE = 921600
FS_HZ     = 1000         # muestras por segundo del firmware
WINDOW_S  = 5.0          # segundos visibles en pantalla
WINDOW_N  = int(FS_HZ * WINDOW_S)
UPDATE_MS = 33           # refresco UI ~30 FPS

COLUMNS = ["timestamp_us", "raw", "centered", "filtered", "rectified", "voltage_v"]

COLORS = {
    "raw":       "#58a6ff",  # azul
    "filtered":  "#3fb950",  # verde
    "rectified": "#f78166",  # salmon
    "voltage_v": "#d2a8ff",  # violeta
}

# ─────────────────────────────────────────────
#  Auto-deteccion de puerto serie
# ─────────────────────────────────────────────
def auto_detect_port() -> str:
    keywords = ["CP210", "CH340", "FTDI", "USB Serial", "ACM", "usbserial"]
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description or "") + (p.manufacturer or "")
        if any(k.lower() in desc.lower() for k in keywords):
            return p.device
    if ports:
        return ports[-1].device
    raise RuntimeError("No se encontro ningun puerto serie. Conecte el ESP32-S3")


# ─────────────────────────────────────────────
#  Buffer compartido entre hilos
# ─────────────────────────────────────────────
class DataBuffer:
    """Ring buffer thread-safe para cada canal."""
    def __init__(self, size: int):
        self.lock  = threading.Lock()
        self._bufs = {col: deque([0.0] * size, maxlen=size) for col in COLUMNS[1:]}
        self.total_samples = 0
        self.drops         = 0

    def push(self, row: dict):
        with self.lock:
            for col in COLUMNS[1:]:
                self._bufs[col].append(row[col])
            self.total_samples += 1

    def snapshot(self, *cols):
        with self.lock:
            return tuple(np.array(self._bufs[c], dtype=np.float32) for c in cols)


# ─────────────────────────────────────────────
#  Hilo de lectura Serial (no bloquea el UI)
# ─────────────────────────────────────────────
class SerialReader(threading.Thread):
    def __init__(self, port: str, buf: DataBuffer):
        super().__init__(daemon=True, name="SerialReader")
        self.port      = port
        self.buf       = buf
        self._stop     = threading.Event()
        self.connected = False
        self.error_msg = ""

    def run(self):
        try:
            ser = serial.Serial(self.port, BAUD_RATE, timeout=1)
            self.connected = True
            print(f"[OK] Conectado a {self.port} @ {BAUD_RATE} baud")
        except serial.SerialException as e:
            self.error_msg = str(e)
            print(f"[ERROR] {e}")
            return

        while not self._stop.is_set():
            try:
                raw_line = ser.readline()
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="ignore").strip()

                # Ignorar cabeceras, mensajes de boot y lineas vacias
                if not line or line.startswith(("#", "[", ">")):
                    continue

                parts = line.split(",")
                if len(parts) != len(COLUMNS):
                    self.buf.drops += 1
                    continue

                row = {col: float(parts[i]) for i, col in enumerate(COLUMNS)}
                self.buf.push(row)

            except (ValueError, UnicodeDecodeError):
                self.buf.drops += 1
            except serial.SerialException as e:
                print(f"[ERROR] Serial: {e}")
                break

        ser.close()

    def stop(self):
        self._stop.set()


# ─────────────────────────────────────────────
#  Ventana principal pyqtgraph
# ─────────────────────────────────────────────
class EMGMonitor:
    def __init__(self, buf: DataBuffer, reader: SerialReader):
        self.buf    = buf
        self.reader = reader

        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        self.win = pg.GraphicsLayoutWidget(title="MyoTensor — EMG Monitor")
        self.win.resize(1280, 800)
        self.win.setBackground("#0d1117")

        self._build_ui()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(UPDATE_MS)
        self.t_start = time.time()

    def _build_ui(self):
        pg.setConfigOptions(antialias=True, foreground="#c9d1d9")

        def make_plot(title, ylabel, row, col=0, colspan=1):
            p = self.win.addPlot(row=row, col=col, colspan=colspan)
            p.setTitle(f"<span style='font-size:11pt;color:#c9d1d9'>{title}</span>")
            p.setLabel("left",   ylabel,       color="#8b949e")
            p.setLabel("bottom", "tiempo (s)", color="#8b949e")
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setMenuEnabled(False)
            return p

        self.p_raw  = make_plot("Señal Cruda (ADC)",          "cuentas", row=0, col=0)
        self.p_filt = make_plot("Señal Filtrada (Notch+BPF)", "cuentas", row=0, col=1)
        self.p_rect = make_plot("Envolvente EMG (|filtrada|)", "cuentas", row=1, colspan=2)
        self.p_volt = make_plot("Tensión de entrada",          "V",       row=2, colspan=2)

        pen = lambda c: pg.mkPen(c, width=1)
        self.c_raw  = self.p_raw. plot(pen=pen(COLORS["raw"]))
        self.c_filt = self.p_filt.plot(pen=pen(COLORS["filtered"]))
        self.c_rect = self.p_rect.plot(pen=pen(COLORS["rectified"]))
        self.c_volt = self.p_volt.plot(pen=pen(COLORS["voltage_v"]))

        self.lbl_status = self.win.addLabel(
            text="Esperando datos...", row=3, col=0, colspan=2,
            color="#8b949e", size="9pt"
        )

    def _update(self):
        raw, filtered, rectified, voltage = self.buf.snapshot(
            "raw", "filtered", "rectified", "voltage_v"
        )
        t = np.linspace(-WINDOW_S, 0, len(raw))

        self.c_raw. setData(t, raw)
        self.c_filt.setData(t, filtered)
        self.c_rect.setData(t, rectified)
        self.c_volt.setData(t, voltage)

        elapsed = time.time() - self.t_start
        rate    = self.buf.total_samples / elapsed if elapsed > 0 else 0
        self.lbl_status.setText(
            f"  Muestras: {self.buf.total_samples}  |  "
            f"Rate: {rate:.0f} Hz  |  "
            f"Drops: {self.buf.drops}  |  "
            f"Elapsed: {elapsed:.1f}s",
            color="#8b949e"
        )

    def run(self):
        self.win.show()
        print("[UI] Monitor abierto. Cierra la ventana o Ctrl+C para salir.")
        try:
            self.app.exec_()
        finally:
            self.reader.stop()


# ─────────────────────────────────────────────
#  Punto de entrada
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MyoTensor — EMG Monitor")
    parser.add_argument("--port", type=str, default=None,
                        help="Puerto serie (ej: /dev/ttyACM0)")
    args = parser.parse_args()

    port = args.port or auto_detect_port()
    print(f"[INFO] Puerto: {port}")

    buf    = DataBuffer(WINDOW_N)
    reader = SerialReader(port, buf)
    reader.start()

    time.sleep(0.5)
    if not reader.connected and reader.error_msg:
        print(f"[FATAL] {reader.error_msg}")
        sys.exit(1)

    EMGMonitor(buf, reader).run()


if __name__ == "__main__":
    main()
