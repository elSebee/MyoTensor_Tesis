#!/usr/bin/env python3
"""
=============================================================
 emg_monitor.py — Monitor EMG en tiempo real | MyoTensor
=============================================================
 Recibe datos CSV del ESP32-S3 via WiFi UDP Broadcast
 y los grafica en tiempo real con pyqtgraph.

 Formato UDP esperado del firmware (mismo que antes via Serial):
   timestamp_us,raw,centered,filtered,rectified,voltage_v

 Uso:
   python emg_monitor.py                  # escucha en UDP_PORT
   python emg_monitor.py --port 5005      # sobrescribir puerto

 Dependencias:
   pip install pyqtgraph PyQt5 numpy
=============================================================
"""

import sys
import argparse
import threading
import time
import socket
from collections import deque

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

# ─────────────────────────────────────────────
#  Constantes
# ─────────────────────────────────────────────
UDP_PORT  = 5005         # debe coincidir con UDP_PORT en config.h
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
#  Hilo de recepcion WiFi UDP (reemplaza SerialReader)
# ─────────────────────────────────────────────
class WiFiReader(threading.Thread):
    """Escucha paquetes UDP en 0.0.0.0:UDP_PORT y los parsea como CSV.
    Drop-in replacement de SerialReader.
    """
    def __init__(self, port: int, buf: DataBuffer):
        super().__init__(daemon=True, name="WiFiReader")
        self.port      = port
        self.buf       = buf
        self._stop     = threading.Event()
        self.connected = False
        self.error_msg = ""
        self._first_packet = True
        self._last_log     = time.time()
        self._last_count   = 0

    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.bind(("", self.port))
            sock.settimeout(1.0)   # wake each second para chequear _stop
            self.connected = True
            print(f"[OK] Escuchando UDP en 0.0.0.0:{self.port}")
        except OSError as e:
            self.error_msg = str(e)
            print(f"[ERROR] {e}")
            return

        residuo = ""   # fragmento incompleto de linea entre paquetes
        while not self._stop.is_set():
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError as e:
                print(f"[ERROR] UDP: {e}")
                break

            # Un paquete puede contener 0, 1 o varias lineas CSV
            texto = residuo + data.decode("utf-8", errors="ignore")
            lineas = texto.split("\n")
            residuo = lineas[-1]   # ultima parte (puede estar incompleta)

            for line in lineas[:-1]:
                line = line.strip()
                if not line or line.startswith(("#", "[", ">")):
                    continue
                parts = line.split(",")
                if len(parts) != len(COLUMNS):
                    self.buf.drops += 1
                    continue
                try:
                    row = {col: float(parts[i]) for i, col in enumerate(COLUMNS)}
                    self.buf.push(row)

                    # ── Primer paquete recibido ──────────────────────
                    if self._first_packet:
                        self._first_packet = False
                        print(f"[OK] ¡Conexión exitosa! Datos recibidos desde {addr[0]}:{addr[1]}")
                        print(f"     Formato: {', '.join(COLUMNS)}")

                    # ── Log periódico cada 5 s ───────────────────────
                    now = time.time()
                    if now - self._last_log >= 5.0:
                        delta   = self.buf.total_samples - self._last_count
                        rate    = delta / (now - self._last_log)
                        print(f"[UDP] {self.buf.total_samples} muestras | "
                              f"{rate:.0f} Hz | drops: {self.buf.drops}")
                        self._last_log   = now
                        self._last_count = self.buf.total_samples

                except ValueError:
                    self.buf.drops += 1

        sock.close()

    def stop(self):
        self._stop.set()


# ─────────────────────────────────────────────
#  Ventana principal pyqtgraph
# ─────────────────────────────────────────────
class EMGMonitor:
    def __init__(self, buf: DataBuffer, reader: WiFiReader):
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
    parser.add_argument("--port", type=int, default=UDP_PORT,
                        help=f"Puerto UDP de escucha (default: {UDP_PORT})")
    args = parser.parse_args()

    print(f"[INFO] Modo WiFi UDP | escuchando en 0.0.0.0:{args.port}")
    print(f"[INFO] El ESP32 debe estar en la misma red LAN")

    buf    = DataBuffer(WINDOW_N)
    reader = WiFiReader(args.port, buf)
    reader.start()

    time.sleep(0.5)
    if not reader.connected and reader.error_msg:
        print(f"[FATAL] {reader.error_msg}")
        sys.exit(1)

    EMGMonitor(buf, reader).run()


if __name__ == "__main__":
    main()
