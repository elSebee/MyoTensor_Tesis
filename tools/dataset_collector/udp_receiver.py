"""
=============================================================
 udp_receiver.py — Hilo de recepcion UDP | Dataset Collector
=============================================================
 Basado en WiFiReader de emg_monitor.py.
 Parsea CSV del firmware modo dataset:
   timestamp_us,raw,centered,filtered,voltage_v

 Diferencias con el monitor original:
   - Soporta callbacks on_sample para que DataWriter registre datos
   - Columnas sin rectified (se calcula en post-proceso)
=============================================================
"""

import socket
import threading
import time
from collections import deque

import numpy as np

# Columnas del CSV que envia el firmware en modo DATASET_MODE
COLUMNS = ["timestamp_us", "raw", "centered", "filtered", "voltage_v"]


class DataBuffer:
    """Ring buffer thread-safe para cada canal EMG."""

    def __init__(self, size: int):
        self.lock = threading.Lock()
        self._bufs = {col: deque([0.0] * size, maxlen=size) for col in COLUMNS[1:]}
        self.total_samples = 0
        self.drops = 0

    def push(self, row: dict):
        with self.lock:
            for col in COLUMNS[1:]:
                self._bufs[col].append(row[col])
            self.total_samples += 1

    def snapshot(self, *cols):
        with self.lock:
            return tuple(np.array(self._bufs[c], dtype=np.float32) for c in cols)


class UDPReceiver(threading.Thread):
    """Hilo daemon que escucha paquetes UDP broadcast y los parsea como CSV.

    Parametros:
        port:       Puerto UDP a escuchar (default 5005)
        buf:        DataBuffer para grafico en tiempo real
        callbacks:  Lista de funciones on_sample(dict) para registrar datos
    """

    def __init__(self, port: int, buf: DataBuffer, callbacks=None):
        super().__init__(daemon=True, name="UDPReceiver")
        self.port = port
        self.buf = buf
        self.callbacks = callbacks or []
        self._stop = threading.Event()
        self.connected = False
        self.error_msg = ""
        self._first_packet = True
        self._last_log = time.time()
        self._last_count = 0

    def add_callback(self, fn):
        """Registrar un callback que recibe un dict por cada muestra."""
        self.callbacks.append(fn)

    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.bind(("", self.port))
            sock.settimeout(1.0)
            self.connected = True
            print(f"[OK] Escuchando UDP en 0.0.0.0:{self.port}")
        except OSError as e:
            self.error_msg = str(e)
            print(f"[ERROR] {e}")
            return

        residuo = ""
        while not self._stop.is_set():
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError as e:
                print(f"[ERROR] UDP: {e}")
                break

            texto = residuo + data.decode("utf-8", errors="ignore")
            lineas = texto.split("\n")
            residuo = lineas[-1]

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

                    # Notificar a todos los callbacks
                    for cb in self.callbacks:
                        cb(row)

                    if self._first_packet:
                        self._first_packet = False
                        print(f"[OK] Datos recibidos desde {addr[0]}:{addr[1]}")

                    # Log periodico cada 5s
                    now = time.time()
                    if now - self._last_log >= 5.0:
                        delta = self.buf.total_samples - self._last_count
                        rate = delta / (now - self._last_log)
                        print(f"[UDP] {self.buf.total_samples} muestras | "
                              f"{rate:.0f} Hz | drops: {self.buf.drops}")
                        self._last_log = now
                        self._last_count = self.buf.total_samples

                except ValueError:
                    self.buf.drops += 1

        sock.close()

    def stop(self):
        self._stop.set()
