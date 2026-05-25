"""
=============================================================
 udp_receiver.py — Hilo de recepcion UDP | Dataset Collector
=============================================================
 Parsea el protocolo binario del firmware Streamer.

 Formato del paquete (little-endian, 212 bytes):
   Header (12 bytes):
     magic      : uint16 = 0xEB90  (sincronismo)
     n_samples  : uint8            (= BATCH_SIZE, tipicamente 50)
     reserved   : uint8  = 0x00
     seq_num    : uint32           (contador — detectar perdida)
     timestamp0 : uint32           (µs de la primera muestra)
   Payload:
     float32[n_samples]            (señal filtrada post-DSP)

 Reconstruccion de timestamps en Python:
   timestamp_k = timestamp_0 + k * 1000  (µs, Fs = 1000 Hz)

 Columnas expuestas via DataBuffer.snapshot():
   "timestamp_us"  float64 — microsegundos Unix relativo al firmware
   "filtered"      float32 — señal post-DSP (Notch + HPF + LPF)
=============================================================
"""

import socket
import struct
import threading
import time
from collections import deque

import numpy as np

# ============================================================
# Constantes del protocolo — deben coincidir con config.h
# ============================================================
PACKET_MAGIC = 0xEB90
FS_HZ        = 1000
SAMPLE_US    = 1_000_000 // FS_HZ          # 1000 µs entre muestras

# Formato del header: little-endian, sin padding
#   H = uint16 (magic)
#   B = uint8  (n_samples)
#   B = uint8  (reserved)
#   I = uint32 (seq_num)
#   I = uint32 (timestamp0)
HEADER_FMT  = "<HBBII"
HEADER_SIZE = struct.calcsize(HEADER_FMT)   # = 12 bytes

# Columnas disponibles en DataBuffer.snapshot()
COLUMNS = ["timestamp_us", "filtered"]


class DataBuffer:
    """Ring buffer thread-safe para señal EMG.

    Interfaz:
        push_batch(timestamps, filtered) — insertar desde hilo UDP
        snapshot(*cols)                  — leer para GUI/grabacion
            cols opcionales: "timestamp_us", "filtered"
            sin cols: devuelve (timestamp_us_array, filtered_array)
    """

    def __init__(self, size: int):
        self.lock = threading.Lock()
        self._bufs = {
            "timestamp_us": deque([0.0] * size, maxlen=size),
            "filtered":     deque([0.0] * size, maxlen=size),
        }
        self.total_samples = 0
        self.drops         = 0   # muestras descartadas en firmware (ring lleno)
        self.pkt_loss      = 0   # paquetes perdidos (gap en seq_num)

    def push_batch(self, timestamps: np.ndarray, filtered: np.ndarray):
        """Insertar un batch de N muestras de una sola vez (hilo UDP)."""
        with self.lock:
            for ts, filt in zip(timestamps, filtered):
                self._bufs["timestamp_us"].append(float(ts))
                self._bufs["filtered"].append(float(filt))
            self.total_samples += len(filtered)

    def snapshot(self, *cols):
        """Devolver arrays numpy de las columnas solicitadas.

        Uso:
            filtered, = buf.snapshot("filtered")
            ts, filt  = buf.snapshot("timestamp_us", "filtered")
            ts, filt  = buf.snapshot()   # sin args: devuelve ambas columnas
        """
        if not cols:
            cols = ("timestamp_us", "filtered")
        with self.lock:
            return tuple(
                np.array(self._bufs[c], dtype=np.float32) for c in cols
            )


class UDPReceiver(threading.Thread):
    """Hilo daemon que escucha paquetes UDP binarios del Streamer.

    Parametros:
        port:      Puerto UDP (default 5005)
        buf:       DataBuffer para grafico en tiempo real
        callbacks: Lista de funciones on_batch(timestamps, filtered)
    """

    def __init__(self, port: int, buf: DataBuffer, callbacks=None):
        super().__init__(daemon=True, name="UDPReceiver")
        self.port       = port
        self.buf        = buf
        self.callbacks  = callbacks or []
        self._stop      = threading.Event()
        self.connected  = False
        self.error_msg  = ""

        self._first_pkt    = True
        self._last_log     = time.time()
        self._last_count   = 0
        self._last_seq: int | None = None

    def add_callback(self, fn):
        """Registrar un callback que recibe (timestamps_us, filtered) por batch."""
        self.callbacks.append(fn)

    # ----------------------------------------------------------
    # Hilo principal de recepcion
    # ----------------------------------------------------------
    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.bind(("", self.port))
            sock.settimeout(1.0)
            self.connected = True
            print(f"[OK] Escuchando UDP binario en 0.0.0.0:{self.port}")
        except OSError as e:
            self.error_msg = str(e)
            print(f"[ERROR] {e}")
            return

        while not self._stop.is_set():
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError as e:
                print(f"[ERROR] UDP: {e}")
                break

            if data == b"MYOTENSOR_PING":
                try:
                    reply_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    reply_sock.sendto(b"MYOTENSOR_PONG", (addr[0], self.port))
                    reply_sock.close()
                    print(f"[INFO] Discovery: Recibido PING de ESP32 ({addr[0]}). Enviado PONG.")
                except Exception as e:
                    print(f"[ERROR] Discovery error al responder PONG: {e}")
                continue

            self._parse_packet(data, addr)

        sock.close()

    # ----------------------------------------------------------
    # Parseo del paquete binario
    # ----------------------------------------------------------
    def _parse_packet(self, data: bytes, addr):
        # Verificar tamaño minimo
        if len(data) < HEADER_SIZE:
            self.buf.drops += len(data)
            return

        # Deserializar header
        magic, n_samples, _reserved, seq_num, timestamp0 = \
            struct.unpack_from(HEADER_FMT, data, 0)

        # Verificar magic word
        if magic != PACKET_MAGIC:
            return

        # Verificar tamaño del payload
        expected = HEADER_SIZE + n_samples * 4
        if len(data) < expected:
            self.buf.drops += n_samples
            return

        # Detectar perdida de paquetes (o reboot del ESP)
        if self._last_seq is not None:
            if seq_num < self._last_seq:
                # O es un paquete desordenado (llego tarde) o el ESP se reinicio
                if seq_num < 10 and self._last_seq > 100:
                    print(f"[INFO] ESP reiniciado (seq {self._last_seq} → {seq_num})")
                    self._last_seq = seq_num
                else:
                    # Paquete desordenado (out-of-order). Lo procesamos pero no retrocedemos
                    # self._last_seq para evitar cascadas de falsos positivos en las perdidas.
                    pass
            else:
                gap = seq_num - self._last_seq - 1
                if gap > 0 and gap < 1000:
                    self.buf.pkt_loss += gap
                    lost_samples = gap * n_samples
                    print(f"[WARN] Paquetes perdidos: {gap} "
                          f"(seq {self._last_seq+1}..{seq_num-1}) "
                          f"≈ {lost_samples} muestras")
                self._last_seq = seq_num
        else:
            self._last_seq = seq_num

        # Deserializar payload: N floats contiguos
        fmt_payload = f"<{n_samples}f"
        filtered = np.array(
            struct.unpack_from(fmt_payload, data, HEADER_SIZE),
            dtype=np.float32
        )

        # Reconstruir timestamps: ts_k = ts_0 + k * SAMPLE_US
        timestamps = timestamp0 + np.arange(n_samples, dtype=np.float64) * SAMPLE_US

        # Empujar al buffer de visualizacion
        self.buf.push_batch(timestamps, filtered)

        # Notificar callbacks
        for cb in self.callbacks:
            cb(timestamps, filtered)

        # Primer paquete recibido
        if self._first_pkt:
            self._first_pkt = False
            pkt_bytes = HEADER_SIZE + n_samples * 4
            print(f"[OK] Datos recibidos desde {addr[0]}:{addr[1]} "
                  f"| {pkt_bytes}B/pkt | {n_samples} muestras/pkt")

        # Log periodico cada 5s
        now = time.time()
        if now - self._last_log >= 5.0:
            delta = self.buf.total_samples - self._last_count
            rate  = delta / (now - self._last_log)
            print(f"[UDP] {self.buf.total_samples} muestras | "
                  f"{rate:.0f} Hz | pkt_loss: {self.buf.pkt_loss} | "
                  f"drops_fw: {self.buf.drops}")
            self._last_log   = now
            self._last_count = self.buf.total_samples

    def stop(self):
        self._stop.set()
