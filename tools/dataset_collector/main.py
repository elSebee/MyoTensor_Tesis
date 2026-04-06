#!/usr/bin/env python3
"""
=============================================================
 main.py — Punto de entrada del Dataset Collector | MyoTensor
=============================================================
 Orquesta todos los modulos:
   1. Lee protocol.yaml
   2. Crea DataBuffer + UDPReceiver
   3. Crea ProtocolEngine + DataWriter
   4. Crea GUI y conecta señales
   5. Inicia el loop de Qt

 Uso:
   python main.py                      # usa protocol.yaml local
   python main.py --config otro.yaml   # config alternativa
   python main.py --port 5005          # sobrescribir puerto

 Dependencias:
   pip install -r ../requirements.txt
=============================================================
"""

import argparse
import os
import sys
import time

import yaml
from PyQt5.QtWidgets import QApplication

from udp_receiver import DataBuffer, UDPReceiver
from protocol_engine import ProtocolEngine
from data_writer import DataWriter
from gui import MainWindow, FS_HZ, MONITOR_WINDOW_S


def load_config(path: str) -> dict:
    """Carga el archivo de configuracion YAML."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="MyoTensor — Recolector de Dataset EMG"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Ruta al archivo protocol.yaml (default: protocol.yaml en el mismo directorio)"
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Puerto UDP de escucha (sobreescribe el valor en protocol.yaml)"
    )
    args = parser.parse_args()

    # ── Determinar directorio base (donde vive main.py) ──────
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Cargar configuracion ─────────────────────────────────
    config_path = args.config or os.path.join(base_dir, "protocol.yaml")
    if not os.path.exists(config_path):
        print(f"[FATAL] No se encontro el archivo de configuracion: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    print(f"[OK] Configuracion cargada desde: {config_path}")

    # Sobreescribir puerto si se paso por argumento
    port = args.port or config.get("network", {}).get("port", 5005)

    # ── Resolver ruta de salida ──────────────────────────────
    output_base = config.get("output", {}).get("base_dir", "../../dataset/raw")
    output_dir = os.path.normpath(os.path.join(base_dir, output_base))

    # ── Resolver ruta de assets ──────────────────────────────
    assets_dir = os.path.join(base_dir, "assets")

    # ── Resumen del protocolo ────────────────────────────────
    proto = config["protocol"]
    gestures = config["gestures"]
    n_gestos = len(gestures)
    n_reps = proto["repetitions"]
    dur_gesto = proto["gesture_duration_s"]
    dur_rest = proto["rest_duration_s"]
    total_s = n_gestos * n_reps * (dur_gesto + dur_rest)

    print(f"\n{'='*55}")
    print(f"  MyoTensor — Recolector de Dataset")
    print(f"{'='*55}")
    print(f"  Gestos      : {n_gestos} ({', '.join(g['name'] for g in gestures)})")
    print(f"  Repeticiones: {n_reps} por gesto")
    print(f"  Duracion    : {dur_gesto}s gesto + {dur_rest}s reposo")
    print(f"  Total       : {total_s:.0f}s ({total_s/60:.1f} minutos)")
    print(f"  Puerto UDP  : {port}")
    print(f"  Salida      : {output_dir}")
    print(f"{'='*55}\n")

    # ── Crear componentes ────────────────────────────────────
    buf = DataBuffer(int(FS_HZ * MONITOR_WINDOW_S))
    receiver = UDPReceiver(port, buf)
    engine = ProtocolEngine(config)
    writer = DataWriter(engine, config, subject_info={}, output_dir=output_dir)

    # Registrar callback del writer en el receiver
    receiver.add_callback(writer.on_sample)

    # ── Iniciar recepcion UDP ────────────────────────────────
    receiver.start()
    time.sleep(0.3)

    if not receiver.connected and receiver.error_msg:
        print(f"[FATAL] No se pudo abrir socket UDP: {receiver.error_msg}")
        sys.exit(1)

    # ── Iniciar GUI ──────────────────────────────────────────
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(buf, engine, writer, config, assets_dir)
    window.show()

    print("[UI] Ventana abierta. Cierra la ventana o Ctrl+C para salir.")

    try:
        sys.exit(app.exec_())
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
