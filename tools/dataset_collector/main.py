#!/usr/bin/env python3
"""
=============================================================
 main.py — Punto de entrada del Dataset Collector | MyoTensor
=============================================================
 Orquesta todos los modulos:
   1. Lee protocol.yaml
   2. Crea DataBuffer + UDPReceiver
   3. Crea Calibrator + ProtocolEngine + DataWriter
   4. Conecta callbacks UDP (calibrator.on_batch, writer.on_batch)
   5. Crea GUI (paquete gui/) y conecta señales
   6. Inicia el loop de Qt

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

from udp_receiver  import DataBuffer, UDPReceiver
from protocol_engine import ProtocolEngine
from calibrator    import Calibrator
from data_writer   import DataWriter
from gui           import MainWindow, FS_HZ, MONITOR_WINDOW_S


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
        help="Ruta al archivo protocol.yaml (default: protocol.yaml en el mismo directorio)",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Puerto UDP de escucha (sobreescribe el valor en protocol.yaml)",
    )
    args = parser.parse_args()

    # ── Directorio base (donde vive main.py) ─────────────────
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Cargar configuracion ──────────────────────────────────
    config_path = args.config or os.path.join(base_dir, "protocol.yaml")
    if not os.path.exists(config_path):
        print(f"[FATAL] No se encontro el archivo de configuracion: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    print(f"[OK] Configuracion cargada desde: {config_path}")

    port       = args.port or config.get("network", {}).get("port", 5005)
    output_dir = os.path.normpath(
        os.path.join(base_dir, config.get("output", {}).get("base_dir", "../../dataset/raw"))
    )
    assets_dir = os.path.join(base_dir, "assets")

    # ── Resumen del protocolo ─────────────────────────────────
    proto    = config["protocol"]
    gestures = config["gestures"]
    cal_cfg  = config.get("calibration", {})
    n_g      = len(gestures)
    n_sets   = proto.get("n_sets", 10)
    d_g      = proto["gesture_duration_s"]
    d_r      = proto["rest_duration_s"]
    d_cal    = cal_cfg.get("noise_duration_s", 5.0) + cal_cfg.get("mvc_duration_s", 3.0)
    total_s  = n_sets * n_g * (d_g + d_r)

    print(f"\n{'='*55}")
    print(f"  MyoTensor — Recolector de Dataset")
    print(f"{'='*55}")
    print(f"  Gestos       : {n_g} ({', '.join(g['name'] for g in gestures)})")
    print(f"  Sets         : {n_sets} (Protocolo de Bloques Aleatorizados)")
    print(f"  Ciclo        : {d_g}s gesto + {d_r}s reposo")
    print(f"  Calibracion  : {d_cal:.0f}s (ruido + MVC)")
    print(f"  Total aprox  : {total_s:.0f}s ({total_s/60:.1f} min)")
    print(f"  Puerto UDP   : {port}")
    print(f"  Salida       : {output_dir}")
    print(f"{'='*55}\n")

    # ── Crear componentes ─────────────────────────────────────
    buf        = DataBuffer(int(FS_HZ * MONITOR_WINDOW_S))
    receiver   = UDPReceiver(port, buf)
    engine     = ProtocolEngine(config)
    calibrator = Calibrator(engine, config)   # engine primero
    writer     = DataWriter(engine, config, subject_info={}, output_dir=output_dir)

    # Registrar callbacks UDP (orden: calibrator primero, writer segundo)
    receiver.add_callback(calibrator.on_batch)
    receiver.add_callback(writer.on_batch)

    # ── Iniciar recepcion UDP ─────────────────────────────────
    receiver.start()
    time.sleep(0.3)

    if not receiver.connected and receiver.error_msg:
        print(f"[FATAL] No se pudo abrir socket UDP: {receiver.error_msg}")
        sys.exit(1)

    # ── Iniciar GUI ───────────────────────────────────────────
    app    = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(buf, engine, writer, calibrator, config, assets_dir)
    window.show()

    print("[UI] Ventana abierta. Cierra la ventana o Ctrl+C para salir.")

    try:
        sys.exit(app.exec_())
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
