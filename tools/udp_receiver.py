import socket
import sys
import struct

UDP_PORT = 5005

# Crear socket UDP y enlazar a todas las interfaces
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    sock.bind(("0.0.0.0", UDP_PORT))
except Exception as e:
    print(f"Error al abrir puerto UDP {UDP_PORT}: {e}")
    sys.exit(1)

print("\033[96m=== Receptor UDP de Clasificaciones MyoTensor ===\033[0m")
print(f"Escuchando transmisiones en puerto {UDP_PORT} (WiFi)...")
print("Presiona Ctrl+C para salir.\n")

GESTURES = ["REPOSO", "PALMA", "PUNO", "PAZ"]
COLORS = [
    "\033[90m",  # Gris para REPOSO
    "\033[92m",  # Verde para PALMA
    "\033[91m",  # Rojo para PUÑO
    "\033[94m"   # Azul para PAZ
]
RESET = "\033[0m"

try:
    while True:
        data, addr = sock.recvfrom(1024)
        if len(data) == 10:  # 1 + 1 + 4 + 4 = 10 bytes
            try:
                # '<BBff' -> Little-endian, uint8_t, uint8_t, float, float
                filt_pred, raw_pred, mav, rms = struct.unpack("<BBff", data)
                
                name = GESTURES[filt_pred] if 0 <= filt_pred < len(GESTURES) else "DESCONOCIDO"
                color = COLORS[filt_pred] if 0 <= filt_pred < len(COLORS) else RESET
                
                # Imprimir cada lectura en una nueva línea para que el terminal se desplace
                print(
                    f"[WiFi] Desde {addr[0]:<15} | Gesto: {color}{name:<8}{RESET} (Crudo: {raw_pred}) | MAV: {mav:.4f} | RMS: {rms:.4f}"
                )
            except struct.error:
                # Ignorar paquetes mal formados
                pass
except KeyboardInterrupt:
    print("\n\nSaliendo del receptor UDP...")
    sock.close()
