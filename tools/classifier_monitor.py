import socket
import sys
import struct
import time
import select
import termios
import tty
import os
from datetime import datetime

UDP_PORT = 5005

# Socket UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
sock.setblocking(False)

try:
    sock.bind(("0.0.0.0", UDP_PORT))
except Exception as e:
    print(f"Error al abrir puerto UDP {UDP_PORT}: {e}")
    sys.exit(1)

# Enviar PING broadcast inicial
try:
    sock.sendto(b"MYOTENSOR_PING", ("255.255.255.255", UDP_PORT))
except Exception:
    pass

GESTURES = ["REPOSO", "PALMA", "PUNO", "PAZ"]
COLORS = [
    "\033[90m",  # Gris para REPOSO
    "\033[92m",  # Verde para PALMA
    "\033[91m",  # Rojo para PUÑO
    "\033[94m"   # Azul para PAZ
]
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"

def render_bar(val, max_val=0.8, width=15):
    val = max(0.0, min(val, max_val))
    filled = int((val / max_val) * width)
    return f"[{'=' * filled}{' ' * (width - filled)}]"

def set_nonblocking_terminal():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return fd, old_settings

print(f"{CYAN}{BOLD}=================================================================={RESET}")
print(f"{CYAN}{BOLD}   MyoTensor Monitor — Inferencia y Calibración Inalámbrica (UDP) {RESET}")
print(f"{CYAN}{BOLD}=================================================================={RESET}")
print(f"  [Teclas de Control]:")
print(f"    {YELLOW}{BOLD}[C]{RESET} : Iniciar Calibración Rápida On-Device (4 seg: Reposo + MVC)")
print(f"    {YELLOW}{BOLD}[R]{RESET} : Restaurar Calibración a valores de fábrica (dataset)")
print(f"    {YELLOW}{BOLD}[Q]{RESET} : Salir\n")
print(f"Escuchando transmisiones en puerto UDP {UDP_PORT}...\n")

fd, old_settings = set_nonblocking_terminal()
last_esp_ip = "255.255.255.255"

def run_calibration_flow(target_ip):
    print(f"\n{YELLOW}{BOLD}=================================================================={RESET}")
    print(f"{YELLOW}{BOLD}>>> INICIANDO CALIBRACIÓN INALÁMBRICA EN ESP32 ({target_ip}) <<<{RESET}")
    print(f"{YELLOW}{BOLD}=================================================================={RESET}")
    try:
        sock.sendto(b"CMD_CALIB", (target_ip, UDP_PORT))
    except Exception as e:
        print(f"Error enviando comando: {e}")
        return

    # Fase 1: 2s Reposo
    print(f"\n{CYAN}{BOLD}[FASE 1/2] MANTÉN EL BRAZO EN REPOSO ABSOLUTO (2 seg)...{RESET}")
    for sec in range(2, 0, -1):
        print(f"           Midiendo ruido basal... {sec}s")
        time.sleep(1.0)
        
    # Fase 2: 2s MVC
    print(f"\n{YELLOW}{BOLD}[FASE 2/2] ¡¡CONTRAE CON FUERZA MÁXIMA (PUÑO/PALMA)!! (2 seg)...{RESET}")
    for sec in range(2, 0, -1):
        print(f"           Midiendo voltaje pico MVC... {sec}s")
        time.sleep(1.0)

    # Drenar paquetes viejos acumulados en el socket durante el sleep
    while True:
        try:
            sock.recvfrom(1024)
        except BlockingIOError:
            break
        except Exception:
            break

    print(f"\n{GREEN}{BOLD}✅ ¡CALIBRACIÓN COMPLETADA Y GUARDADA EN FLASH!{RESET}")
    print(f"{CYAN}--- Reanudando inferencia en cascada ---{RESET}\n")

try:
    while True:
        # Esperar datos de red o tecla pulsada
        rlist, _, _ = select.select([sys.stdin, sock], [], [], 0.05)
        
        # 1. Teclas del usuario
        if sys.stdin in rlist:
            char = sys.stdin.read(1)
            if char.lower() == 'q':
                break
            elif char.lower() == 'c':
                run_calibration_flow(last_esp_ip)
            elif char.lower() == 'r':
                print(f"\n{YELLOW}[RESET] Restaurando calibración a valores de fábrica...{RESET}\n")
                sock.sendto(b"CMD_RESET", (last_esp_ip, UDP_PORT))

        # 2. Paquetes UDP del ESP32
        if sock in rlist:
            data, addr = sock.recvfrom(1024)
            last_esp_ip = addr[0]

            if data == b"MYOTENSOR_PING":
                sock.sendto(b"MYOTENSOR_PONG", addr)
                continue

            if len(data) == 10:
                filt_pred, raw_pred, mav, rms = struct.unpack("<BBff", data)
                name = GESTURES[filt_pred] if 0 <= filt_pred < len(GESTURES) else f"CLASE_{filt_pred}"
                color = COLORS[filt_pred] if 0 <= filt_pred < len(COLORS) else RESET
                bar = render_bar(mav, max_val=0.8, width=15)
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-4]
                
                # Impresión en cascada línea por línea
                print(f"[{ts}] {addr[0]:<14} | Gesto: {color}{BOLD}{name:<8}{RESET} (Crudo: {raw_pred}) | MAV: {mav:.4f} {bar} | RMS: {rms:.4f}")

except KeyboardInterrupt:
    pass
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    sock.close()
    print("\n\nSaliendo del monitor.")
