# Firmware Threshold — MyoTensor (XIAO ESP32-S3)

Firmware de control mioeléctrico de prótesis basado en detección de umbral de activación (**Threshold / MAV**).

---

## 🏛️ Arquitectura Dual-Core FreeRTOS

```
┌─────────────────────────────────────────────────────────────┐
│                 NÚCLEO 1 — Tarea de Adquisición             │
│                                                             │
│  MCP3208 (SPI 1MHz)  ──>  DC Offset  ──>  Filtros IIR SIMD  │
│  (1000 Hz, Canal 0)       (1837)         (50/60Hz, 20-400Hz)│
│                                    │                        │
│                                    ▼                        │
│                         SPSC Ring Buffer (2048)             │
│                                    │                        │
│                        xTaskNotifyGive(Core 0)              │
└────────────────────────────────────┬────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────┐
│                   NÚCLEO 0 — Tarea de Umbral                │
│                                                             │
│  ulTaskNotifyTake()  ──>  Extracción MAV/RMS (Ventana 200)  │
│                                    │                        │
│                                    ▼                        │
│                          Evaluación de Umbral               │
│                        (Histéresis 80% al abrir)            │
│                                    │                        │
│                ┌───────────────────┼───────────────────┐    │
│                ▼                   ▼                   ▼    │
│         ESP-NOW (<1ms)         UDP (5005)          Serial   │
│     (Prosthesis_Controller)    (PC Monitor)        (Debug)  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Lógica de Decisión y Control

1. **Cálculo de MAV**:
   $$\text{MAV} = \frac{1}{N} \sum_{i=0}^{N-1} |x[i]|$$
2. **Histéresis Antirrebote**:
   - **Mano Abierta $\rightarrow$ Cerrada**: Requiere $\text{MAV} \ge \text{Umbral}$.
   - **Mano Cerrada $\rightarrow$ Abierta**: Requiere $\text{MAV} < \text{Umbral} \times 0.80$.
3. **Mapeo de Poses con `Prosthesis_Controller`**:
   - $\text{MAV} \ge \text{Umbral}$: Envía `GESTURE_CLOSED` (Clase 2: Puño) $\rightarrow$ Servos a $180^\circ$.
   - $\text{MAV} < \text{Umbral} \times 0.80$: Envía `GESTURE_OPEN` (Clase 0: Reposo) $\rightarrow$ Servos a $60^\circ$ o $0^\circ$.

---

## ⚙️ Calibración en Vivo y Persistencia en Flash (NVS)

El firmware almacena el umbral en la memoria Flash NVS del ESP32-S3 usando `Preferences`.

### Comandos Serial / Teclas Rápidas (Monitor @ 921600 baudios):
- `c` o `C`: Inicia la calibración rápida de 4 segundos (2s reposo + 2s MVC con ajuste automático del umbral).
- `r` o `R`: Restaura los valores de fábrica por defecto.
- `+`: Aumenta el umbral en +5 cuentas ADC.
- `-`: Reduce el umbral en -5 cuentas ADC.

### Comandos UDP (Puerto 5005):
- `CMD_CALIB`: Dispara la calibración de 4 segundos remotamente desde Python.
- `CMD_RESET`: Restaura la calibración de fábrica.

---

## 🚀 Compilación y Carga

```bash
# Compilar
pio run -d firmware/threshold -e threshold

# Flashear al XIAO ESP32-S3
pio run -d firmware/threshold -e threshold -t upload

# Abrir monitor serial
pio device monitor -b 921600
```
