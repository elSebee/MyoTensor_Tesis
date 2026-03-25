# MyoTensor — Tesis de Título

> Sistema de adquisición y clasificación de señales EMG para control de prótesis de mano
> mediante una red neuronal BiLSTM embebida en un ESP32-S3.

---

## Estructura del proyecto

```
MyoTensor_Tesis/
│
├── firmware/                        # Código C++ para microcontroladores (PlatformIO)
│   └── adquisicionS3/               # Firmware principal — XIAO ESP32-S3
│       ├── src/                     # Lógica: main, tasks, adc, dsp
│       ├── include/                 # Cabeceras y constantes globales (config.h)
│       └── platformio.ini           # Configuración de placa y dependencias
│
├── intelligence/                    # Deep Learning — Python
│   ├── datasets/
│   │   ├── raw/                     # Grabaciones originales NinaPro (no versionado)
│   │   └── processed/               # Arrays X/y listos para entrenar (no versionado)
│   ├── notebooks/                   # Jupyter Notebooks de experimentación
│   │   ├── explorar_ninapro.ipynb   # Análisis exploratorio del dataset
│   │   ├── procesar_ninapro.ipynb   # Pipeline de preprocesamiento
│   │   ├── entrenamiento_NN.ipynb   # Entrenamiento v1
│   │   └── entrenamiento_NN_v2.ipynb# Entrenamiento v2 (actual)
│   └── models/                      # Modelos entrenados (.h5, .tflite)
│
├── tools/                           # Herramientas de desarrollo y monitoreo
│   ├── emg_monitor.py               # Monitor EMG en tiempo real (pyqtgraph + Serial)
│   └── requirements.txt             # Dependencias Python del proyecto
│
├── hardware/                        # Diseño del circuito analógico
│   ├── schematics/                  # PDFs de esquemáticos
│   ├── gerbers/                     # Archivos para fabricación de PCB
│   ├── easyeda_source/              # Fuentes del proyecto EasyEDA
│   └── datasheets/                  # Hojas de datos de componentes
│
├── docs/                            # Documentación de la tesis
│   └── images/                      # Figuras, capturas y gráficos
│
├── .gitignore
└── README.md
```

---

## Stack tecnológico

| Capa | Tecnología | Rol |
|---|---|---|
| Sensor | OPA333 + electrodos EMG | Amplificación de señal biológica |
| ADC | MCP3208 (12 bits, SPI) | Conversión analógico-digital |
| Microcontrolador | XIAO ESP32-S3 | Adquisición, DSP y (futuro) inferencia |
| Firmware | C++ / PlatformIO / FreeRTOS | Control dual-core, filtros IIR |
| Monitor | Python / pyqtgraph / pyserial | Visualización en tiempo real vía Serial |
| Deep Learning | PyTorch / Jupyter | Clasificación de gestos (BiLSTM) |
| Dataset | NinaPro DB5 | Datos EMG de referencia para entrenamiento |

---

## Firmware — pipeline de señal

```
Electrodo → OPA333 → MCP3208 ──SPI──► ESP32-S3 (Core 1 @ 1kHz)
                                          │
                                     readADC()
                                     Notch 50Hz → HPF 20Hz → LPF 450Hz
                                     Rectificación
                                          │
                                     Serial CSV @ 921600 baud
                                          │
                                     emg_monitor.py → gráfico en tiempo real
```

---

## Herramientas

### Monitor EMG en tiempo real

```bash
cd tools/
pip install -r requirements.txt

# Auto-detección de puerto:
python emg_monitor.py

# Puerto manual:
python emg_monitor.py --port /dev/ttyACM0
```

**Formato CSV esperado del firmware:**
```
# timestamp_us,raw,centered,filtered,rectified,voltage_v
1500000,2048,210.50,18.34,15.21,1.6511
```

---

## Estado del proyecto

- [x] Circuito analógico EMG (OPA333, single-supply)
- [x] Firmware adquisición 1kHz con filtros IIR (Notch + HPF + LPF)
- [x] Monitor en tiempo real Serial→Python
- [x] Exploración y preprocesamiento NinaPro DB5
- [x] Entrenamiento BiLSTM v1 y v2
- [ ] Migrar comunicación Serial → WiFi (UDP)
- [ ] Integrar modelo .tflite en ESP32-S3 (TFLite Micro)
- [ ] Control de servos vía PCA9685 según gesto clasificado