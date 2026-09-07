#pragma once
/*
 * ============================================================
 *  config.h — Constantes globales del Firmware Threshold
 * ============================================================
 */

// ============================================================
// PINES SPI — MCP3208 en XIAO ESP32-S3
// ============================================================
#define PIN_SPI_SCK   7   // D8
#define PIN_SPI_MISO  8   // D9
#define PIN_SPI_MOSI  9   // D10
#define PIN_CS        2   // D1 — Chip Select MCP3208
#define PIN_LED       21  // LED de usuario en XIAO ESP32-S3

// ============================================================
// ADQUISICION EMG
// ============================================================
#define FS_HZ         1000
#define SAMPLE_US     (1000000 / FS_HZ) // 1000 us entre muestras
#define VREF_ADC      3.3f
#define ADC_RES       4095              // 12 bits MCP3208
#define DC_OFFSET     1837.0f           // V_REF ~1.48V → ~1837 cuentas
#define EMG_CHANNEL   0                 // Canal activo (0-7)

// ============================================================
// VENTANA DE ANALISIS Y UMBRALES
// ============================================================
#define WINDOW_SIZE     200   // 200 muestras = 200 ms
#define WINDOW_STRIDE   50    // 50 muestras = 50 ms (20 evaluaciones/seg)

// Calibración por defecto (persiste en memoria Flash NVS)
#define DEFAULT_MVC_VOLTAGE_V     189.845f  // MVC de referencia en cuentas ADC
#define DEFAULT_NOISE_THRESHOLD   20.0f     // Ruido basal típico MAV
#define DEFAULT_THRESHOLD_MAV     45.0f     // Umbral de activación MAV (cuentas ADC)
#define THRESHOLD_HYSTERESIS      0.80f     // Histéresis: abrir al caer por debajo del 80% del umbral

// Códigos de Gesto compatibles con Prosthesis_Controller
#define GESTURE_OPEN    0   // 0: Reposo / Mano Abierta
#define GESTURE_CLOSED  2   // 2: Puño / Mano Cerrada

// ============================================================
// FREERTOS — Configuracion de tareas
// ============================================================
#define TASK_ACQ_STACK     4096   // bytes — Core 1: DSP liviano @ 1kHz
#define TASK_THRESH_STACK  16384  // bytes — Core 0: Umbral + ESP-NOW + UDP
#define TASK_ACQ_PRIORITY  5      // alta  — deadline 1kHz crítico
#define TASK_THRESH_PRIORITY 3    // media — evaluación periódica
#define TASK_ACQ_CORE      1      // Core 1: adquisición
#define TASK_THRESH_CORE   0      // Core 0: control por umbral

// ============================================================
// WIFI / UDP — Telemetría y Calibración
// ============================================================
#define UDP_PORT           5005
#define UDP_BROADCAST_IP   "255.255.255.255"
#define WIFI_TIMEOUT_MS    15000  // 15 s máximo esperando conexión

// ============================================================
// SPSC RING BUFFER — Intercambio entre Core 1 → Core 0
// ============================================================
#define RING_SIZE   2048u          // debe ser potencia de 2
#define RING_MASK   (RING_SIZE-1)  // máscara para wrap de índices
