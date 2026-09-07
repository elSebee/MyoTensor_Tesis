#pragma once
/*
 * ============================================================
 *  config.h — Constantes globales del proyecto MyoTensor S3
 * ============================================================
 */

#include "model_config.h"

// ============================================================
// PINES SPI — MCP3208 en XIAO ESP32-S3
// ============================================================
#define PIN_SPI_SCK   7   // D8
#define PIN_SPI_MISO  8   // D9
#define PIN_SPI_MOSI  9   // D10
#define PIN_CS        2   // D1 — Chip Select MCP3208
#define PIN_LED       21  // LED de usuario en XIAO ESP32-S3

// ============================================================
// PINES I2C — Driver PCA9685 de Servos en XIAO ESP32-S3
// ============================================================
#define PIN_I2C_SDA      5   // D4 en XIAO ESP32-S3
#define PIN_I2C_SCL      6   // D5 en XIAO ESP32-S3
#define PCA9685_I2C_ADDR 0x40
#define SERVO_FREQ_HZ    50
#define SERVOMIN         150 // Pulso para ~0°
#define SERVOMAX         600 // Pulso para ~180°

// ============================================================
// ADQUISICION EMG
// ============================================================
#define FS_HZ         1000
#define SAMPLE_US     (1000000 / FS_HZ) // 1000 us entre muestras
#define VREF_ADC      3.3f
#define ADC_RES       4095      // 12 bits MCP3208
#define DC_OFFSET     1837.0f   // V_REF ~1.48V → ~1837 cuentas
#define EMG_CHANNEL   0         // Canal activo (0-7)

// ============================================================
// PARAMETROS DINAMICOS DEL MODELO (Heredados de model_config.h)
// ============================================================
#define WINDOW_SIZE     MODEL_WINDOW_SIZE
#define WINDOW_STRIDE   MODEL_WINDOW_STRIDE
#define MVC_VOLTAGE_V   MODEL_MVC_VOLTAGE_V

// ============================================================
// FREERTOS — Configuracion de tareas
// ============================================================
#define TASK_ACQ_STACK     4096   // bytes — DSP liviano
#define TASK_INF_STACK     32768  // bytes — inferencia + servos
#define TASK_ACQ_PRIORITY  5      // alta  — deadline 1kHz critico
#define TASK_INF_PRIORITY  3      // media — inferencia
#define TASK_ACQ_CORE      1      // Core 1: adquisicion
#define TASK_INF_CORE      0      // Core 0: inferencia

// ============================================================
// WIFI / UDP — Streaming EMG y Comandos de Calibracion
// ============================================================
#define UDP_PORT           5005
#define UDP_BROADCAST_IP   "255.255.255.255"
#define WIFI_TIMEOUT_MS    15000  // 15 s maximo esperando conexion

// ============================================================
// SPSC RING BUFFER — Intercambio entre Core 1 → Core 0
// ============================================================
#define RING_SIZE   2048u          // debe ser potencia de 2
#define RING_MASK   (RING_SIZE-1)  // mascara para wrap de indices
