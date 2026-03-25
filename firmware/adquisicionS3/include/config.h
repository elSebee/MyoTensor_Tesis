#pragma once
/*
 * ============================================================
 *  config.h — Constantes globales del proyecto MyoTensor S3
 * ============================================================
 *  Unico lugar donde ajustar parametros del sistema.
 *  Incluir en todos los modulos que necesiten estas constantes.
 * ============================================================
 */

// ============================================================
// PINES — XIAO ESP32-S3
// ============================================================
#define PIN_SPI_SCK   7   // D8
#define PIN_SPI_MISO  8   // D9
#define PIN_SPI_MOSI  9   // D10
#define PIN_CS        2   // D1 — Chip Select MCP3208

// #define PIN_I2C_SDA  5  // D4 [PCA9685 — pendiente]
// #define PIN_I2C_SCL  6  // D5 [PCA9685 — pendiente]

// ============================================================
// ADQUISICION EMG
// ============================================================
#define FS_HZ        1000
#define SAMPLE_US    (1000000 / FS_HZ)   // 1000 us entre muestras
#define VREF_ADC     3.3f
#define ADC_RES      4095                 // 12 bits MCP3208
#define DC_OFFSET    1837.0f             // V_REF ~1.48V → ~1837 cuentas
#define EMG_CHANNEL  0                   // Canal activo (0-7)

// ============================================================
// VENTANA DE INFERENCIA
// ------------------------------------------------------------
//  Ventana deslizante con 50% de solapamiento:
//    WINDOW_SIZE   = duracion de la ventana en muestras
//    WINDOW_STRIDE = paso entre ventanas en muestras
//  Con fs=1kHz: 200 muestras = 200ms | stride = 100ms
//  → Core 0 se activa ~10 veces por segundo
// ============================================================
#define WINDOW_SIZE    200   // muestras por ventana
#define WINDOW_STRIDE  100   // paso (50% overlap)

// ============================================================
// FREERTOS — Configuracion de tasks
// ============================================================
#define TASK_ACQ_STACK     4096   // bytes — DSP liviano
#define TASK_INF_STACK     8192   // bytes — reserva para TFLite
#define TASK_ACQ_PRIORITY  5      // alta  — deadline 1kHz critico
#define TASK_INF_PRIORITY  3      // media — inferencia no critica
#define TASK_ACQ_CORE      1      // Core 1: adquisicion
#define TASK_INF_CORE      0      // Core 0: inferencia

// ============================================================
// PCA9685 — Servo driver I2C [pendiente]
// ============================================================
// #define PCA9685_ADDR       0x40
// #define PCA9685_MODE1      0x00
// #define PCA9685_PRESCALE   0xFE
// #define PCA9685_LED0_ON_L  0x06
// #define SERVO_FREQ_HZ      50
// #define SERVO_MIN_US       500
// #define SERVO_MAX_US       2500
// #define PCA_TICKS          4096
