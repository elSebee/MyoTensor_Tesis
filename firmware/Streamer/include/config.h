#pragma once
/*
 * ============================================================
 *  config.h — Constantes globales del Streamer
 * ============================================================
 *  Firmware: ADC + DSP + SPSC Ring Buffer + UDP binario
 *
 *  Protocolo UDP (binario, little-endian):
 *    Header (12 bytes):
 *      magic      : uint16 = 0xEB90
 *      n_samples  : uint8  = BATCH_SIZE
 *      reserved   : uint8  = 0x00
 *      seq_num    : uint32  (contador de paquete)
 *      timestamp0 : uint32  (µs de la primera muestra)
 *    Payload:
 *      float[BATCH_SIZE]  (señal filtrada, 4 bytes c/u)
 *
 *    Total: 12 + BATCH_SIZE×4 bytes
 *    Con BATCH_SIZE=50: 212 bytes → 20 pkt/s
 * ============================================================
 */

// ============================================================
// PINES — XIAO ESP32-S3
// ============================================================
#define PIN_SPI_SCK   7   // D8
#define PIN_SPI_MISO  8   // D9
#define PIN_SPI_MOSI  9   // D10
#define PIN_CS        2   // D1 — Chip Select MCP3208

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
// FREERTOS — Configuracion de tasks
// ============================================================
#define TASK_ACQ_STACK     4096   // bytes — DSP liviano
#define TASK_UDP_STACK     8192   // bytes — WiFi + lwIP
#define TASK_ACQ_PRIORITY  5      // alta  — deadline 1kHz critico
#define TASK_UDP_PRIORITY  3      // media — transmision no critica
#define TASK_ACQ_CORE      1      // Core 1: adquisicion
#define TASK_UDP_CORE      0      // Core 0: UDP streaming

// ============================================================
// WIFI / UDP — Streaming unicast hacia host Python
// ============================================================
//  IMPORTANTE: Usar unicast (IP del PC) en lugar de broadcast.
//  Broadcast fuerza al router a transmitir a velocidad basica
//  (1-2 Mbps), saturando el tiempo de aire. Unicast permite
//  negociar la maxima velocidad (54/150+ Mbps).
//
//  Configurar UDP_TARGET_IP en secrets.h con la IP de tu PC.
// ============================================================
#define UDP_PORT          5005
#define WIFI_TIMEOUT_MS   15000   // 15 s maximo esperando conexion

// ============================================================
// SPSC RING BUFFER — Intercambio entre Core 1 → Core 0
// ============================================================
//  Potencia de 2 obligatoria: permite usar mascara bit a bit
//  en lugar de modulo (%). Mas rapido y sin division.
//
//  2048 slots × 8 bytes = 16 KB en SRAM interna
//  Margen temporal: 2048 muestras / 1000 Hz ≈ 2 segundos
// ============================================================
#define RING_SIZE   2048u          // debe ser potencia de 2
#define RING_MASK   (RING_SIZE-1)  // mascara para wrap de indices

// ============================================================
// PROTOCOLO UDP BINARIO
// ============================================================
#define BATCH_SIZE    40u          // muestras por paquete (10ms → 100 pkt/s, fluido a 30 FPS)
#define PACKET_MAGIC  0xEB90u      // word de sincronismo
