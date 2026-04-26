#pragma once
/*
 * ============================================================
 *  tasks.h — Interfaz de tasks y estructuras de datos (Streamer)
 * ============================================================
 *
 *  Core 1 — taskAcquisicion: ADC+DSP → SPSC ring buffer
 *  Core 0 — taskUDP:         lee ring → paquete binario → UDP
 *
 *  SPSC = Single Producer (Core 1) / Single Consumer (Core 0)
 *  → lock-free: ring_head solo lo escribe Core 1
 *               ring_tail solo lo escribe Core 0
 * ============================================================
 */

#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <WiFiUdp.h>
#include "config.h"

// ============================================================
// Elemento del ring buffer interno
// ============================================================
// Solo se almacena lo necesario: timestamp + señal filtrada.
// El timestamp de las muestras siguientes se reconstruye en
// Python como: timestamp_k = timestamp_0 + k * SAMPLE_US
// ============================================================
struct RingSample {
  uint32_t timestamp_us;   // 4 bytes — micros() en el momento de la muestra
  float    filtered;       // 4 bytes — señal post-DSP (Notch + HPF + LPF)
};                         // = 8 bytes por slot

// ============================================================
// Cabecera del paquete UDP (12 bytes, little-endian)
// ============================================================
// __attribute__((packed)) garantiza que no haya padding entre campos.
// El payload que sigue es float[n_samples] de la señal filtrada.
// ============================================================
struct __attribute__((packed)) PacketHeader {
  uint16_t magic;       // 0xEB90 — sincronismo, descartar paquetes ajenos
  uint8_t  n_samples;   // cantidad de muestras en este paquete (= BATCH_SIZE)
  uint8_t  reserved;    // 0x00 — alineacion / uso futuro (ej: ID canal)
  uint32_t seq_num;     // contador de paquetes — detectar perdida en Python
  uint32_t timestamp0;  // µs de la primera muestra del batch
};                      // = 12 bytes

// ============================================================
// Ring buffer SPSC — definido en tasks.cpp
// ============================================================
extern RingSample        ring[];
extern volatile uint32_t ring_head;   // escrito solo por Core 1
extern volatile uint32_t ring_tail;   // escrito solo por Core 0
extern volatile uint32_t g_dropped;   // muestras descartadas (ring lleno)

// Socket UDP — definido en tasks.cpp
extern WiFiUDP udp;

// Handle de taskUDP — usado por Core 1 para notificar via xTaskNotifyGive
extern TaskHandle_t hTaskUDP;

// Inicializa el ring buffer
bool tasksInit();

// Core 1: adquisicion EMG + DSP a 1kHz → ring buffer
void taskAcquisicion(void* pvParameters);

// Core 0: drena ring buffer → paquete binario → UDP unicast
void taskUDP(void* pvParameters);
