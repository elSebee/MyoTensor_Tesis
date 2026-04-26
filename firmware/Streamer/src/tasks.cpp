/*
 * ============================================================
 *  tasks.cpp — Tasks FreeRTOS (Streamer)
 * ============================================================
 *
 *  Core 1 — taskAcquisicion : ADC + DSP @ 1kHz → SPSC ring buffer
 *  Core 0 — taskUDP         : ring → paquete binario → UDP unicast
 *
 *  Sincronizacion entre cores: Task Notifications (zero-overhead)
 *    Core 1 escribe al ring y llama xTaskNotifyGive(hTaskUDP)
 *    cuando hay BATCH_SIZE muestras listas.
 *    Core 0 duerme en ulTaskNotifyTake() sin consumir CPU.
 *
 *  Protocolo UDP (little-endian, 212 bytes con BATCH_SIZE=50):
 *    [magic:u16][n:u8][res:u8][seq:u32][ts0:u32] | [f0..fN-1 : float32]
 * ============================================================
 */

#include "tasks.h"
#include "config.h"
#include "secrets.h"
#include "adc.h"
#include "dsp.h"
#include <WiFiUdp.h>

// ============================================================
// Socket UDP
// ============================================================
WiFiUDP udp;

// Handle de taskUDP — asignado desde main.cpp
TaskHandle_t hTaskUDP = nullptr;

// ============================================================
// SPSC Ring Buffer — almacenado en SRAM interna
// ============================================================
RingSample        ring[RING_SIZE];
volatile uint32_t ring_head = 0;
volatile uint32_t ring_tail = 0;
volatile uint32_t g_dropped = 0;

// ============================================================
// tasksInit — Inicializar ring buffer
// ============================================================
bool tasksInit() {
  ring_head = 0;
  ring_tail = 0;
  g_dropped = 0;

  const size_t pkt_size = sizeof(PacketHeader) + BATCH_SIZE * sizeof(float);

  Serial.printf("[OK] Ring buffer : %u slots x %u bytes = %u bytes SRAM\n",
                RING_SIZE, (unsigned)sizeof(RingSample),
                RING_SIZE * (unsigned)sizeof(RingSample));
  Serial.printf("[OK] Paquete UDP : %u bytes (header=%u + payload=%u)\n",
                (unsigned)pkt_size,
                (unsigned)sizeof(PacketHeader),
                (unsigned)(BATCH_SIZE * sizeof(float)));
  return true;
}

// ============================================================
// CORE 1 — taskAcquisicion
// ============================================================
// Pipeline: ADC → centrar → Notch → HPF → LPF → ring buffer
//
// Politica ante ring lleno: DROP (descartar muestra nueva).
// El ring de 2048 slots da ~2s de margen para que Core 0
// se recupere de jitter WiFi sin perder datos.
//
// Cada vez que el nivel del ring cruza BATCH_SIZE, notifica
// a taskUDP via xTaskNotifyGive (costo ~0 CPU).
// ============================================================
void taskAcquisicion(void* pvParameters) {
  TickType_t xLastWakeTime = xTaskGetTickCount();
  const TickType_t xPeriod = pdMS_TO_TICKS(1000 / FS_HZ);

  Serial.println("[ACQ] Core 1 — adquisicion @ 1kHz");

  for (;;) {
    // 1. Leer ADC y aplicar pipeline DSP
    int   raw      = readADC(EMG_CHANNEL);
    float centered = (float)raw - DC_OFFSET;
    float filtered = lpf(hpf(notch(centered)));

    // 2. Escribir en ring si hay espacio (politica DROP si lleno)
    uint32_t h = ring_head;
    if ((h - ring_tail) < RING_SIZE) {
      ring[h & RING_MASK].timestamp_us = (uint32_t)micros();
      ring[h & RING_MASK].filtered     = filtered;

      // Barrera: asegurar que datos esten escritos antes de avanzar head
      __asm__ volatile("" ::: "memory");
      ring_head = h + 1;

      // 3. Cuando hay un batch completo → despertar taskUDP en Core 0
      //    xTaskNotifyGive es ISR-safe y no causa context switch en Core 1
      if (((h + 1) - ring_tail) >= BATCH_SIZE &&
          ((h)     - ring_tail) <  BATCH_SIZE) {
        // Justo cruzamos el umbral — notificar una vez
        xTaskNotifyGive(hTaskUDP);
      }
    } else {
      g_dropped++;
    }

    // 4. Esperar hasta el proximo tick — sin drift acumulado
    vTaskDelayUntil(&xLastWakeTime, xPeriod);
  }
}

// ============================================================
// CORE 0 — taskUDP
// ============================================================
// Duerme en ulTaskNotifyTake() hasta que Core 1 le avisa que
// hay BATCH_SIZE muestras. Cero polling, cero overhead.
//
// Paquete: header (12B) + float[BATCH_SIZE] (200B) = 212B
// Frecuencia: 1000/50 = 20 paquetes/segundo
// ============================================================
void taskUDP(void* pvParameters) {
  static uint32_t seq_num = 0;
  static uint32_t last_dropped = 0;

  // Buffer de paquete — 212 bytes
  static uint8_t pkt[sizeof(PacketHeader) + BATCH_SIZE * sizeof(float)];
  const size_t   PKT_SIZE = sizeof(pkt);

  PacketHeader* hdr     = reinterpret_cast<PacketHeader*>(pkt);
  float*        payload = reinterpret_cast<float*>(pkt + sizeof(PacketHeader));

  TickType_t xLastLog = xTaskGetTickCount();

  Serial.println("[UDP] Core 0 — streaming binario (task notifications)");

  for (;;) {
    // --- Dormir hasta que Core 1 notifique (zero CPU) ---
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    // --- Procesar todos los batches disponibles ---
    // Si Core 0 se atraso, puede haber >1 batch en el ring.
    while ((ring_head - ring_tail) >= BATCH_SIZE) {

      uint32_t t = ring_tail;

      // Construir header
      hdr->magic      = PACKET_MAGIC;
      hdr->n_samples  = BATCH_SIZE;
      hdr->reserved   = 0x00;
      hdr->seq_num    = seq_num++;
      hdr->timestamp0 = ring[t & RING_MASK].timestamp_us;

      // Copiar señal filtrada al payload
      for (uint32_t i = 0; i < BATCH_SIZE; i++) {
        payload[i] = ring[(t + i) & RING_MASK].filtered;
      }

      // Avanzar tail (liberar espacio en ring)
      __asm__ volatile("" ::: "memory");
      ring_tail = t + BATCH_SIZE;

      // Enviar paquete UDP (con reintentos ante ENOMEM)
      bool sent = false;
      for (int retry = 0; retry < 3 && !sent; retry++) {
        if (retry > 0) vTaskDelay(pdMS_TO_TICKS(2));
        udp.beginPacket(UDP_TARGET_IP, UDP_PORT);
        udp.write(pkt, PKT_SIZE);
        sent = (udp.endPacket() != 0);
      }

      if (!sent) {
        Serial.printf("[WARN] Paquete seq=%lu perdido (lwIP ENOMEM)\n",
                      (unsigned long)(seq_num - 1));
      }
    }

    // --- Log periodico cada 5s ---
    if ((xTaskGetTickCount() - xLastLog) >= pdMS_TO_TICKS(5000)) {
      uint32_t d = g_dropped;
      if (d != last_dropped) {
        Serial.printf("[WARN] Ring drops: %lu (+%lu en 5s) | libre: %lu/%u\n",
                      (unsigned long)d,
                      (unsigned long)(d - last_dropped),
                      (unsigned long)(RING_SIZE - (ring_head - ring_tail)),
                      RING_SIZE);
        last_dropped = d;
      }
      xLastLog = xTaskGetTickCount();
    }
  }
}
