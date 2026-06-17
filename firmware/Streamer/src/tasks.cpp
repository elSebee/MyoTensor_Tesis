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
IPAddress udp_target_ip;

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
                
  dsp_init();
  Serial.println("[OK] Filtros DSP SIMD (esp_dsp) inicializados");
  return true;
}

// ============================================================
// discoverTargetIP — Descubre la IP del PC receptor via handshake UDP
// ============================================================
bool discoverTargetIP() {
  Serial.println("[Discovery] Iniciando auto-descubrimiento del PC...");
  
  // Configurar IP de fallback por si acaso
  if (!udp_target_ip.fromString(UDP_TARGET_IP)) {
    // Si no es un string valido de IP, por defecto usar broadcast general
    udp_target_ip = IPAddress(255, 255, 255, 255);
  }

  // Timeout de 15 segundos para buscar
  const uint32_t timeout_ms = 15000;
  uint32_t start_time = millis();
  uint32_t last_ping = 0;
  
  // Configurar LED para parpadeo rápido que indica búsqueda
  #ifdef LED_BUILTIN
  pinMode(LED_BUILTIN, OUTPUT);
  #endif

  while (millis() - start_time < timeout_ms) {
    uint32_t now = millis();
    
    // Parpadeo rápido (50ms ON, 50ms OFF)
    #ifdef LED_BUILTIN
    digitalWrite(LED_BUILTIN, (now / 100) % 2 == 0 ? LOW : HIGH); // LOW enciende el LED en Xiao
    #endif

    // Enviar PING broadcast cada 1 segundo
    if (now - last_ping >= 1000) {
      last_ping = now;
      Serial.printf("[Discovery] Enviando PING broadcast a puerto %d...\n", UDP_PORT);
      
      udp.beginPacket(IPAddress(255, 255, 255, 255), UDP_PORT);
      udp.write((const uint8_t*)"MYOTENSOR_PING", 14);
      udp.endPacket();
    }

    // Verificar si hay respuesta
    int packetSize = udp.parsePacket();
    if (packetSize > 0) {
      char reply[32] = {0};
      udp.read(reply, sizeof(reply) - 1);
      
      if (strcmp(reply, "MYOTENSOR_PONG") == 0) {
        udp_target_ip = udp.remoteIP();
        Serial.printf("[Discovery] ¡Exito! PC encontrado en IP: %s\n", udp_target_ip.toString().c_str());
        #ifdef LED_BUILTIN
        digitalWrite(LED_BUILTIN, LOW); // Dejar LED encendido fijo
        #endif
        return true;
      }
    }
    
    delay(10);
  }

  Serial.printf("[Discovery] Timeout alcanzado. Usando IP de fallback: %s\n", udp_target_ip.toString().c_str());
  #ifdef LED_BUILTIN
  digitalWrite(LED_BUILTIN, HIGH); // Apagar LED
  #endif
  return false;
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

  Serial.println("[ACQ] Core 1 — adquisicion vectorizada @ 1kHz");

  static float raw_batch[BATCH_SIZE];
  static uint32_t ts_batch[BATCH_SIZE];
  static int batch_idx = 0;

  for (;;) {
    // 1. Leer ADC y guardar en batch crudo
    int   raw      = readADC(EMG_CHANNEL);
    float centered = (float)raw - DC_OFFSET;
    
    raw_batch[batch_idx] = centered;
    ts_batch[batch_idx]  = (uint32_t)micros();
    batch_idx++;

    // 2. Si el batch esta completo, procesar y enviar al ring
    if (batch_idx == BATCH_SIZE) {
      float filtered_batch[BATCH_SIZE];
      
      // Aplicar pipeline DSP completo sobre el bloque usando SIMD
      dsp_process_block(raw_batch, filtered_batch, BATCH_SIZE);

      uint32_t h = ring_head;
      
      // Chequear que haya espacio para el batch completo en el ring
      if ((h - ring_tail) + BATCH_SIZE <= RING_SIZE) {
        for (int i = 0; i < BATCH_SIZE; i++) {
          ring[(h + i) & RING_MASK].timestamp_us = ts_batch[i];
          ring[(h + i) & RING_MASK].filtered     = filtered_batch[i];
        }

        // Barrera de memoria
        __asm__ volatile("" ::: "memory");
        ring_head = h + BATCH_SIZE;

        // Despertar taskUDP
        xTaskNotifyGive(hTaskUDP);
      } else {
        g_dropped += BATCH_SIZE;
      }
      
      batch_idx = 0;
    }

    // 3. Esperar hasta el proximo tick — sin drift acumulado
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
        udp.beginPacket(udp_target_ip, UDP_PORT);
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
