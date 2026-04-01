/*
 * ============================================================
 *  tasks.cpp — Tasks FreeRTOS y estado compartido entre cores
 * ============================================================
 *
 *  Modo Inferencia (default):
 *    Core 1 — taskAcquisicion : ADC + DSP a 1kHz → PSRAM + UDP
 *    Core 0 — taskInferencia  : [INACTIVA] esperando .tflite
 *
 *  Modo Dataset (-D DATASET_MODE):
 *    Core 1 — taskAcquisicion : ADC + DSP a 1kHz → FreeRTOS Queue
 *    Core 0 — taskUDP         : drena Queue → batch UDP broadcast
 *
 * ============================================================
 */

#include "tasks.h"
#include "config.h"
#include "adc.h"
#include "dsp.h"
#include <WiFiUdp.h>

// Socket UDP — definido aqui, compartido por ambos modos
WiFiUDP udp;

// ############################################################
// #                    MODO DATASET                          #
// ############################################################
#ifdef DATASET_MODE

// ============================================================
// Estado compartido — Queue de Core 1 → Core 0
// ============================================================
QueueHandle_t xEMGQueue = nullptr;

// ============================================================
// tasksInit — Crear Queue (sin PSRAM, sin semaforos)
// ============================================================
bool tasksInit() {
  xEMGQueue = xQueueCreate(DATASET_QUEUE_SIZE, sizeof(EMGSample));
  if (!xEMGQueue) {
    Serial.println("[ERROR] Fallo al crear xEMGQueue.");
    return false;
  }
  Serial.printf("[OK] Queue creada: %d slots x %d bytes = %d bytes\n",
                DATASET_QUEUE_SIZE, (int)sizeof(EMGSample),
                DATASET_QUEUE_SIZE * (int)sizeof(EMGSample));
  return true;
}

// ============================================================
// CORE 1 — taskAcquisicion (modo dataset)
// ============================================================
// Pipeline DSP identico al modo inferencia.
// En lugar de escribir a PSRAM, encola un EMGSample hacia Core 0.
// Si la cola esta llena, descarta silenciosamente (non-blocking).
//
// Formato CSV generado por Core 0 (taskUDP):
//   timestamp_us,raw,centered,filtered,voltage_v
// ============================================================
void taskAcquisicion(void* pvParameters) {
  TickType_t xLastWakeTime = xTaskGetTickCount();
  const TickType_t xPeriod = pdMS_TO_TICKS(1000 / FS_HZ);

  Serial.println("[ACQ] Core 1 — adquisicion dataset @ 1kHz");

  for (;;) {
    // 1. Leer ADC
    int   raw    = readADC(EMG_CHANNEL);
    float signal = (float)raw;

    // 2. Centrar en 0
    float centered = signal - DC_OFFSET;

    // 3. Pipeline DSP
    float notched  = notch(centered);
    float highpass = hpf(notched);
    float filtered = lpf(highpass);

    // 4. Tension (voltios)
    float voltage = (raw * VREF_ADC) / ADC_RES;

    // 5. Encolar hacia Core 0 (non-blocking)
    EMGSample sample = {
      .timestamp_us = (uint32_t)micros(),
      .raw          = raw,
      .centered     = centered,
      .filtered     = filtered,
      .voltage      = voltage
    };
    xQueueSend(xEMGQueue, &sample, 0);  // descarta si llena

    // 6. Esperar hasta el proximo tick — sin drift
    vTaskDelayUntil(&xLastWakeTime, xPeriod);
  }
}

// ============================================================
// CORE 0 — taskUDP (modo dataset)
// ============================================================
// Drena la Queue y acumula muestras en un buffer de texto CSV.
// Cada DATASET_UDP_BATCH muestras, envia un paquete UDP broadcast.
// Toda la carga WiFi/lwIP se ejecuta en Core 0, dejando Core 1
// libre para mantener el timing critico de 1kHz.
//
// CSV: timestamp_us,raw,centered,filtered,voltage_v
// ============================================================
void taskUDP(void* pvParameters) {
  // Buffer local para batch (~55 bytes por linea CSV)
  char batchBuf[DATASET_UDP_BATCH * 58];
  int  batchLen = 0;
  int  batchCnt = 0;

  Serial.println("[UDP] Core 0 — streaming dataset por UDP");

  for (;;) {
    EMGSample s;
    // Bloquea hasta que hay una muestra — no consume CPU en espera
    if (xQueueReceive(xEMGQueue, &s, portMAX_DELAY) == pdTRUE) {
      batchLen += snprintf(batchBuf + batchLen, sizeof(batchBuf) - batchLen,
                           "%lu,%d,%.2f,%.2f,%.4f\n",
                           (unsigned long)s.timestamp_us,
                           s.raw, s.centered, s.filtered, s.voltage);
      batchCnt++;

      if (batchCnt >= DATASET_UDP_BATCH) {
        udp.beginPacket(UDP_BROADCAST_IP, UDP_PORT);
        udp.write((const uint8_t*)batchBuf, batchLen);
        udp.endPacket();
        batchLen = 0;
        batchCnt = 0;
      }
    }
  }
}

// ############################################################
// #                   MODO INFERENCIA                        #
// ############################################################
#else

// ============================================================
// Estado compartido entre cores — definicion (modo inferencia)
// ============================================================
float*            circBuffer  = nullptr;
float*            inferBuf    = nullptr;
volatile int      writeIdx    = 0;
volatile int      strideCount = 0;
SemaphoreHandle_t xWindowReady;
SemaphoreHandle_t xBufMutex;

// ============================================================
// tasksInit — Alojar PSRAM e inicializar primitivas FreeRTOS
// ============================================================
bool tasksInit() {
  if (!psramFound()) {
    Serial.println("[ERROR] PSRAM no detectada.");
    Serial.println("        Verificar: board_build.arduino.memory_type = qio_opi");
    return false;
  }
  Serial.printf("[OK] PSRAM: %u bytes disponibles\n", ESP.getPsramSize());

  circBuffer = (float*) ps_malloc(WINDOW_SIZE * sizeof(float));
  inferBuf   = (float*) ps_malloc(WINDOW_SIZE * sizeof(float));
  if (!circBuffer || !inferBuf) {
    Serial.println("[ERROR] Fallo al alojar buffers en PSRAM");
    return false;
  }
  memset(circBuffer, 0, WINDOW_SIZE * sizeof(float));
  memset(inferBuf,   0, WINDOW_SIZE * sizeof(float));
  Serial.printf("[OK] Buffers PSRAM: %d bytes c/u\n", WINDOW_SIZE * (int)sizeof(float));

  xWindowReady = xSemaphoreCreateBinary();
  xBufMutex    = xSemaphoreCreateMutex();
  return true;
}

// ============================================================
// CORE 0 — taskInferencia  [INACTIVA — sin modelo .tflite]
// ============================================================
void taskInferencia(void* pvParameters) {
  Serial.println("[INF] Task inferencia en espera — modelo .tflite no cargado.");
  for (;;) {
    xSemaphoreTake(xWindowReady, portMAX_DELAY);
    vTaskDelay(pdMS_TO_TICKS(1));
  }
}

// ============================================================
// CORE 1 — taskAcquisicion (modo inferencia)
// ============================================================
#define N_UDP_BATCH    25
#define UDP_DECIMATION  1

void taskAcquisicion(void* pvParameters) {
  TickType_t xLastWakeTime = xTaskGetTickCount();
  const TickType_t xPeriod = pdMS_TO_TICKS(1000 / FS_HZ);

  char udpBatch[N_UDP_BATCH * 68];
  int  udpLen    = 0;
  int  batchCnt  = 0;
  int  decimCnt  = 0;

  Serial.println("# timestamp_us,raw,centered,filtered,rectified,voltage_v");

  for (;;) {
    // 1. Leer ADC
    int   raw      = readADC(EMG_CHANNEL);
    float signal   = (float)raw;

    // 2. Centrar en 0 (remover DC_OFFSET)
    float centered = signal - DC_OFFSET;

    // 3. Pipeline DSP
    float notched  = notch(centered);
    float highpass = hpf(notched);
    float filtered = lpf(highpass);

    // 4. Rectificacion (envolvente)
    float rectified = fabsf(filtered);

    // 5. Tension de referencia (voltios)
    float voltage = (raw * VREF_ADC) / ADC_RES;

    // 6. Escribir en buffer circular de PSRAM (non-blocking)
    if (xSemaphoreTake(xBufMutex, 0) == pdTRUE) {
      circBuffer[writeIdx] = rectified;
      writeIdx = (writeIdx + 1) % WINDOW_SIZE;
      xSemaphoreGive(xBufMutex);
    }

    // 7. Cada WINDOW_STRIDE muestras → despertar Core 0
    if (++strideCount >= WINDOW_STRIDE) {
      strideCount = 0;
      xSemaphoreGive(xWindowReady);
    }

    // 8. Acumular en batch
    if (++decimCnt >= UDP_DECIMATION) {
      decimCnt = 0;
      udpLen += snprintf(udpBatch + udpLen, sizeof(udpBatch) - udpLen,
                         "%lu,%d,%.2f,%.2f,%.2f,%.4f\n",
                         micros(), raw, centered, filtered, rectified, voltage);
      batchCnt++;

      // 9. Enviar batch
      if (batchCnt >= N_UDP_BATCH) {
        udp.beginPacket(UDP_BROADCAST_IP, UDP_PORT);
        udp.write((const uint8_t*)udpBatch, udpLen);
        udp.endPacket();
        udpLen   = 0;
        batchCnt = 0;
      }
    }

    // 10. Esperar hasta el proximo tick (1ms) — sin drift acumulado
    vTaskDelayUntil(&xLastWakeTime, xPeriod);
  }
}

#endif // DATASET_MODE
