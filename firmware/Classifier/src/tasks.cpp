/*
 * ============================================================
 *  tasks.cpp — Tasks FreeRTOS (Modo Inferencia)
 * ============================================================
 *
 *  Core 1 — taskAcquisicion : ADC + DSP a 1kHz → buffer circular PSRAM
 *  Core 0 — taskInferencia  : TFLite sobre ventana deslizante (pendiente)
 *
 *  Para streaming UDP de dataset, ver: firmware/Streamer
 * ============================================================
 */

#include "tasks.h"
#include "config.h"
#include "adc.h"
#include "dsp.h"

// ============================================================
// Estado compartido entre cores
// ============================================================
float*            circBuffer  = nullptr;
float*            inferBuf    = nullptr;
volatile int      writeIdx    = 0;
volatile int      strideCount = 0;
SemaphoreHandle_t xWindowReady;
SemaphoreHandle_t xBufMutex;

// ============================================================
// tasksInit — Alojar buffers en PSRAM e inicializar semaforos
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
// CORE 1 — taskAcquisicion
// ============================================================
// Lee ADC a 1kHz, aplica pipeline DSP y escribe en buffer
// circular de PSRAM. Cada WINDOW_STRIDE muestras, señaliza
// a Core 0 para inferencia.
// ============================================================
void taskAcquisicion(void* pvParameters) {
  TickType_t xLastWakeTime = xTaskGetTickCount();
  const TickType_t xPeriod = pdMS_TO_TICKS(1000 / FS_HZ);

  Serial.println("[ACQ] Core 1 — adquisicion @ 1kHz");

  for (;;) {
    // 1. Leer ADC
    int   raw     = readADC(EMG_CHANNEL);
    float signal  = (float)raw;

    // 2. Centrar en 0 (remover DC_OFFSET)
    float centered = signal - DC_OFFSET;

    // 3. Pipeline DSP
    float notched  = notch(centered);
    float highpass = hpf(notched);
    float filtered = lpf(highpass);

    // 4. Rectificacion (envolvente)
    float rectified = fabsf(filtered);

    // 5. Escribir en buffer circular de PSRAM (non-blocking)
    if (xSemaphoreTake(xBufMutex, 0) == pdTRUE) {
      circBuffer[writeIdx] = rectified;
      writeIdx = (writeIdx + 1) % WINDOW_SIZE;
      xSemaphoreGive(xBufMutex);
    }

    // 6. Cada WINDOW_STRIDE muestras → despertar Core 0
    if (++strideCount >= WINDOW_STRIDE) {
      strideCount = 0;
      xSemaphoreGive(xWindowReady);
    }

    // 7. Esperar hasta el proximo tick (1ms) — sin drift acumulado
    vTaskDelayUntil(&xLastWakeTime, xPeriod);
  }
}

// ============================================================
// CORE 0 — taskInferencia  [pendiente — sin modelo .tflite]
// ============================================================
void taskInferencia(void* pvParameters) {
  Serial.println("[INF] Task inferencia en espera — modelo .tflite no cargado.");
  for (;;) {
    xSemaphoreTake(xWindowReady, portMAX_DELAY);
    // TODO: copiar ventana de circBuffer a inferBuf y correr TFLite
    vTaskDelay(pdMS_TO_TICKS(1));
  }
}
