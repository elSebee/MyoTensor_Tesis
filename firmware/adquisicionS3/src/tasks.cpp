/*
 * ============================================================
 *  tasks.cpp — Tasks FreeRTOS y estado compartido entre cores
 * ============================================================
 *  Core 1 — taskAcquisicion : ADC + DSP a 1kHz → Serial CSV
 *  Core 0 — taskInferencia  : [INACTIVA] esperando modelo .tflite
 * ============================================================
 */

#include "tasks.h"
#include "config.h"
#include "adc.h"
#include "dsp.h"

// ============================================================
// Estado compartido entre cores — definicion
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
// La task existe pero bloquea esperando el semaforo indefinidamente.
// No consume CPU. Se activa al integrar TFLite en la proxima etapa.
// Para reactivar: descomentar el bloque de inferencia y conectar
// el interprete de TFLite.
// ============================================================
void taskInferencia(void* pvParameters) {
  Serial.println("[INF] Task inferencia en espera — modelo .tflite no cargado.");
  for (;;) {
    // Drena el semaforo sin hacer nada — evita que se acumule
    xSemaphoreTake(xWindowReady, portMAX_DELAY);

    // [PLACEHOLDER] Reemplazar con:
    //   tflInterpreter->SetInput(inferBuf, WINDOW_SIZE);
    //   tflInterpreter->Invoke();
    //   int gesto = tflInterpreter->GetOutput();
    //   setServoAngle(gestureMap[gesto], angleMap[gesto]);

    // Por ahora solo libera el mutex rapidamente para no bloquear Core 1
    vTaskDelay(pdMS_TO_TICKS(1));
  }
}

// ============================================================
// CORE 1 — taskAcquisicion
// ============================================================
// Timing via vTaskDelayUntil() — mas preciso que polling con micros().
// El scheduler de FreeRTOS garantiza el periodo SAMPLE_US sin drift
// acumulado. El output CSV es parseado directamente por Python.
//
// Formato Serial (una linea por muestra):
//   timestamp_us,raw,centered,filtered,rectified,voltage_v
// ============================================================
void taskAcquisicion(void* pvParameters) {
  TickType_t xLastWakeTime = xTaskGetTickCount();

  // Periodo en ticks FreeRTOS (tick = 1ms por defecto en ESP-IDF)
  // A 1kHz, SAMPLE_US = 1000us = 1ms = exactamente 1 tick
  const TickType_t xPeriod = pdMS_TO_TICKS(1000 / FS_HZ);

  // Cabecera CSV — Python la usa para nombrar columnas
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

    // 8. Output CSV — timestamp, raw, centered, filtered, rect, voltios
    // Formato compacto para minimizar bytes en serial a 921600
    Serial.printf("%lu,%d,%.2f,%.2f,%.2f,%.4f\n",
                  micros(), raw, centered, filtered, rectified, voltage);

    // 9. Esperar hasta el proximo tick (1ms) — sin drift acumulado
    vTaskDelayUntil(&xLastWakeTime, xPeriod);
  }
}
