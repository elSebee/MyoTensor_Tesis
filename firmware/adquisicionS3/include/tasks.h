#pragma once
/*
 * ============================================================
 *  tasks.h — Interfaz de las tasks FreeRTOS y estado compartido
 * ============================================================
 *
 *  Modo Inferencia (default):
 *    Core 1 — taskAcquisicion: ADC+DSP → circBuffer PSRAM
 *    Core 0 — taskInferencia:  TFLite sobre ventana (pendiente)
 *
 *  Modo Dataset (-D DATASET_MODE):
 *    Core 1 — taskAcquisicion: ADC+DSP → FreeRTOS Queue
 *    Core 0 — taskUDP:         drena Queue → batch UDP
 *
 * ============================================================
 */

#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <WiFiUdp.h>

// Socket UDP broadcast — compartido por ambos modos
extern WiFiUDP udp;

// Inicializa primitivas FreeRTOS (Queue o PSRAM segun modo)
// Llamar desde setup() antes de xTaskCreatePinnedToCore()
bool tasksInit();

// Task Core 1: adquisicion EMG + DSP a 1kHz (comun a ambos modos)
void taskAcquisicion(void* pvParameters);

#ifdef DATASET_MODE
// ── Modo Dataset ─────────────────────────────────────────────
#include <freertos/queue.h>
extern QueueHandle_t xEMGQueue;

// Task Core 0: drena la queue y envia batch UDP
void taskUDP(void* pvParameters);

#else
// ── Modo Inferencia ──────────────────────────────────────────
extern float*             circBuffer;
extern float*             inferBuf;
extern volatile int       writeIdx;
extern volatile int       strideCount;
extern SemaphoreHandle_t  xWindowReady;
extern SemaphoreHandle_t  xBufMutex;

// Task Core 0: inferencia del modelo sobre ventana de PSRAM
void taskInferencia(void* pvParameters);

#endif
