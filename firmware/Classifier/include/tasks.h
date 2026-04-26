#pragma once
/*
 * ============================================================
 *  tasks.h — Interfaz de las tasks FreeRTOS (modo Inferencia)
 * ============================================================
 *
 *  Core 1 — taskAcquisicion: ADC + DSP → circBuffer PSRAM
 *  Core 0 — taskInferencia:  TFLite sobre ventana deslizante
 *
 *  Para streaming UDP de dataset, ver: firmware/Streamer
 * ============================================================
 */

#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>

// Estado compartido entre cores
extern float*            circBuffer;
extern float*            inferBuf;
extern volatile int      writeIdx;
extern volatile int      strideCount;
extern SemaphoreHandle_t xWindowReady;
extern SemaphoreHandle_t xBufMutex;

// Inicializa PSRAM y primitivas FreeRTOS
// Llamar desde setup() antes de xTaskCreatePinnedToCore()
bool tasksInit();

// Core 1: adquisicion EMG + DSP a 1kHz
void taskAcquisicion(void* pvParameters);

// Core 0: inferencia TFLite sobre ventana de PSRAM
void taskInferencia(void* pvParameters);
