#pragma once
/*
 * ============================================================
 *  tasks.h — Interfaz de las tasks FreeRTOS y estado compartido
 * ============================================================
 *  Estado compartido entre cores (declarado en tasks.cpp):
 *    - circBuffer : buffer circular en PSRAM, escrito por Core 1
 *    - inferBuf   : snapshot de ventana en PSRAM, leido por Core 0
 *    - writeIdx   : indice actual de escritura en circBuffer
 *    - strideCount: contador de muestras hacia el proximo stride
 *    - xWindowReady: semaforo binario — Core 1 → Core 0
 *    - xBufMutex   : mutex para proteger la copia del buffer
 * ============================================================
 */

#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>

// Estado compartido entre tasks (definido en tasks.cpp)
extern float*             circBuffer;
extern float*             inferBuf;
extern volatile int       writeIdx;
extern volatile int       strideCount;
extern SemaphoreHandle_t  xWindowReady;
extern SemaphoreHandle_t  xBufMutex;

// Inicializa los buffers en PSRAM y las primitivas FreeRTOS
// Llamar desde setup() antes de xTaskCreatePinnedToCore()
bool tasksInit();

// Task Core 1: adquisicion EMG + DSP a 1kHz
void taskAcquisicion(void* pvParameters);

// Task Core 0: inferencia del modelo sobre ventana de PSRAM
void taskInferencia(void* pvParameters);
