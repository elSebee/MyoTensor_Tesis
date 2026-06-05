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

// ============================================================
// Elemento del ring buffer interno
// ============================================================
struct RingSample {
  uint32_t timestamp_us;   // 4 bytes — micros() en el momento de la muestra
  float    filtered;       // 4 bytes — señal post-DSP (Notch + HPF + LPF)
};                         // = 8 bytes por slot

// Estado compartido entre cores (SPSC lock-free)
extern RingSample        ring[];
extern volatile uint32_t ring_head;
extern volatile uint32_t ring_tail;
extern volatile uint32_t g_dropped;
extern float*            inferBuf;

// Handle de taskInferencia — usado por Core 1 para notificar via xTaskNotifyGive
extern TaskHandle_t hTaskInferencia;


// Inicializa PSRAM y primitivas FreeRTOS
// Llamar desde setup() antes de xTaskCreatePinnedToCore()
bool tasksInit();

// Core 1: adquisicion EMG + DSP a 1kHz
void taskAcquisicion(void* pvParameters);

// Core 0: inferencia TFLite sobre ventana de PSRAM
void taskInferencia(void* pvParameters);
