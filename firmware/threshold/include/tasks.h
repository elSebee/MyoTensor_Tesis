#pragma once
/*
 * ============================================================
 *  tasks.h — Interfaz de Tareas FreeRTOS (Firmware Threshold)
 * ============================================================
 *
 *  Core 1 — taskAcquisicion: ADC (MCP3208 SPI) + DSP IIR @ 1kHz → SPSC Ring Buffer
 *  Core 0 — taskThreshold  : Extracción MAV/RMS + Detección de Umbral + ESP-NOW + UDP
 * ============================================================
 */

#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>

// ============================================================
// Elemento del Ring Buffer Interno
// ============================================================
struct RingSample {
  uint32_t timestamp_us;   // 4 bytes — micros() en el momento de la muestra
  float    filtered;       // 4 bytes — señal post-DSP (Notch + HPF + LPF)
};

// Estado compartido entre cores (SPSC lock-free)
extern RingSample        ring[];
extern volatile uint32_t ring_head;
extern volatile uint32_t ring_tail;
extern volatile uint32_t g_dropped;
extern float*            windowBuf;

// Handle de la tarea en Core 0
extern TaskHandle_t hTaskThreshold;

// Parámetros de Calibración / Umbral activos
extern float g_active_mvc;
extern float g_active_noise;
extern float g_active_threshold;

// Inicializa memoria, PSRAM, DSP y ESP-NOW
bool tasksInit();

// Dispara la máquina de calibración en vivo (2s reposo + 2s contracción)
void trigger_calibration();

// Core 1: adquisición EMG + DSP a 1kHz
void taskAcquisicion(void* pvParameters);

// Core 0: lógica de umbral para abrir/cerrar prótesis
void taskThreshold(void* pvParameters);
