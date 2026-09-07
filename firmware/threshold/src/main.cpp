/*
 * ============================================================
 *  main.cpp — Threshold Firmware | XIAO Seeed ESP32-S3
 * ============================================================
 *  Modo: ADC + DSP + Buffer Circular SPSC + Control por Umbral (Threshold)
 *
 *  Compilar y flashear:
 *    pio run -e threshold -t upload
 *
 *  Módulos:
 *    config.h  — Constantes globales y pines
 *    adc       — Lectura MCP3208 vía SPI
 *    dsp       — Filtros IIR EMG (Notch 50/60Hz, HPF 20Hz, LPF 400Hz)
 *    tasks     — FreeRTOS Core 1 (Adquisición/DSP) + Core 0 (Umbral/Prótesis)
 * ============================================================
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiManager.h>
#include <esp_wifi.h>
#include "config.h"
#include "adc.h"
#include "tasks.h"
#include "secrets.h"

void setup() {
  Serial.begin(921600);
  delay(1000);
  Serial.println("\n=======================================================");
  Serial.println("   MyoTensor S3 — Firmware de Control por Umbral       ");
  Serial.println("=======================================================");

  // --- WiFi vía WiFiManager (Opcional para telemetría UDP) ---
  WiFiManager wm;
  wm.setConfigPortalTimeout(15); // Si no conecta en 15s, continúa sin WiFi (funciona 100% autónomo)
  Serial.println("[WiFi] Iniciando autoConnect (timeout 15s)...");
  
  if (!wm.autoConnect("MyoTensor_Threshold")) {
    Serial.println("[WARN] Portal WiFi timeout. Continuando en modo autónomo (ESP-NOW + Serial)...");
  } else {
    wm.stopWebPortal();
    Serial.printf("[OK] WiFi conectado a %s — IP: %s\n",
                  WiFi.SSID().c_str(), WiFi.localIP().toString().c_str());
  }

  // Optimizaciones WiFi (desactivar ahorro de energía)
  WiFi.setSleep(false);
  esp_wifi_set_ps(WIFI_PS_NONE);

  // --- Hardware ADC ---
  adcInit();
  Serial.println("[OK] ADC MCP3208 (SPI) iniciado");

  // --- Inicialización Memoria, DSP, Parámetros NVS y ESP-NOW ---
  if (!tasksInit()) {
    Serial.println("[FATAL] Fallo en tasksInit(). Sistema detenido.");
    while (true) delay(1000);
  }

  // --- Core 1: Adquisición EMG + DSP @ 1kHz ---
  xTaskCreatePinnedToCore(
    taskAcquisicion,
    "ACQ",
    TASK_ACQ_STACK,
    nullptr,
    TASK_ACQ_PRIORITY,
    nullptr,
    TASK_ACQ_CORE
  );

  // --- Core 0: Evaluación de Umbral + Control Prótesis ESP-NOW ---
  xTaskCreatePinnedToCore(
    taskThreshold,
    "THRESH",
    TASK_THRESH_STACK,
    nullptr,
    TASK_THRESH_PRIORITY,
    &hTaskThreshold, // Guardar handle para xTaskNotifyGive desde Core 1
    TASK_THRESH_CORE
  );

  // --- Resumen del sistema ---
  Serial.println("\n=== Sistema FreeRTOS Dual-Core Iniciado ===");
  Serial.printf("    Fs      : %d Hz (periodo %d us)\n", FS_HZ, SAMPLE_US);
  Serial.printf("    Ventana : %d muestras (%d ms)\n", WINDOW_SIZE, WINDOW_SIZE);
  Serial.printf("    Stride  : %d muestras (%d ms)\n", WINDOW_STRIDE, WINDOW_STRIDE);
  Serial.println("    Core 1  : taskAcquisicion — ADC+DSP SIMD @ 1kHz → Ring Buffer");
  Serial.println("    Core 0  : taskThreshold  — MAV vs Umbral → ESP-NOW / UDP / Serial");
  Serial.println("\n[Teclas Serial]: 'c' = Calibrar (4s) | 'r' = Reset | '+' = Subir Umbral | '-' = Bajar Umbral\n");
}

// Toda la lógica vive en las tareas FreeRTOS
void loop() {
  vTaskDelete(nullptr);
}
