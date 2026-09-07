/*
 * ============================================================
 *  main.cpp — Classifier Firmware | XIAO Seeed ESP32-S3
 * ============================================================
 *  Modo: ADC + DSP + buffer circular PSRAM + Inferencia SVM (micromlgen)
 *
 *  Compilar y flashear:
 *    pio run -e classifier -t upload
 *
 *  Modulos:
 *    config.h  — constantes globales
 *    adc       — lectura MCP3208 via SPI
 *    dsp       — filtros IIR EMG (Notch, HPF, LPF)
 *    tasks     — FreeRTOS Core 0 (inferencia) + Core 1 (ADC/DSP)
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
  Serial.println("\n=== Classifier — MyoTensor S3 ===");

  // --- WiFi via WiFiManager ---
  WiFiManager wm;
  // wm.resetSettings();

  wm.setConfigPortalTimeout(180);
  Serial.println("[WiFi] Iniciando autoConnect...");
  if (!wm.autoConnect("MyoTensor_Classifier")) {
    Serial.println("[ERROR] Fallo la conexion o se alcanzo el timeout del portal");
    delay(3000);
    ESP.restart();
  }
  wm.stopWebPortal();
  Serial.printf("\n[OK] WiFi conectado a %s — IP: %s | Portal detenido\n",
                WiFi.SSID().c_str(), WiFi.localIP().toString().c_str());

  // Optimizaciones WiFi
  WiFi.setSleep(false);
  esp_wifi_set_ps(WIFI_PS_NONE);
  WiFi.setTxPower(WIFI_POWER_8_5dBm);
  Serial.println("[OK] WiFi Power Save desactivado y potencia ajustada a 8.5dBm");

  // --- Hardware ---
  adcInit();
  Serial.println("[OK] ADC MCP3208 (SPI) iniciado");

  // --- Inicializacion PSRAM y semaforos FreeRTOS ---
  if (!tasksInit()) {
    Serial.println("[FATAL] Fallo en tasksInit(). Sistema detenido.");
    while (true) delay(1000);
  }

  // Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  // for (uint8_t ch = 0; ch < 6; ch++) setServoAngle(ch, 90.0f);

  // --- Core 1: Adquisicion EMG + DSP ---
  xTaskCreatePinnedToCore(
    taskAcquisicion,
    "ACQ",
    TASK_ACQ_STACK,
    nullptr,
    TASK_ACQ_PRIORITY,
    nullptr,
    TASK_ACQ_CORE
  );

  // --- Core 0: Inferencia SVM ---
  xTaskCreatePinnedToCore(
    taskInferencia,
    "INF",
    TASK_INF_STACK,
    nullptr,
    TASK_INF_PRIORITY,
    &hTaskInferencia, // ← save handle para xTaskNotifyGive desde Core 1
    TASK_INF_CORE
  );

  // --- Resumen del sistema ---
  Serial.println("=== Dual-core FreeRTOS iniciado ===");
  Serial.printf("    Fs      : %d Hz  (periodo %d us)\n", FS_HZ, SAMPLE_US);
  Serial.printf("    Ventana : %d muestras = %d ms\n", WINDOW_SIZE, WINDOW_SIZE);
  Serial.printf("    Stride  : %d muestras = %d ms\n", WINDOW_STRIDE, WINDOW_STRIDE);
  Serial.println("    Core 1  : taskAcquisicion — ADC+DSP @ 1kHz → Ring Buffer");
  Serial.println("    Core 0  : taskInferencia  — SVM (micromlgen)");
}

// Toda la logica vive en las tasks — loop() no se usa
void loop() {
  vTaskDelete(nullptr);
}
