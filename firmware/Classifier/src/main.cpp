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
 *    pca9685   — driver servos MG996R via I2C [pendiente]
 * ============================================================
 */

#include <Arduino.h>
#include "config.h"
#include "adc.h"
#include "tasks.h"
// #include "pca9685.h"  // [pendiente — control de servos]

void setup() {
  Serial.begin(921600);
  delay(500);
  Serial.println("\n=== Classifier — MyoTensor S3 ===");

  // --- Hardware ---
  adcInit();
  Serial.println("[OK] ADC MCP3208 (SPI) iniciado");

  // --- Inicializacion PSRAM y semaforos FreeRTOS ---
  if (!tasksInit()) {
    Serial.println("[FATAL] Fallo en tasksInit(). Sistema detenido.");
    while (true) delay(1000);
  }

  // --- PCA9685 — control de servos [pendiente] ---
  // Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  // pca9685Init();
  // for (uint8_t ch = 0; ch < 6; ch++) setServoAngle(ch, 90.0f);
  // Serial.println("[OK] PCA9685 (I2C) iniciado");

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