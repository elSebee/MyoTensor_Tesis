/*
 * ============================================================
 *  main.cpp — MyoTensor Firmware v2.0 | XIAO Seeed ESP32-S3
 * ============================================================
 *  Punto de entrada. Solo inicializa hardware y lanza tasks.
 *  Toda la logica vive en los modulos correspondientes.
 *
 *  Modulos:
 *    config.h    — constantes globales
 *    adc         — lectura MCP3208 via SPI
 *    dsp         — filtros IIR EMG (Notch, HPF, LPF)
 *    tasks       — FreeRTOS Core 0 (inferencia) + Core 1 (ADC)
 *    pca9685     — driver servos MG996R via I2C [pendiente]
 * ============================================================
 */

#include <Arduino.h>
#include "config.h"
#include "adc.h"
#include "tasks.h"
// #include "pca9685.h"  // [pendiente]

void setup() {
  Serial.begin(921600);
  while (!Serial) delay(10);

  Serial.println("\n=== MyoTensor S3 v2.0 — Iniciando ===");

  // --- Hardware ---
  adcInit();
  Serial.println("[OK] ADC MCP3208 (SPI) iniciado");

  // --- [PCA9685 — descomentar al conectar modulo de servos] ---
  // Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  // pca9685Init();
  // for (uint8_t ch = 0; ch < 6; ch++) setServoAngle(ch, 90.0f);
  // Serial.println("[OK] PCA9685 (I2C) iniciado");

  // --- PSRAM + FreeRTOS ---
  if (!tasksInit()) {
    Serial.println("[FATAL] Fallo en tasksInit(). Sistema detenido.");
    while (true) delay(1000);
  }

  // --- Lanzar task Core 1: Adquisicion EMG ---
  xTaskCreatePinnedToCore(
    taskAcquisicion,
    "ACQ",
    TASK_ACQ_STACK,
    nullptr,
    TASK_ACQ_PRIORITY,
    nullptr,
    TASK_ACQ_CORE
  );

  // --- Lanzar task Core 0: Inferencia ---
  xTaskCreatePinnedToCore(
    taskInferencia,
    "INF",
    TASK_INF_STACK,
    nullptr,
    TASK_INF_PRIORITY,
    nullptr,
    TASK_INF_CORE
  );

  Serial.println("=== Dual-core FreeRTOS iniciado ===");
  Serial.printf("    Fs      : %d Hz  (periodo %d us)\n",  FS_HZ, SAMPLE_US);
  Serial.printf("    Ventana : %d muestras = %d ms\n", WINDOW_SIZE,   WINDOW_SIZE);
  Serial.printf("    Stride  : %d muestras = %d ms\n", WINDOW_STRIDE, WINDOW_STRIDE);
  Serial.println(">--- CSV streaming @ 921600 baud ---");
  Serial.println(">--- Core 0 (inferencia): en espera de modelo .tflite ---");
}

// Toda la logica vive en las tasks — loop() no se usa
void loop() {
  vTaskDelete(nullptr);
}