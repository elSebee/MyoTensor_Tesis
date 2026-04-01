/*
 * ============================================================
 *  main.cpp — MyoTensor Firmware | XIAO Seeed ESP32-S3
 * ============================================================
 *  Punto de entrada. Solo inicializa hardware y lanza tasks.
 *  Toda la logica vive en los modulos correspondientes.
 *
 *  Compilar con:
 *    pio run -e inferencia   → modo inferencia + servos
 *    pio run -e dataset      → modo recoleccion de datos
 *
 *  Modulos:
 *    config.h    — constantes globales
 *    adc         — lectura MCP3208 via SPI
 *    dsp         — filtros IIR EMG (Notch, HPF, LPF)
 *    tasks       — FreeRTOS Core 0 + Core 1
 *    pca9685     — driver servos MG996R via I2C [pendiente]
 * ============================================================
 */

#include <Arduino.h>
#include <WiFi.h>
#include "config.h"
#include "secrets.h"
#include "adc.h"
#include "tasks.h"
// #include "pca9685.h"  // [pendiente — solo modo inferencia]

void setup() {
  Serial.begin(921600);
  delay(500);

  #ifdef DATASET_MODE
    Serial.println("\n=== MyoTensor S3 — Modo DATASET ===");
  #else
    Serial.println("\n=== MyoTensor S3 — Modo INFERENCIA ===");
  #endif

  // --- Hardware ---
  adcInit();
  Serial.println("[OK] ADC MCP3208 (SPI) iniciado");

  // --- WiFi ---
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.printf("[WiFi] Conectando a '%s'", WIFI_SSID);
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED) {
    if (millis() - t0 > WIFI_TIMEOUT_MS) {
      Serial.println("\n[FATAL] WiFi: timeout. Verificar SSID/PASS en secrets.h");
      while (true) delay(1000);
    }
    delay(250);
    Serial.print(".");
  }
  Serial.printf("\n[OK] WiFi conectado — IP: %s\n", WiFi.localIP().toString().c_str());

  // --- UDP Broadcast ---
  udp.begin(UDP_PORT);
  Serial.printf("[OK] UDP broadcast listo en puerto %d\n", UDP_PORT);

  // --- Inicializacion FreeRTOS (Queue o PSRAM segun modo) ---
  if (!tasksInit()) {
    Serial.println("[FATAL] Fallo en tasksInit(). Sistema detenido.");
    while (true) delay(1000);
  }

  #ifndef DATASET_MODE
    // --- PCA9685 — solo en modo inferencia ---
    // Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
    // pca9685Init();
    // for (uint8_t ch = 0; ch < 6; ch++) setServoAngle(ch, 90.0f);
    // Serial.println("[OK] PCA9685 (I2C) iniciado");
  #endif

  // --- Lanzar task Core 1: Adquisicion EMG (comun a ambos modos) ---
  xTaskCreatePinnedToCore(
    taskAcquisicion,
    "ACQ",
    TASK_ACQ_STACK,
    nullptr,
    TASK_ACQ_PRIORITY,
    nullptr,
    TASK_ACQ_CORE
  );

  // --- Lanzar task Core 0 (distinta segun modo) ---
  #ifdef DATASET_MODE
    xTaskCreatePinnedToCore(
      taskUDP,
      "UDP",
      TASK_INF_STACK,       // reutiliza el stack reservado para Core 0
      nullptr,
      TASK_INF_PRIORITY,
      nullptr,
      TASK_INF_CORE
    );
  #else
    xTaskCreatePinnedToCore(
      taskInferencia,
      "INF",
      TASK_INF_STACK,
      nullptr,
      TASK_INF_PRIORITY,
      nullptr,
      TASK_INF_CORE
    );
  #endif

  // --- Resumen del sistema ---
  Serial.println("=== Dual-core FreeRTOS iniciado ===");
  Serial.printf("    Fs : %d Hz  (periodo %d us)\n", FS_HZ, SAMPLE_US);

  #ifdef DATASET_MODE
    Serial.printf("    Queue : %d slots x %d bytes\n",
                  DATASET_QUEUE_SIZE, (int)sizeof(EMGSample));
    Serial.println("    Core 0 : taskUDP — streaming dataset");
    Serial.println("    Core 1 : taskAcquisicion — ADC+DSP @ 1kHz");
    Serial.println("    CSV    : timestamp_us,raw,centered,filtered,voltage_v");
  #else
    Serial.printf("    Ventana : %d muestras = %d ms\n", WINDOW_SIZE, WINDOW_SIZE);
    Serial.printf("    Stride  : %d muestras = %d ms\n", WINDOW_STRIDE, WINDOW_STRIDE);
    Serial.println(">--- Core 0 (inferencia): en espera de modelo .tflite ---");
  #endif
}

// Toda la logica vive en las tasks — loop() no se usa
void loop() {
  vTaskDelete(nullptr);
}