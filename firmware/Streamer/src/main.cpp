/*
 * ============================================================
 *  main.cpp — Streamer Firmware | XIAO Seeed ESP32-S3
 * ============================================================
 *  Modo: ADC + DSP + SPSC Ring Buffer + UDP binario unicast
 *
 *  Compilar y flashear:
 *    pio run -e streamer -t upload
 *
 *  Optimizaciones de red:
 *    1. WiFi Power Save desactivado (radio siempre encendida)
 *    2. Unicast en lugar de broadcast (velocidad 54+ Mbps)
 *    3. Task Notifications en lugar de polling (zero CPU wait)
 * ============================================================
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiManager.h>
#include <esp_wifi.h>
#include "config.h"
#include "secrets.h"
#include "adc.h"
#include "tasks.h"

void setup() {
  Serial.begin(921600);
  delay(3000); // Dar tiempo al Monitor Serial para conectarse
  Serial.println("\n=== Streamer — MyoTensor S3 ===");

  // --- Hardware ---
  adcInit();
  Serial.println("[OK] ADC MCP3208 (SPI) iniciado");

  // --- WiFi via WiFiManager ---
  WiFiManager wm;
  wm.setConfigPortalTimeout(180);
  Serial.println("[WiFi] Iniciando autoConnect...");
  if (!wm.autoConnect("MyoTensor_Streamer")) {
    Serial.println("[ERROR] Fallo la conexion o se alcanzo el timeout del portal");
    delay(3000);
    ESP.restart();
  }
  wm.stopWebPortal();
  Serial.printf("\n[OK] WiFi conectado a %s — IP: %s | Portal detenido\n",
                WiFi.SSID().c_str(),
                WiFi.localIP().toString().c_str());

  // --- [FIX 1] Desactivar WiFi Power Save Mode ---
  // Por defecto, ESP32 usa WIFI_PS_MIN_MODEM: la radio se apaga
  // entre beacons del router para ahorrar bateria. Esto causa
  // picos de latencia de decenas de ms que desbordan lwIP.
  // Con PS_NONE, la radio queda encendida permanentemente.
  WiFi.setSleep(false);
  esp_wifi_set_ps(WIFI_PS_NONE);
  Serial.println("[OK] WiFi Power Save desactivado (WIFI_PS_NONE)");

  // --- [FIX 2] Reducir Potencia de Transmision ---
  // Unicast introduce ACKs MAC-level. Estos picos de transmision (hasta 300mA)
  // pueden causar caidas de voltaje en VCC e introducir "pulsos" en el ADC.
  // Reducir la potencia a 8.5dBm limita el ruido RF y el consumo pico.
  WiFi.setTxPower(WIFI_POWER_8_5dBm);
  Serial.println("[OK] Potencia de TX reducida a 8.5dBm para proteger ADC");

  // --- UDP unicast ---
  udp.begin(UDP_PORT);

  // --- Descubrimiento automatico de la IP del PC ---
  discoverTargetIP();

  Serial.printf("[OK] UDP unicast listo — destino: %s:%d\n", udp_target_ip.toString().c_str(), UDP_PORT);

  // --- Inicializar ring buffer SPSC ---
  if (!tasksInit()) {
    Serial.println("[FATAL] Fallo en tasksInit(). Sistema detenido.");
    while (true) delay(1000);
  }

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

  // --- Core 0: Streaming UDP (handle guardado para task notifications) ---
  xTaskCreatePinnedToCore(
    taskUDP,
    "UDP",
    TASK_UDP_STACK,
    nullptr,
    TASK_UDP_PRIORITY,
    &hTaskUDP,       // ← save handle para xTaskNotifyGive desde Core 1
    TASK_UDP_CORE
  );

  // --- Resumen del sistema ---
  const size_t pkt_bytes = sizeof(PacketHeader) + BATCH_SIZE * sizeof(float);
  Serial.println("=== Dual-core FreeRTOS iniciado ===");
  Serial.printf("    Fs        : %d Hz  (periodo %d us)\n", FS_HZ, SAMPLE_US);
  Serial.printf("    Ring      : %u slots x %u bytes = %u bytes SRAM\n",
                RING_SIZE, (unsigned)sizeof(RingSample),
                RING_SIZE * (unsigned)sizeof(RingSample));
  Serial.printf("    Paquete   : %u bytes (header=%u + payload=%u)\n",
                (unsigned)pkt_bytes,
                (unsigned)sizeof(PacketHeader),
                (unsigned)(BATCH_SIZE * sizeof(float)));
  Serial.printf("    Batch     : %u muestras -> %u pkt/s\n",
                BATCH_SIZE, (unsigned)(FS_HZ / BATCH_SIZE));
  Serial.printf("    Destino   : %s:%d (unicast)\n", udp_target_ip.toString().c_str(), UDP_PORT);
  Serial.println("    WiFi PS   : NONE (radio siempre activa)");
  Serial.println("    Core 1    : taskAcquisicion — ADC+DSP @ 1kHz");
  Serial.println("    Core 0    : taskUDP — binario + task notifications");
}

// Toda la logica vive en las tasks — loop() no se usa
void loop() {
  vTaskDelete(nullptr);
}