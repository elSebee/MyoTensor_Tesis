/*
 * ============================================================
 *  main.cpp — Receptor ESP-NOW con Auto-Sintonizador de Canal
 *  Placa: ESP32-S3 DevKit + PCA9685
 * ============================================================
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include "config_receiver.h"
#include "servo_controller.h"

struct __attribute__((packed)) GesturePacket {
  uint8_t  filtered_prediction; // 0: Reposo, 1: Palma, 2: Puño, 3: Paz
  uint8_t  raw_prediction;
  float    mav;
  float    rms;
  uint32_t timestamp;
};

static volatile uint32_t s_last_packet_time = 0;
static uint32_t s_last_hop_time = 0;
static uint8_t s_current_channel = 1;
static bool s_channel_locked = false;
static const char *gesture_names[] = {"REPOSO", "PALMA", "PUNO", "PAZ"};

// Callback de Recepción ESP-NOW (Tiempo Real)
void on_esp_now_data_recv(const uint8_t *mac_addr, const uint8_t *incoming_data, int len) {
  if (len == sizeof(GesturePacket)) {
    GesturePacket packet;
    memcpy(&packet, incoming_data, sizeof(GesturePacket));
    s_last_packet_time = millis();

    // Bloquear el canal en cuanto se recibe el primer paquete válido
    if (!s_channel_locked) {
      s_channel_locked = true;
      Serial.printf("\n🎯 [ESP-NOW] ¡SINCRONIZADO Y BLOQUEADO EN CANAL %d!\n", s_current_channel);
    }

    // Mover servos a la pose clasificada
    servo_set_pose(packet.filtered_prediction);

    const char *name = (packet.filtered_prediction < 4) ? gesture_names[packet.filtered_prediction] : "DESC";
    Serial.printf("[ESP-NOW Ch%d] Gesto: %-7s (Crudo: %d) | MAV: %.4f | RMS: %.4f\n",
                  s_current_channel, name, packet.raw_prediction, packet.mav, packet.rms);
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n==========================================================");
  Serial.println("   MyoTensor — Receptor ESP-NOW con Auto-Sintonizador de Canal");
  Serial.println("   Placa: ESP32-S3 DevKit + Driver PCA9685");
  Serial.println("==========================================================");

  // 1. Inicializar I2C y Servos
  if (!servo_controller_init()) {
    Serial.println("[FATAL] Error en driver PCA9685!");
  }

  // 2. Inicializar WiFi en modo Estación sin conexión a router
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  // Desactivar ahorro de energía del módem (máxima sensibilidad de recepción)
  esp_wifi_set_ps(WIFI_PS_NONE);

  // Iniciar en Canal 1
  s_current_channel = 1;
  esp_wifi_set_channel(s_current_channel, WIFI_SECOND_CHAN_NONE);

  // 3. Inicializar ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("[FATAL] Fallo al inicializar ESP-NOW!");
    return;
  }

  // 4. Registrar Callback de Recepción
  esp_now_register_recv_cb(on_esp_now_data_recv);
  Serial.println("[OK] Receptor iniciado. Buscando canal del emisor...");
  s_last_hop_time = millis();
}

void loop() {
  uint32_t now = millis();

  // ── Auto-Sintonizador de Canal ──
  // Si no hemos recibido datos en 1.5 segundos, escanear canales 1 al 13
  if (now - s_last_packet_time > 1500) {
    if (s_channel_locked) {
      s_channel_locked = false;
      Serial.println("⚠️ [ESP-NOW] Señal perdida. Reanudando escaneo de canales...");
    }

    if (now - s_last_hop_time > 150) { // Saltar de canal cada 150 ms
      s_current_channel = (s_current_channel % 13) + 1;
      esp_wifi_set_channel(s_current_channel, WIFI_SECOND_CHAN_NONE);
      s_last_hop_time = now;
    }
  }

  // ── Watchdog de Seguridad (Failsafe) ──
  // Si no se reciben datos en 600 ms, regresar servos a reposo
  if (now - s_last_packet_time > FAILSAFE_TIMEOUT_MS) {
    servo_set_pose(0);
  }

  delay(20);
}
