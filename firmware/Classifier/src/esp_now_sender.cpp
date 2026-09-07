/*
 * ============================================================
 *  esp_now_sender.cpp — Emisión ESP-NOW con Sintonización de Canal
 * ============================================================
 */

#include "esp_now_sender.h"
#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

static uint8_t s_broadcast_mac[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
static bool s_esp_now_ready = false;

bool esp_now_sender_init() {
  if (WiFi.getMode() == WIFI_OFF) {
    WiFi.mode(WIFI_STA);
  }

  // Desactivar ahorro de energía del módem (0% pérdidas de paquetes)
  esp_wifi_set_ps(WIFI_PS_NONE);

  if (esp_now_init() != ESP_OK) {
    Serial.println("[ERROR] Fallo al inicializar ESP-NOW!");
    s_esp_now_ready = false;
    return false;
  }

  uint8_t active_channel = WiFi.channel();
  if (active_channel == 0) active_channel = 1;

  // Registrar Peer Broadcast en el canal activo del radio
  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, s_broadcast_mac, 6);
  peerInfo.channel = active_channel;
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    // Si ya existía, modificar el canal
    esp_now_mod_peer(&peerInfo);
  }

  s_esp_now_ready = true;
  Serial.printf("[OK] Emisor ESP-NOW listo en Canal %d (MAC: %s)\n", 
                active_channel, WiFi.macAddress().c_str());
  return true;
}

void esp_now_send_gesture(uint8_t filtered_pred, uint8_t raw_pred, float mav, float rms) {
  if (!s_esp_now_ready) return;

  GesturePacket packet;
  packet.filtered_prediction = filtered_pred;
  packet.raw_prediction = raw_pred;
  packet.mav = mav;
  packet.rms = rms;
  packet.timestamp = millis();

  esp_now_send(s_broadcast_mac, (const uint8_t *)&packet, sizeof(packet));
}
