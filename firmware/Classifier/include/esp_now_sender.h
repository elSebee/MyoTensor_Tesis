#pragma once
/*
 * ============================================================
 *  esp_now_sender.h — Emisor ESP-NOW Inalámbrico (<1 ms)
 * ============================================================
 */

#include <cstdint>

// Estructura del paquete inalámbrico (14 bytes)
struct __attribute__((packed)) GesturePacket {
  uint8_t  filtered_prediction; // 0: Reposo, 1: Palma, 2: Puño, 3: Paz
  uint8_t  raw_prediction;      // Gesto crudo sin suavizado
  float    mav;                 // Amplitud media absoluta
  float    rms;                 // Energía eficaz
  uint32_t timestamp;           // Marca de tiempo en ms
};

// Inicializar ESP-NOW en modo broadcast
bool esp_now_sender_init();

// Transmitir predicción a todos los receptores en el canal
void esp_now_send_gesture(uint8_t filtered_pred, uint8_t raw_pred, float mav, float rms);
