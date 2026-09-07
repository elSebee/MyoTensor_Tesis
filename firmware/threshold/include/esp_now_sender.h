#pragma once
/*
 * ============================================================
 *  esp_now_sender.h — Emisor ESP-NOW Inalámbrico (<1 ms)
 * ============================================================
 */

#include <cstdint>

// Estructura del paquete inalámbrico compatible con Prosthesis_Controller (14 bytes)
struct __attribute__((packed)) GesturePacket {
  uint8_t  filtered_prediction; // 0: Reposo/Abierto, 2: Puño/Cerrado
  uint8_t  raw_prediction;      // Estado crudo antes de histéresis
  float    mav;                 // Amplitud Media Absoluta (MAV)
  float    rms;                 // Energía Eficaz (RMS)
  uint32_t timestamp;           // Marca de tiempo en ms
};

// Inicializar ESP-NOW en modo broadcast
bool esp_now_sender_init();

// Transmitir predicción/estado a todos los receptores en el canal
void esp_now_send_gesture(uint8_t filtered_pred, uint8_t raw_pred, float mav, float rms);
