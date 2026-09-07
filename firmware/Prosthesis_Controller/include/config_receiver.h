#pragma once
/*
 * ============================================================
 *  config_receiver.h — Configuración del Receptor y Servos
 *  ESP32-S3 DevKit + PCA9685
 * ============================================================
 */

#include <cstdint>

// ---------- Pines I2C en ESP32-S3 DevKit ----------
#define PIN_I2C_SDA      8   // GPIO 8 (Idéntico a tu test en DevKit)
#define PIN_I2C_SCL      9   // GPIO 9 (Idéntico a tu test en DevKit)

// ---------- Configuración PCA9685 ----------
#define PCA9685_I2C_ADDR 0x40
#define SERVO_FREQ_HZ    50  // 50Hz estándar
#define SERVOMIN         150 // ~0°
#define SERVOMAX         600 // ~180°

// ---------- Canales PCA9685 por Dedo ----------
const uint8_t PULGAR  = 0;
const uint8_t INDICE  = 1;
const uint8_t MEDIO   = 2;
const uint8_t ANULAR  = 3;
const uint8_t MENIQUE = 4;
const uint8_t DEDO6   = 5;

const uint8_t dedos[6] = {PULGAR, INDICE, MEDIO, ANULAR, MENIQUE, DEDO6};

// ---------- Poses Angulares por Gesto ----------
// [PULGAR, INDICE, MEDIO, ANULAR, MENIQUE, DEDO6]
const int pose_reposo[6] = { 60,  60,  60,  60,  60,  60}; // Relajado natural
const int pose_palma[6]  = {  0,   0,   0,   0,   0,   0}; // Abierto total
const int pose_puno[6]   = {180, 180, 180, 180, 180, 180}; // Cerrado total
const int pose_paz[6]    = {180,   0,   0, 180, 180, 180}; // V de Victoria

// ---------- Tiempo Límite de Seguridad (Failsafe) ----------
#define FAILSAFE_TIMEOUT_MS 600 // Si no recibe datos en 600ms, vuelve a reposo
