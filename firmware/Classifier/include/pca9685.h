#pragma once
/*
 * ============================================================
 *  pca9685.h — Interfaz del driver de servos PCA9685 [pendiente]
 * ============================================================
 *  Chip   : PCA9685 — 16 canales PWM, I2C
 *  Servos : MG996R (180°), pulso 500-2500 us, 50 Hz
 *
 *  Para activar:
 *    1. Descomentar #include <Wire.h> en main.cpp
 *    2. Descomentar los defines de config.h
 *    3. Llamar pca9685Init() en setup()
 *    4. Llamar setServoAngle() desde taskInferencia()
 * ============================================================
 */

// #include <Arduino.h>
// #include <Wire.h>
// #include "config.h"

// // Inicializa el PCA9685 a 50 Hz para control de servos MG996R
// void pca9685Init();

// // Mueve el servo del canal ch al angulo indicado (0.0 - 180.0 grados)
// void setServoAngle(uint8_t ch, float angle);
