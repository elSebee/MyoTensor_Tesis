/*
 * ============================================================
 *  pca9685.cpp — Driver de servos PCA9685 [pendiente]
 * ============================================================
 *  Para activar: descomentar todo el archivo y el .h
 * ============================================================
 */

// #include "pca9685.h"
// #include "config.h"

// void pca9685Write(uint8_t reg, uint8_t val) {
//   Wire.beginTransmission(PCA9685_ADDR);
//   Wire.write(reg);
//   Wire.write(val);
//   Wire.endTransmission();
// }

// void pca9685Init() {
//   pca9685Write(PCA9685_MODE1, 0x10);  // sleep
//   uint8_t pre = (uint8_t)(25000000.0f / (PCA_TICKS * SERVO_FREQ_HZ) - 0.5f);
//   pca9685Write(PCA9685_PRESCALE, pre);
//   pca9685Write(PCA9685_MODE1, 0xA0);  // wake + auto-increment
//   delay(5);
// }

// void setServoAngle(uint8_t ch, float angle) {
//   angle = constrain(angle, 0.0f, 180.0f);
//   float pulse_us = SERVO_MIN_US + (angle / 180.0f) * (SERVO_MAX_US - SERVO_MIN_US);
//   uint16_t ticks = (uint16_t)((pulse_us / 20000.0f) * PCA_TICKS);
//   uint8_t base = PCA9685_LED0_ON_L + 4 * ch;
//   Wire.beginTransmission(PCA9685_ADDR);
//   Wire.write(base);
//   Wire.write(0x00); Wire.write(0x00);         // ON en tick 0
//   Wire.write(ticks & 0xFF); Wire.write(ticks >> 8);  // OFF
//   Wire.endTransmission();
// }
