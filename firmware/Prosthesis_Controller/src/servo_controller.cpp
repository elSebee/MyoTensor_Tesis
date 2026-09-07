#include "servo_controller.h"
#include "config_receiver.h"
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

static Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA9685_I2C_ADDR);
static bool s_pca_ok = false;
static int s_current_gesture = -1;

bool servo_controller_init() {
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, 400000); // 400kHz Fast I2C
  
  Wire.beginTransmission(PCA9685_I2C_ADDR);
  if (Wire.endTransmission() != 0) {
    Serial.println("[ERROR] PCA9685 no detectado en 0x40. Revisa cables I2C y alimentacion!");
    s_pca_ok = false;
    return false;
  }

  pwm.begin();
  pwm.setPWMFreq(SERVO_FREQ_HZ);
  s_pca_ok = true;
  Serial.println("[OK] PCA9685 inicializado a 50Hz.");

  // Pose inicial: Reposo
  servo_set_pose(0);
  return true;
}

void servo_move(uint8_t channel, int angle_deg) {
  if (!s_pca_ok) return;
  angle_deg = constrain(angle_deg, 0, 180);
  int pulse = map(angle_deg, 0, 180, SERVOMIN, SERVOMAX);
  pwm.setPWM(channel, 0, pulse);
}

void servo_set_pose(int gesture_class) {
  if (gesture_class == s_current_gesture) return;
  s_current_gesture = gesture_class;

  const int *target = nullptr;
  switch (gesture_class) {
    case 0: target = pose_reposo; break;
    case 1: target = pose_palma;  break;
    case 2: target = pose_puno;   break;
    case 3: target = pose_paz;    break;
    default: target = pose_reposo; break;
  }

  if (target && s_pca_ok) {
    for (int i = 0; i < 6; i++) {
      servo_move(dedos[i], target[i]);
    }
  }
}
