#pragma once
#include <cstdint>

bool servo_controller_init();
void servo_move(uint8_t channel, int angle_deg);
void servo_set_pose(int gesture_class);
