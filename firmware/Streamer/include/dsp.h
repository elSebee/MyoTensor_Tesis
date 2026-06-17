#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void dsp_init();
void dsp_process_block(const float *input, float *output, int len);

#ifdef __cplusplus
}
#endif
