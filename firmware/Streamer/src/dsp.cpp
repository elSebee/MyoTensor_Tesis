#include "dsp.h"
#include "esp_dsp.h"

// Coeficientes: {b0, b1, b2, a1, a2}
// Formula esp_dsp: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] + a1*y[n-1] + a2*y[n-2]

// NOTCH 50 Hz
static float coeffs_notch50[5] = {0.9949f, -1.8924f, 0.9949f, -1.8924f, 0.9898f};
static float state_notch50[2] = {0.0f, 0.0f};

// NOTCH 60 Hz
static float coeffs_notch60[5] = {0.9939f, -1.8482f, 0.9939f, -1.8482f, 0.9878f};
static float state_notch60[2] = {0.0f, 0.0f};

// HPF 20 Hz
static float coeffs_hpf20[5] = {0.9565f, -1.9131f, 0.9565f, -1.9112f, 0.9150f};
static float state_hpf20[2] = {0.0f, 0.0f};

// LPF 500 Hz
static float coeffs_lpf500[5] = {0.2929f, 0.5858f, 0.2929f, 0.0f, 0.0f};
static float state_lpf500[2] = {0.0f, 0.0f};

void dsp_init() {
    // Inicializar estados a cero
    for(int i=0; i<2; i++) {
        state_notch50[i] = 0.0f;
        state_notch60[i] = 0.0f;
        state_hpf20[i] = 0.0f;
        state_lpf500[i] = 0.0f;
    }
}

// Aplica el pipeline DSP completo sobre un bloque usando SIMD
void dsp_process_block(const float *input, float *output, int len) {
    // Buffer intermedio para la cascada (alojado en stack, rápido)
    float temp1[len];
    float temp2[len];
    
    // 1. Notch 50 Hz
    dsps_biquad_f32_ae32(input, temp1, len, coeffs_notch50, state_notch50);
    
    // 2. Notch 60 Hz
    dsps_biquad_f32_ae32(temp1, temp2, len, coeffs_notch60, state_notch60);
    
    // 3. HPF 20 Hz
    dsps_biquad_f32_ae32(temp2, temp1, len, coeffs_hpf20, state_hpf20);
    
    // 4. LPF 500 Hz
    dsps_biquad_f32_ae32(temp1, output, len, coeffs_lpf500, state_lpf500);
}
