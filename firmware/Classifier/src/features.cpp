#include "features.h"
#include "model_config.h"
#include "esp_dsp.h"
#include <cmath>

float compute_mav(const float *x, int w) {
  float sum = 0.0f;
  int i = 0;
  // Loop Unrolling 4x para maximizar IPC (Instructions Per Cycle)
  for (; i <= w - 4; i += 4) {
    sum += fabsf(x[i]) + fabsf(x[i+1]) + fabsf(x[i+2]) + fabsf(x[i+3]);
  }
  // Procesar las muestras restantes si 'w' no es múltiplo de 4
  for (; i < w; i++) {
    sum += fabsf(x[i]);
  }
  return sum / w;
}

float compute_rms(const float *x, int w) {
  float sum_sq = 0.0f;
  // Producto punto del vector consigo mismo usando SIMD (128-bit MAC)
  dsps_dotprod_f32(x, x, &sum_sq, w);
  return sqrtf(sum_sq / w);
}

float compute_wl(const float *x, int w) {
  float wl = 0.0f;
  int i = 0;
  // Loop Unrolling 4x
  for (; i <= w - 1 - 4; i += 4) {
    wl += fabsf(x[i + 1] - x[i]) + 
          fabsf(x[i + 2] - x[i + 1]) + 
          fabsf(x[i + 3] - x[i + 2]) + 
          fabsf(x[i + 4] - x[i + 3]);
  }
  for (; i < w - 1; i++) {
    wl += fabsf(x[i + 1] - x[i]);
  }
  return wl;
}

float compute_zc(const float *x, int w, float threshold) {
  int count = 0;
  int i = 0;
  // Loop Unrolling 4x
  for (; i <= w - 1 - 4; i += 4) {
    if ((x[i] * x[i + 1] < 0.0f) && (fabsf(x[i] - x[i + 1]) > threshold)) count++;
    if ((x[i+1] * x[i + 2] < 0.0f) && (fabsf(x[i+1] - x[i + 2]) > threshold)) count++;
    if ((x[i+2] * x[i + 3] < 0.0f) && (fabsf(x[i+2] - x[i + 3]) > threshold)) count++;
    if ((x[i+3] * x[i + 4] < 0.0f) && (fabsf(x[i+3] - x[i + 4]) > threshold)) count++;
  }
  for (; i < w - 1; i++) {
    if ((x[i] * x[i + 1] < 0.0f) && (fabsf(x[i] - x[i + 1]) > threshold)) count++;
  }
  return (float)count;
}

float compute_ssc(const float *x, int w, float threshold) {
  int count = 0;
  int i = 1;
  // Loop Unrolling 4x
  for (; i <= w - 1 - 4; i += 4) {
    float d1 = x[i] - x[i - 1]; float d2 = x[i + 1] - x[i];
    if ((d1 * d2 < 0.0f) && (fabsf(d1) > threshold) && (fabsf(d2) > threshold)) count++;
    
    d1 = x[i+1] - x[i]; d2 = x[i + 2] - x[i+1];
    if ((d1 * d2 < 0.0f) && (fabsf(d1) > threshold) && (fabsf(d2) > threshold)) count++;
    
    d1 = x[i+2] - x[i+1]; d2 = x[i + 3] - x[i+2];
    if ((d1 * d2 < 0.0f) && (fabsf(d1) > threshold) && (fabsf(d2) > threshold)) count++;
    
    d1 = x[i+3] - x[i+2]; d2 = x[i + 4] - x[i+3];
    if ((d1 * d2 < 0.0f) && (fabsf(d1) > threshold) && (fabsf(d2) > threshold)) count++;
  }
  for (; i < w - 1; i++) {
    float d1 = x[i] - x[i - 1];
    float d2 = x[i + 1] - x[i];
    if ((d1 * d2 < 0.0f) && (fabsf(d1) > threshold) && (fabsf(d2) > threshold)) count++;
  }
  return (float)count;
}

float compute_var(const float *x, int w) {
  float sum = 0.0f;
  int i = 0;
  for (; i <= w - 4; i += 4) {
    sum += x[i] + x[i+1] + x[i+2] + x[i+3];
  }
  for (; i < w; i++) sum += x[i];
  float mean = sum / w;
  
  float sum_sq = 0.0f;
  // Producto punto vectorial usando SIMD
  dsps_dotprod_f32(x, x, &sum_sq, w);
  
  // Fórmula optimizada O(1) de Varianza Poblacional idéntica a np.var(x):
  // Var = (Sum(X^2) - N * Mean^2) / N
  float var_val = (sum_sq - w * mean * mean) / w;
  return (var_val < 0.0f) ? 0.0f : var_val;
}

void scale_features(const float *raw_feats, float *scaled_feats) {
  for (int i = 0; i < 6; i++) {
    scaled_feats[i] = (raw_feats[i] - feature_means[i]) / feature_stds[i];
  }
}
