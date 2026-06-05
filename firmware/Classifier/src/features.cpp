#include "features.h"
#include "scaler_params.h"
#include <cmath>

float compute_mav(const float *x, int w) {
  float sum = 0.0f;
  for (int i = 0; i < w; i++) {
    sum += fabsf(x[i]);
  }
  return sum / w;
}

float compute_rms(const float *x, int w) {
  float sum_sq = 0.0f;
  for (int i = 0; i < w; i++) {
    sum_sq += x[i] * x[i];
  }
  return sqrtf(sum_sq / w);
}

float compute_wl(const float *x, int w) {
  float wl = 0.0f;
  for (int i = 0; i < w - 1; i++) {
    wl += fabsf(x[i + 1] - x[i]);
  }
  return wl;
}

float compute_zc(const float *x, int w, float threshold) {
  int count = 0;
  for (int i = 0; i < w - 1; i++) {
    bool sign_change = (x[i] * x[i + 1] < 0.0f);
    bool diff_above = (fabsf(x[i] - x[i + 1]) > threshold);
    if (sign_change && diff_above) {
      count++;
    }
  }
  return (float)count;
}

float compute_ssc(const float *x, int w, float threshold) {
  int count = 0;
  for (int i = 1; i < w - 1; i++) {
    float d1 = x[i] - x[i - 1];
    float d2 = x[i + 1] - x[i];
    bool slope_change = (d1 * d2 < 0.0f);
    bool diff_above = (fabsf(d1) > threshold) && (fabsf(d2) > threshold);
    if (slope_change && diff_above) {
      count++;
    }
  }
  return (float)count;
}

float compute_var(const float *x, int w) {
  float sum = 0.0f;
  for (int i = 0; i < w; i++) {
    sum += x[i];
  }
  float mean = sum / w;
  float sum_diff_sq = 0.0f;
  for (int i = 0; i < w; i++) {
    float diff = x[i] - mean;
    sum_diff_sq += diff * diff;
  }
  return sum_diff_sq / (w - 1);
}

void scale_features(const float *raw_feats, float *scaled_feats) {
  for (int i = 0; i < 6; i++) {
    scaled_feats[i] = (raw_feats[i] - feature_means[i]) / feature_stds[i];
  }
}
