#pragma once
/*
 * ============================================================
 *  features.h — Extracción de Características Temporales
 * ============================================================
 */

#include <Arduino.h>

const float NOISE_THRESHOLD = 0.005f;

float compute_mav(const float *x, int w);
float compute_rms(const float *x, int w);
float compute_wl(const float *x, int w);
float compute_zc(const float *x, int w, float threshold = NOISE_THRESHOLD);
float compute_ssc(const float *x, int w, float threshold = NOISE_THRESHOLD);
float compute_var(const float *x, int w);

void scale_features(const float *raw_feats, float *scaled_feats);
