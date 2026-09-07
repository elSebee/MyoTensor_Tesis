#pragma once
/*
 * ============================================================
 *  features.h — Prototipos para Extraccion de Caracteristicas
 * ============================================================
 */

#include "config.h"

// Umbral de ruido dinamico heredado de model_config.h
#define NOISE_THRESHOLD MODEL_NOISE_THRESHOLD

float compute_mav(const float *x, int w);
float compute_rms(const float *x, int w);
float compute_wl(const float *x, int w);
float compute_zc(const float *x, int w, float threshold = NOISE_THRESHOLD);
float compute_ssc(const float *x, int w, float threshold = NOISE_THRESHOLD);
float compute_var(const float *x, int w);

void scale_features(const float *raw_feats, float *scaled_feats);
