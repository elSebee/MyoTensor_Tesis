#pragma once
/*
 * =============================================================
 *  model_config.h — Configuración Dinámica para CNN-TCN (ED Calibrado 10%)
 *  Modelo: tcn_ED_calib10_int8 (W=250 ms, S=50 ms @ 1 kHz)
 * =============================================================
 */

// Parámetros de Ventana Canónica de la Tesis
#define MODEL_WINDOW_SIZE      250
#define MODEL_WINDOW_STRIDE    50

// Calibración de Fábrica de la Sesión (Sujeto S01 - ED)
#define MODEL_MVC_VOLTAGE_V    476.4597f
#define MODEL_NOISE_THRESHOLD  0.03188f

// Parámetros del StandardScaler para características clásicas (Compatibilidad)
const float feature_means[6] = { 0.11422369f, 0.14775253f, 17.74001650f, 36.99496220f, 58.51385390f, 0.03493297f };
const float feature_stds[6]  = { 0.08844918f, 0.11644882f, 14.20595133f, 19.85856730f, 33.75012543f, 0.04646605f };

// Parámetros de Escala para Señal Temporal CNN-TCN (Entrada normalizada por %MVC)
const float signal_mean = 0.0f;
const float signal_std  = 1.0f;
