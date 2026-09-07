#pragma once
/*
 * =============================================================
 *  model_config.h — Configuración Dinámica Autogenerada
 *  Generado automáticamente desde el pipeline de Python
 * =============================================================
 */

// Parámetros de Ventana
#define MODEL_WINDOW_SIZE      200
#define MODEL_WINDOW_STRIDE    100

// Calibración de Fábrica de la Sesión
#define MODEL_MVC_VOLTAGE_V    189.84517f
#define MODEL_NOISE_THRESHOLD  0.07440f

// Parámetros del StandardScaler (MAV, RMS, WL, ZC, SSC, VAR)
const float feature_means[6] = { 0.20749316f, 0.26569015f, 42.18455848f, 49.68233343f, 91.82361073f, 0.10258632f };
const float feature_stds[6]  = { 0.14546442f, 0.20518429f, 35.28810429f, 26.82730279f, 21.93785341f, 0.22838535f };
