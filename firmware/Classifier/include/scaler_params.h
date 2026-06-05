#pragma once
/*
 * =============================================================
 *  scaler_params.h — Parámetros de normalización del StandardScaler
 * =============================================================
 */

// Medias de las características (MAV, RMS, WL, ZC, SSC, VAR)
const float feature_means[6] = { 0.29915272f, 0.38865380f, 54.49609358f, 62.73089905f, 80.85946609f, 0.19429551f };

// Desviaciones estándar de las características (MAV, RMS, WL, ZC, SSC, VAR)
const float feature_stds[6] = { 0.14838732f, 0.20737231f, 21.23280157f, 12.77665914f, 5.50626956f, 0.23211395f };
