#pragma once
/*
 * ============================================================
 *  model_engine.h — Interfaz unificada de motores de inferencia
 * ============================================================
 *  Permite desacoplar tasks.cpp de la implementación específica
 *  de cada modelo (SVM, Random Forest, CNN-TCN).
 * ============================================================
 */

#include <Arduino.h>

// Inicializa el modelo seleccionado (alocación de tensores, memoria, etc.)
bool model_init();

// Ejecuta la inferencia sobre la ventana de EMG normalizada.
// Retorna la clase predicha cruda (0: REPOSO, 1: PALMA, 2: PUÑO, 3: PAZ)
// y calcula las métricas MAV y RMS para transmisión UDP y telemetría.
int model_predict(const float *window, int window_size, float &mav, float &rms);

// Imprime información de depuración detallada por el monitor Serial
void model_log_debug(int raw_pred, int filtered_pred, float mav, float rms);
