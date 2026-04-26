#pragma once
/*
 * ============================================================
 *  dsp.h — Interfaz del pipeline de filtros IIR EMG
 * ============================================================
 *  Todos los filtros son Biquad IIR en Direct Form I.
 *  Coeficientes calculados para fs = 1000 Hz.
 *
 *  Pipeline recomendado (en orden):
 *    1. notch()   — cascada 50 Hz + 60 Hz (Q=30 cada etapa)
 *    2. hpf()     — elimina artefactos de movimiento < 20 Hz
 *    3. lpf()     — promediador suave (caso Nyquist 500 Hz)
 *    4. fabsf()   — rectificacion (envolvente de amplitud)
 * ============================================================
 */

#include <Arduino.h>

// Notch en cascada 50 Hz + 60 Hz (Q=30 por etapa).
// Rechaza red electrica Argentina (50Hz) y EE.UU./armónicos (60Hz).
// Internamente aplica dos biquads en serie — firma identica a notch simple.
float notch(float x);

// Pasa-alto 20 Hz, 2° orden Butterworth: elimina offset DC y artefactos
float hpf(float x);

// Pasa-bajo 500 Hz, 2° orden Butterworth (caso Nyquist — promediador suave)
float lpf(float x);
