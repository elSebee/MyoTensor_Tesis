/*
 * ============================================================
 *  dsp.cpp — Pipeline de filtros IIR EMG
 * ============================================================
 *  Todos los filtros son Biquad de 2° orden, Direct Form I.
 *  fs = 1000 Hz. Variables de estado static (no compartidas).
 *
 *  Coeficientes calculados con la formula estandar:
 *    ω₀ = 2π·f₀/fs
 *    α  = sin(ω₀) / (2·Q)
 *    b0 = b2 = 1/(1+α)
 *    b1 = a1 = −2·cos(ω₀)/(1+α)
 *    a2 =      (1−α)/(1+α)
 *
 *    H(z) = b0·(1 − 2cos(ω₀)·z⁻¹ + z⁻²)
 *           ─────────────────────────────────
 *           (1 − 2rcos(ω₀)·z⁻¹ + r²·z⁻²)
 * ============================================================
 */

#include "dsp.h"

// ============================================================
// NOTCH — Cascada 50 Hz + 60 Hz (Q = 30 cada etapa)
// ============================================================
// Etapa 1: 50 Hz — rechaza interferencia de red (Europa/Argentina)
//   ω₀ = π/10    cos = 0.95106   sin = 0.30902   α = 0.0051503
//   b0=b2=0.9949  b1=-1.8924  a2=0.9898
// Etapa 2: 60 Hz — rechaza interferencia de red (EE.UU. / HVDC)
//   ω₀ = 6π/50   cos = 0.92974   sin = 0.36811   α = 0.0061352
//   b0=b2=0.9939  b1=-1.8482  a2=0.9878
// ============================================================
static float n50_x1=0, n50_x2=0, n50_y1=0, n50_y2=0;
static float n60_x1=0, n60_x2=0, n60_y1=0, n60_y2=0;

float notch(float x) {
  // -- Etapa 1: 50 Hz --
  float w = 0.9949f*x      - 1.8924f*n50_x1 + 0.9949f*n50_x2
                           + 1.8924f*n50_y1  - 0.9898f*n50_y2;
  n50_x2=n50_x1; n50_x1=x;
  n50_y2=n50_y1; n50_y1=w;

  // -- Etapa 2: 60 Hz (entrada = salida de la etapa anterior) --
  float y = 0.9939f*w      - 1.8482f*n60_x1 + 0.9939f*n60_x2
                           + 1.8482f*n60_y1  - 0.9878f*n60_y2;
  n60_x2=n60_x1; n60_x1=w;
  n60_y2=n60_y1; n60_y1=y;

  return y;
}

// ============================================================
// HPF — Pasa-alto 20 Hz, 2° orden Butterworth
// ============================================================
// Elimina offset DC residual y artefactos de movimiento < 20 Hz.
// Coeficientes Butterworth: f_c=20Hz, fs=1000Hz
// ============================================================
static float h_x1=0, h_x2=0, h_y1=0, h_y2=0;
float hpf(float x) {
  float y = 0.9565f*x - 1.9131f*h_x1 + 0.9565f*h_x2
                      + 1.9112f*h_y1  - 0.9150f*h_y2;
  h_x2=h_x1; h_x1=x;
  h_y2=h_y1; h_y1=y;
  return y;
}

// ============================================================
// LPF — Pasa-bajo 500 Hz, 2° orden Butterworth
// ============================================================
// Limita la banda util EMG a < 500 Hz.
// Nota: a 1kHz el LPF Butterworth de 500 Hz es un caso limite
// (f_c = fs/2 = Nyquist). Coefficientes degenerados: pasa todo
// sin atenuar — es efectivamente un FIR promediador 3 puntos.
// Conservado por compatibilidad; el HPF y el Notch hacen el
// trabajo real de limpieza de la señal.
// ============================================================
static float l_x1=0, l_x2=0, l_y1=0, l_y2=0;
float lpf(float x) {
  float y = 0.2929f*x + 0.5858f*l_x1 + 0.2929f*l_x2
                      + 0.0000f*l_y1  - 0.0000f*l_y2;
  l_x2=l_x1; l_x1=x;
  l_y2=l_y1; l_y1=y;
  return y;
}
