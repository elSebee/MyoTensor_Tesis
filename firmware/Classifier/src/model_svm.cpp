#if defined(MODEL_TYPE_SVM)
/*
 * ============================================================
 *  model_svm.cpp — Motor de Inferencia Support Vector Machine
 * ============================================================
 */

#include "model_engine.h"
#include "features.h"
#include "svm_model.h"

extern float g_active_noise;
static Eloquent::ML::Port::SVM svm;

bool model_init() {
  Serial.println("[MODEL] Motor SVM (micromlgen) inicializado.");
  return true;
}

int model_predict(const float *window, int window_size, float &mav, float &rms) {
  float raw_features[6];
  float scaled_features[6];

  // 1. Extraccion de 6 caracteristicas temporales
  raw_features[0] = compute_mav(window, window_size);
  raw_features[1] = compute_rms(window, window_size);
  raw_features[2] = compute_wl(window, window_size);
  raw_features[3] = compute_zc(window, window_size, g_active_noise);
  raw_features[4] = compute_ssc(window, window_size, g_active_noise);
  raw_features[5] = compute_var(window, window_size);

  mav = raw_features[0];
  rms = raw_features[1];

  // 2. Normalizacion estandar (Z-Score)
  scale_features(raw_features, scaled_features);

  // 3. Inferencia SVM
  return svm.predict(scaled_features);
}

void model_log_debug(int raw_pred, int filtered_pred, float mav, float rms) {
  const char *gesture_names[] = {"REPOSO", "PALMA", "PUNO", "PAZ"};
  const char *raw_name = (raw_pred >= 0 && raw_pred < 4) ? gesture_names[raw_pred] : "DESC";
  const char *filt_name = (filtered_pred >= 0 && filtered_pred < 4) ? gesture_names[filtered_pred] : "DESC";
  
  Serial.printf("[INF-SVM] Crudo: %-6s (%d) | Suavizado: %-6s (%d) | MAV: %.4f | RMS: %.4f\n",
                raw_name, raw_pred, filt_name, filtered_pred, mav, rms);
}

#endif // MODEL_TYPE_SVM
