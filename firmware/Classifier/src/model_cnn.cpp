#if defined(MODEL_TYPE_CNN)
/*
 * ============================================================
 *  model_cnn.cpp — Motor de Inferencia Deep Learning TFLite Micro
 * ============================================================
 */

#include "model_engine.h"
#include "features.h"
#include "NN_model.h"
#include "model_config.h"
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esp_heap_caps.h"

static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input_tensor = nullptr;
static TfLiteTensor *output_tensor = nullptr;
static float cnn_inv_scale = 1.0f;
static float cnn_offset = 0.0f;

bool model_init() {
  Serial.println("[MODEL] Inicializando motor CNN/TCN (TFLite Micro)...");
  const tflite::Model *model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("[ERROR] Model schema version mismatch.\n");
    return false;
  }

  static tflite::AllOpsResolver resolver;
  // Tamaño optimizado para TinyML-TCN SIMD (~30 KB requeridos)
  constexpr int kTensorArenaSize = 48 * 1024;
  static uint8_t *tensor_arena = nullptr;
  if (!tensor_arena) {
    // Intentar primero en SRAM interna rápida (240 MHz, 0 wait states) con alineación 16B para SIMD
    tensor_arena = (uint8_t *)heap_caps_aligned_alloc(16, kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (tensor_arena) {
      Serial.println("[MODEL] tensor_arena alojado en SRAM INTERNA (Fast SIMD).");
    } else {
      // Fallback a PSRAM externa
      tensor_arena = (uint8_t *)heap_caps_aligned_alloc(16, kTensorArenaSize, MALLOC_CAP_SPIRAM);
      if (tensor_arena) {
        Serial.println("[MODEL] tensor_arena alojado en PSRAM (Fallback).");
      } else {
        Serial.println("[ERROR] Fallo al alojar tensor_arena!");
        return false;
      }
    }
  }

  static tflite::MicroErrorReporter micro_error_reporter;
  static tflite::ErrorReporter *error_reporter = &micro_error_reporter;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("[ERROR] AllocateTensors() falló!");
    return false;
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);
  float input_scale = input_tensor->params.scale;
  int32_t input_zero_point = input_tensor->params.zero_point;

  // Pre-cálculo de escala para optimización FMA hardware
  cnn_inv_scale = 1.0f / (signal_std * input_scale);
  cnn_offset = -(signal_mean * cnn_inv_scale) + input_zero_point;

  Serial.printf("[MODEL] TFLite listo. Scale: %.6f, ZeroPoint: %d\n", input_scale, input_zero_point);
  return true;
}

int model_predict(const float *window, int window_size, float &mav, float &rms) {
  mav = compute_mav(window, window_size);
  rms = compute_rms(window, window_size);

  int8_t *input_data = input_tensor->data.int8;
  for (int i = 0; i < window_size; i++) {
    float raw_val = window[i];
    float scaled_val = raw_val * cnn_inv_scale + cnn_offset;
    int32_t q = (int32_t)(scaled_val + (scaled_val >= 0 ? 0.5f : -0.5f));
    
    if (q < -128) q = -128;
    else if (q > 127) q = 127;
    
    input_data[i] = (int8_t)q;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    return 0;
  }

  int8_t *output_data = output_tensor->data.int8;
  int max_idx = 0;
  int8_t max_val = output_data[0];
  for (int c = 1; c < 4; c++) {
    if (output_data[c] > max_val) {
      max_val = output_data[c];
      max_idx = c;
    }
  }
  return max_idx;
}

void model_log_debug(int raw_pred, int filtered_pred, float mav, float rms) {
  const char *gesture_names[] = {"REPOSO", "PALMA", "PUNO", "PAZ"};
  const char *raw_name = (raw_pred >= 0 && raw_pred < 4) ? gesture_names[raw_pred] : "DESC";
  const char *filt_name = (filtered_pred >= 0 && filtered_pred < 4) ? gesture_names[filtered_pred] : "DESC";

  if (output_tensor) {
    Serial.printf("[INF-CNN] Crudo: %-6s (%d) | Suavizado: %-6s (%d) | Out Prob: [%d, %d, %d, %d] | MAV: %.4f\n",
                  raw_name, raw_pred, filt_name, filtered_pred,
                  output_tensor->data.int8[0], output_tensor->data.int8[1],
                  output_tensor->data.int8[2], output_tensor->data.int8[3],
                  mav);
  }
}

#endif // MODEL_TYPE_CNN
