/*
 * ============================================================
 *  tasks.cpp — Tasks FreeRTOS (Modo Inferencia)
 * ============================================================
 *
 *  Core 1 — taskAcquisicion : ADC + DSP a 1kHz → buffer circular PSRAM
 *  Core 0 — taskInferencia  : Extracción de Características + Inferencia
 * ============================================================
 */

#include "tasks.h"
#include "adc.h"
#include "config.h"
#include "dsp.h"
#include "features.h"
#include <cmath>
#include <WiFi.h>
#include <WiFiUdp.h>

#if defined(MODEL_TYPE_SVM)
#include "svm_model.h"
#elif defined(MODEL_TYPE_RF)
#include "rf_model.h"
#elif defined(MODEL_TYPE_CNN)
#include "NN_model.h"
#include "cnn_scaler_params.h"
#include "esp_heap_caps.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#endif

// ============================================================
// Estado compartido entre cores (SPSC lock-free)
// ============================================================
RingSample ring[RING_SIZE];
volatile uint32_t ring_head = 0;
volatile uint32_t ring_tail = 0;
volatile uint32_t g_dropped = 0;
float *inferBuf = nullptr;
TaskHandle_t hTaskInferencia = nullptr;
WiFiUDP udp;

// ── Filtro de Votación Mayoritaria (Majority Vote) con BUFFER_SIZE = 5 ──
int majority_vote(int new_prediction) {
  static int vote_buffer[5] = {0, 0, 0, 0, 0};
  static int buffer_idx = 0;

  // Insertar nueva predicción en el buffer circular
  vote_buffer[buffer_idx] = new_prediction;
  buffer_idx = (buffer_idx + 1) % 5;

  // Contar frecuencias para las 4 clases (Reposo, Palma, Puño, Paz)
  int counts[4] = {0, 0, 0, 0};
  for (int i = 0; i < 5; i++) {
    int pred = vote_buffer[i];
    if (pred >= 0 && pred < 4) {
      counts[pred]++;
    }
  }

  // Encontrar la moda estadística
  int mode_class = 0;
  int max_count = -1;
  for (int c = 0; c < 4; c++) {
    if (counts[c] > max_count) {
      max_count = counts[c];
      mode_class = c;
    }
  }
  return mode_class;
}

// ============================================================
// tasksInit — Alojar buffers en PSRAM e inicializar ring buffer
// ============================================================
bool tasksInit() {
  ring_head = 0;
  ring_tail = 0;
  g_dropped = 0;

  if (!psramFound()) {
    Serial.println("[ERROR] PSRAM no detectada.");
    Serial.println(
        "        Verificar: board_build.arduino.memory_type = qio_opi");
    return false;
  }
  Serial.printf("[OK] PSRAM: %u bytes disponibles\n", ESP.getPsramSize());

  inferBuf = (float *)ps_malloc(WINDOW_SIZE * sizeof(float));
  if (!inferBuf) {
    Serial.println("[ERROR] Fallo al alojar inferBuf en PSRAM");
    return false;
  }
  memset(inferBuf, 0, WINDOW_SIZE * sizeof(float));

  Serial.printf("[OK] Ring buffer : %u slots x %u bytes = %u bytes SRAM\n",
                RING_SIZE, (unsigned)sizeof(RingSample),
                RING_SIZE * (unsigned)sizeof(RingSample));
  Serial.printf("[OK] Buffer de Inferencia PSRAM: %d bytes\n",
                WINDOW_SIZE * (int)sizeof(float));
  return true;
}

// ============================================================
// CORE 1 — taskAcquisicion
// ============================================================
void taskAcquisicion(void *pvParameters) {
  TickType_t xLastWakeTime = xTaskGetTickCount();
  const TickType_t xPeriod = pdMS_TO_TICKS(1000 / FS_HZ);

  Serial.println("[ACQ] Core 1 — adquisicion @ 1kHz");
  dsp_init(); // Inicializar estados de los filtros IIR

  float raw_block[WINDOW_STRIDE];
  float filtered_block[WINDOW_STRIDE];
  int block_idx = 0;

  for (;;) {
    // 1. Leer ADC y quitar DC Offset
    int raw = readADC(EMG_CHANNEL);
    raw_block[block_idx++] = (float)raw - DC_OFFSET;

    // 2. Al llenar el bloque, aplicar DSP vectorial (SIMD)
    if (block_idx == WINDOW_STRIDE) {
      dsp_process_block(raw_block, filtered_block, WINDOW_STRIDE);

      uint32_t h = ring_head;
      for (int i = 0; i < WINDOW_STRIDE; i++) {
        if ((h - ring_tail) < RING_SIZE) {
          ring[h & RING_MASK].timestamp_us = (uint32_t)micros();
          ring[h & RING_MASK].filtered = filtered_block[i];
          h++;
        } else {
          g_dropped++;
        }
      }

      // Barrera de memoria (asegura escritura antes de actualizar head)
      __asm__ volatile("" ::: "memory");
      ring_head = h;

      // 3. Notificar a Core 0 si hay una ventana de 300 muestras lista
      if ((ring_head - ring_tail) >= WINDOW_SIZE) {
        xTaskNotifyGive(hTaskInferencia);
      }

      // Reiniciar índice del bloque
      block_idx = 0;
    }

    // 4. Esperar al siguiente tick de 1 ms
    vTaskDelayUntil(&xLastWakeTime, xPeriod);
  }
}

// ============================================================
// CORE 0 — taskInferencia
// ============================================================
void taskInferencia(void *pvParameters) {
  
  // ==========================================
  // CONFIGURACIÓN INICIAL (Fuera del Bucle)
  // ==========================================
#if defined(MODEL_TYPE_SVM)
  Eloquent::ML::Port::SVM svm;
  float raw_features[6];
  float scaled_features[6];
  Serial.println("[INF-SVM] Core 0 — Task inferencia SVM iniciada");

#elif defined(MODEL_TYPE_RF)
  Eloquent::ML::Port::RandomForest rf;
  float raw_features[6];
  Serial.println("[INF-RF] Core 0 — Task inferencia RF iniciada");

#elif defined(MODEL_TYPE_CNN)
  Serial.println("[INF-CNN] Task inferencia CNN (TFLite Micro) iniciada");
  Serial.flush();
  vTaskDelay(pdMS_TO_TICKS(100));

  const tflite::Model *model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("[ERROR] Model schema version mismatch.\n");
    vTaskDelete(NULL);
  }

  static tflite::AllOpsResolver resolver;
  constexpr int kTensorArenaSize = 500 * 1024;
  static uint8_t *tensor_arena = nullptr;
  if (!tensor_arena) {
    tensor_arena = (uint8_t *)heap_caps_aligned_alloc(16, kTensorArenaSize, MALLOC_CAP_SPIRAM);
    if (!tensor_arena) {
      Serial.println("[ERROR] Fallo al alojar tensor_arena!");
      vTaskDelete(NULL);
    }
  }

  static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  TfLiteStatus allocate_status = interpreter.AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("[ERROR] AllocateTensors() falló!");
    vTaskDelete(NULL);
  }

  TfLiteTensor *input = interpreter.input(0);
  TfLiteTensor *output = interpreter.output(0);
  float input_scale = input->params.scale;
  int32_t input_zero_point = input->params.zero_point;
  
  // ======================================================================
  // [OPTIMIZACIÓN DE HW]: Pre-calcular factores de escala para evitar 
  // divisiones lentas por cada muestra en el ciclo de inferencia.
  // formula original: val = (raw - mean) / std; q = val / scale + z_point
  // formula rápida: q = raw * (1 / (std*scale)) + (z_point - mean / (std*scale))
  // ======================================================================
  const float cnn_inv_scale = 1.0f / (signal_std * input_scale);
  const float cnn_offset = -(signal_mean * cnn_inv_scale) + input_zero_point;
  
  Serial.printf("[INF-CNN] TFLite inicializado. Scale: %.6f, ZeroPoint: %d\n", input_scale, input_zero_point);
#else
  Serial.println("[INF] Core 0 — Ningún modelo especificado");
#endif

  if (WiFi.status() == WL_CONNECTED) {
    udp.begin(UDP_PORT);
    Serial.printf("[WiFi] Servidor UDP iniciado en puerto %d\n", UDP_PORT);
  }

  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    while ((ring_head - ring_tail) >= WINDOW_SIZE) {
      uint32_t t = ring_tail;

      // ==========================================
      // MÓDULO 1: Window Manager
      // ==========================================
      for (int i = 0; i < WINDOW_SIZE; i++) {
        inferBuf[i] = ring[(t + i) & RING_MASK].filtered / MVC_VOLTAGE_V;
      }
      __asm__ volatile("" ::: "memory");
      ring_tail = t + WINDOW_STRIDE;

      int raw_prediction = 0;

      // ==========================================
      // MÓDULO 2: Feature Extractor
      // (Se omite para CNN para ahorrar CPU)
      // ==========================================
#if defined(MODEL_TYPE_SVM) || defined(MODEL_TYPE_RF)
      raw_features[0] = compute_mav(inferBuf, WINDOW_SIZE);
      raw_features[1] = compute_rms(inferBuf, WINDOW_SIZE);
      raw_features[2] = compute_wl(inferBuf, WINDOW_SIZE);
      raw_features[3] = compute_zc(inferBuf, WINDOW_SIZE, NOISE_THRESHOLD);
      raw_features[4] = compute_ssc(inferBuf, WINDOW_SIZE, NOISE_THRESHOLD);
      raw_features[5] = compute_var(inferBuf, WINDOW_SIZE);
#endif

      // ==========================================
      // MÓDULO 3: Motor de Inferencia
      // ==========================================
#if defined(MODEL_TYPE_SVM)
      scale_features(raw_features, scaled_features);
      raw_prediction = svm.predict(scaled_features);

#elif defined(MODEL_TYPE_RF)
      // RF no requiere escalado de características
      raw_prediction = rf.predict(raw_features);

#elif defined(MODEL_TYPE_CNN)
      int8_t *input_data = input->data.int8;
      for (int i = 0; i < WINDOW_SIZE; i++) {
        // [OPTIMIZACIÓN]: Fusión de escalado Z-Score + Cuantización INT8.
        // Utiliza una sola instrucción fma (fused multiply-add) del ESP32-S3 y
        // casteo rápido truncando en lugar de usar librerías <cmath> pesadas.
        float raw_val = inferBuf[i];
        float scaled_val = raw_val * cnn_inv_scale + cnn_offset;
        int32_t q = (int32_t)(scaled_val + (scaled_val >= 0 ? 0.5f : -0.5f));
        
        // Saturación rápida por ifs (más rápido que std::min/max)
        if (q < -128) q = -128;
        else if (q > 127) q = 127;
        
        input_data[i] = (int8_t)q;
      }

      if (interpreter.Invoke() != kTfLiteOk) {
        Serial.println("[ERROR] Invoke falló!");
        continue;
      }

      int8_t *output_data = output->data.int8;
      int max_idx = 0;
      int8_t max_val = output_data[0];
      for (int c = 1; c < 4; c++) {
        if (output_data[c] > max_val) {
          max_val = output_data[c];
          max_idx = c;
        }
      }
      raw_prediction = max_idx;
#endif

      // ==========================================
      // MÓDULO 4: Post-procesamiento
      // ==========================================
      int filtered_prediction = majority_vote(raw_prediction);

      // ==========================================
      // MÓDULO 5: Control Actuador y UDP
      // ==========================================
      if (WiFi.status() == WL_CONNECTED) {
        struct __attribute__((packed)) {
          uint8_t filtered_prediction;
          uint8_t raw_prediction;
          float mav;
          float rms;
        } packet;

        packet.filtered_prediction = (uint8_t)filtered_prediction;
        packet.raw_prediction = (uint8_t)raw_prediction;
        packet.mav = compute_mav(inferBuf, WINDOW_SIZE);
        packet.rms = compute_rms(inferBuf, WINDOW_SIZE);

        udp.beginPacket(UDP_BROADCAST_IP, UDP_PORT);
        udp.write((const uint8_t*)&packet, sizeof(packet));
        udp.endPacket();
      }

      // [PENDIENTE] Control I2C Servomotores 
      // if (filtered_prediction == 1) { setServoAngle(180); } ...

      // Impresión en Monitor Serie
      const char *gesture_names[] = {"REPOSO", "PALMA", "PUNO", "PAZ"};
#if defined(MODEL_TYPE_CNN)
      Serial.printf("[INF-CNN] Crudo: %-6s (%d) | Suavizado: %-6s (%d) | Out "
                    "Prob: [%d, %d, %d, %d]\n",
                    gesture_names[raw_prediction], raw_prediction,
                    gesture_names[filtered_prediction], filtered_prediction,
                    output->data.int8[0], output->data.int8[1],
                    output->data.int8[2], output->data.int8[3]);
#else
      Serial.printf("[INF] Crudo: %-6s (%d) | Suavizado: %-6s (%d) | MAV: %.4f "
                    "| RMS: %.4f\n",
                    gesture_names[raw_prediction], raw_prediction,
                    gesture_names[filtered_prediction], filtered_prediction,
                    compute_mav(inferBuf, WINDOW_SIZE),
                    compute_rms(inferBuf, WINDOW_SIZE));
#endif
      
      // Ceder CPU
      vTaskDelay(pdMS_TO_TICKS(1));
    }
  }
}
