/*
 * ============================================================
 *  tasks.cpp — Tasks FreeRTOS (Modo Inferencia)
 * ============================================================
 *
 *  Core 1 — taskAcquisicion : ADC + DSP a 1kHz → buffer circular PSRAM
 *  Core 0 — taskInferencia  : TFLite sobre ventana deslizante (pendiente)
 *
 *  Para streaming UDP de dataset, ver: firmware/Streamer
 * ============================================================
 */

#include "tasks.h"
#include "adc.h"
#include "config.h"
#include "dsp.h"
#include "features.h"
#include <cmath>

#if defined(MODEL_TYPE_SVM)
#include "svm_model.h"
#elif defined(MODEL_TYPE_CNN)
#include "NN_model.h"
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
// Pipeline: ADC → centrar → Notch → HPF → LPF → ring buffer
//
// Politica ante ring lleno: DROP (descartar muestra nueva).
//
// Cada vez que el nivel del ring cruza WINDOW_SIZE, notifica
// a taskInferencia en el Core 0 via xTaskNotifyGive.
// ============================================================
void taskAcquisicion(void *pvParameters) {
  TickType_t xLastWakeTime = xTaskGetTickCount();
  const TickType_t xPeriod = pdMS_TO_TICKS(1000 / FS_HZ);

  Serial.println("[ACQ] Core 1 — adquisicion @ 1kHz");

  for (;;) {
    // 1. Leer ADC y aplicar pipeline DSP
    int raw = readADC(EMG_CHANNEL);
    float centered = (float)raw - DC_OFFSET;
    float filtered = lpf(hpf(notch(centered)));

    // Enviar telemetría en formato Teleplot (o Serial Plotter)
    // Serial.printf(">raw:%d\n>filt:%.3f\n", raw, filtered);

    // 2. Escribir en ring si hay espacio (politica DROP si lleno)
    uint32_t h = ring_head;
    if ((h - ring_tail) < RING_SIZE) {
      ring[h & RING_MASK].timestamp_us = (uint32_t)micros();
      ring[h & RING_MASK].filtered = filtered;

      // Barrera: asegurar que datos esten escritos antes de avanzar head
      __asm__ volatile("" ::: "memory");
      ring_head = h + 1;

      // 3. Cuando hay suficientes muestras para una ventana → despertar
      // taskInferencia en Core 0
      if (((h + 1) - ring_tail) >= WINDOW_SIZE) {
        xTaskNotifyGive(hTaskInferencia);
      }
    } else {
      g_dropped++;
    }

    // 4. Esperar hasta el proximo tick — sin drift acumulado
    vTaskDelayUntil(&xLastWakeTime, xPeriod);
  }
}

// ============================================================
// CORE 0 — taskInferencia
// ============================================================
// Duerme en ulTaskNotifyTake() hasta que Core 1 le avisa que
// hay al menos WINDOW_SIZE muestras en el ring.
//
// En cada ciclo copia WINDOW_SIZE muestras del ring buffer y
// desliza la cola (ring_tail) en WINDOW_STRIDE muestras.
// ============================================================
void taskInferencia(void *pvParameters) {
#if defined(MODEL_TYPE_SVM)
  Eloquent::ML::Port::SVM svm;
  float raw_features[6];
  float scaled_features[6];
  Serial.println("[INF-SVM] Core 0 — Task inferencia SVM iniciada");

#elif defined(MODEL_TYPE_RF)
  Serial.println("[INF-RF] Core 0 — Task inferencia RF iniciada (Stub)");

#elif defined(MODEL_TYPE_CNN)
  Serial.println(
      "[INF-CNN] Core 0 — Task inferencia CNN (TFLite Micro) iniciada");

  // 1. Instanciar variables de TFLite
  const tflite::Model *model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("[ERROR] Model schema version %d is not equal to supported "
                  "version %d.\n",
                  model->version(), TFLITE_SCHEMA_VERSION);
    vTaskDelete(NULL);
  }

  // 2. Resolver operaciones (AllOpsResolver registra todos los ops disponibles)
  static tflite::AllOpsResolver resolver;

  // 3. Definir y alinear Tensor Arena en SRAM
  constexpr int kTensorArenaSize = 160 * 1024;
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

  // 4. Inicializar Intérprete
  static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                              kTensorArenaSize);

  // Alojar tensores
  TfLiteStatus allocate_status = interpreter.AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("[ERROR] AllocateTensors() falló!");
    vTaskDelete(NULL);
  }

  // Obtener tensores de entrada y salida
  TfLiteTensor *input = interpreter.input(0);
  TfLiteTensor *output = interpreter.output(0);

  // Validar dimensiones de la entrada (debe ser 1 x 300 x 1)
  if (input->dims->size != 3 || input->dims->data[0] != 1 ||
      input->dims->data[1] != WINDOW_SIZE || input->dims->data[2] != 1) {
    Serial.println(
        "[ERROR] El tensor de entrada no tiene la forma esperada (1, 300, 1)!");
    vTaskDelete(NULL);
  }

  float input_scale = input->params.scale;
  int32_t input_zero_point = input->params.zero_point;
  Serial.printf("[INF-CNN] TFLite inicializado. Arena utilizada: %d bytes. "
                "Scale: %.6f, ZeroPoint: %d\n",
                interpreter.arena_used_bytes(), input_scale, input_zero_point);
#else
  Serial.println("[INF] Core 0 — Ningún modelo especificado (Flag faltante)");
#endif

  for (;;) {
    // --- Dormir hasta que la adquisicion notifique ---
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    // --- Procesar todas las ventanas disponibles ---
    while ((ring_head - ring_tail) >= WINDOW_SIZE) {
      uint32_t t = ring_tail;

      // Copiar de ring buffer a inferBuf (ventana de WINDOW_SIZE muestras)
      // Normalizamos cada muestra con el voltaje MVC_VOLTAGE_V para alinear con
      // el dataset de entrenamiento
      for (int i = 0; i < WINDOW_SIZE; i++) {
        inferBuf[i] = ring[(t + i) & RING_MASK].filtered / MVC_VOLTAGE_V;
      }

      // Avanzar tail por WINDOW_STRIDE (avanza la ventana deslizante)
      __asm__ volatile("" ::: "memory");
      ring_tail = t + WINDOW_STRIDE;

      int raw_prediction = 0;

#if defined(MODEL_TYPE_SVM)
      // 1. Extracción de Características (MAV, RMS, WL, ZC, SSC, VAR)
      raw_features[0] = compute_mav(inferBuf, WINDOW_SIZE);
      raw_features[1] = compute_rms(inferBuf, WINDOW_SIZE);
      raw_features[2] = compute_wl(inferBuf, WINDOW_SIZE);
      raw_features[3] = compute_zc(inferBuf, WINDOW_SIZE, NOISE_THRESHOLD);
      raw_features[4] = compute_ssc(inferBuf, WINDOW_SIZE, NOISE_THRESHOLD);
      raw_features[5] = compute_var(inferBuf, WINDOW_SIZE);

      // 2. Estandarización Z-Score
      scale_features(raw_features, scaled_features);

      // DEBUG: Imprimir features crudas y escaladas
      Serial.printf("[DBG] Raw:    MAV=%.4f RMS=%.4f WL=%.2f ZC=%.0f SSC=%.0f VAR=%.6f\n",
                    raw_features[0], raw_features[1], raw_features[2],
                    raw_features[3], raw_features[4], raw_features[5]);
      Serial.printf("[DBG] Scaled: MAV=%.2f RMS=%.2f WL=%.2f ZC=%.2f SSC=%.2f VAR=%.2f\n",
                    scaled_features[0], scaled_features[1], scaled_features[2],
                    scaled_features[3], scaled_features[4], scaled_features[5]);

      // 3. Inferencia SVM
      raw_prediction = svm.predict(scaled_features);

#elif defined(MODEL_TYPE_RF)
      // Inferencia Random Forest (Placeholder - retorna clase 0 / reposo)
      raw_prediction = 0;

#elif defined(MODEL_TYPE_CNN)
      // 1. Cuantización de Entrada (Float32 -> Int8)
      // input_val_int8 = round(float_val / scale) + zero_point
      int8_t *input_data = input->data.int8;
      for (int i = 0; i < WINDOW_SIZE; i++) {
        float val = inferBuf[i];
        int32_t quantized = std::round(val / input_scale) + input_zero_point;
        input_data[i] = (int8_t)std::max(-128, std::min(127, (int)quantized));
      }

      // 2. Invocación de Inferencia
      TfLiteStatus invoke_status = interpreter.Invoke();
      if (invoke_status != kTfLiteOk) {
        Serial.println("[ERROR] Invoke falló!");
        continue;
      }

      // 3. ArgMax en el Output (Detección de clase en INT8)
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

      // 4. Suavizado por Votación Mayoritaria
      int filtered_prediction = majority_vote(raw_prediction);

      // 5. Impresión en Monitor Serie
      const char *gesture_names[] = {"REPOSO", "PALMA", "PUNO", "PAZ"};

#if defined(MODEL_TYPE_CNN)
      Serial.printf("[INF-CNN] Crudo: %-6s (%d) | Suavizado: %-6s (%d) | Out "
                    "Prob (INT8): [%d, %d, %d, %d]\n",
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
      // Ceder CPU al IDLE task para evitar trigger del Task Watchdog (WDT)
      taskYIELD();
    }
  }
}
