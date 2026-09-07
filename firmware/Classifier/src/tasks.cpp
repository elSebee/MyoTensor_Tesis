/*
 * ============================================================
 *  tasks.cpp — Coordinador FreeRTOS con Emisión ESP-NOW y UDP
 * ============================================================
 *
 *  Core 1 — taskAcquisicion : ADC + DSP (1kHz) + Calibración On-Device
 *  Core 0 — taskInferencia  : Windowing + Model + Votación + ESP-NOW + UDP
 * ============================================================
 */

#include "tasks.h"
#include "adc.h"
#include "config.h"
#include "dsp.h"
#include "majority_voting.h"
#include "model_engine.h"
#include "esp_now_sender.h"
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Preferences.h>
#include <cmath>

// ============================================================
// Variables Globales y Estado Compartido
// ============================================================
RingSample ring[RING_SIZE];
volatile uint32_t ring_head = 0;
volatile uint32_t ring_tail = 0;
volatile uint32_t g_dropped = 0;
float *inferBuf = nullptr;
TaskHandle_t hTaskInferencia = nullptr;
WiFiUDP udp;
Preferences prefs;

// Parámetros de Calibración Activos (RAM + NVS Flash)
float g_active_mvc = MODEL_MVC_VOLTAGE_V;
float g_active_noise = MODEL_NOISE_THRESHOLD;

// Estado de la Máquina de Calibración
enum CalibState {
  CALIB_NONE,
  CALIB_REST,
  CALIB_MVC,
  CALIB_DONE
};

volatile CalibState g_calib_state = CALIB_NONE;
volatile uint32_t g_calib_samples_count = 0;
static float g_calib_sum_sq = 0.0f;
static float g_calib_sum = 0.0f;
static float g_calib_max_rms = 0.0f;

// Instancia de Votación Mayoritaria (N=5, 4 clases)
static MajorityVote<5, 4> g_majority_voter;

// ============================================================
// Inicialización de Tareas y Memoria
// ============================================================
bool tasksInit() {
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, HIGH); // Apagado inicial

  ring_head = 0;
  ring_tail = 0;
  g_dropped = 0;

  if (!psramFound()) {
    Serial.println("[ERROR] PSRAM no detectada.");
    return false;
  }
  Serial.printf("[OK] PSRAM disponible: %u bytes\n", ESP.getPsramSize());

  inferBuf = (float *)ps_malloc(WINDOW_SIZE * sizeof(float));
  if (!inferBuf) {
    Serial.println("[ERROR] Fallo al alojar inferBuf en PSRAM");
    return false;
  }
  memset(inferBuf, 0, WINDOW_SIZE * sizeof(float));

  // Cargar calibración previa desde memoria Flash NVS
  prefs.begin("myotensor", false);
  g_active_mvc = prefs.getFloat("mvc", MODEL_MVC_VOLTAGE_V);
  g_active_noise = prefs.getFloat("noise", MODEL_NOISE_THRESHOLD);
  Serial.printf("[CALIB] Calibración Activa -> MVC: %.5f | Ruido: %.5f\n", g_active_mvc, g_active_noise);

  // Inicializar Emisor ESP-NOW Inalámbrico (<1 ms hacia la prótesis)
  esp_now_sender_init();

  dsp_init();
  Serial.println("[OK] Filtros DSP SIMD inicializados");
  return true;
}

void trigger_calibration() {
  g_calib_sum = 0.0f;
  g_calib_sum_sq = 0.0f;
  g_calib_max_rms = 0.0f;
  g_calib_samples_count = 0;
  g_calib_state = CALIB_REST;
  Serial.println("[CALIB] Iniciando Fase 1: Medición de Reposo (2 segundos)...");
}

// ============================================================
// CORE 1 — taskAcquisicion (1 kHz)
// ============================================================
void taskAcquisicion(void *pvParameters) {
  TickType_t xLastWakeTime = xTaskGetTickCount();
  const TickType_t xPeriod = pdMS_TO_TICKS(1000 / FS_HZ);

  Serial.println("[ACQ] Core 1 — Adquisición Vectorizada @ 1kHz iniciada");

  static float raw_batch[WINDOW_STRIDE];
  static uint32_t ts_batch[WINDOW_STRIDE];
  static int batch_idx = 0;

  for (;;) {
    int raw = readADC(EMG_CHANNEL);
    float centered = (float)raw - DC_OFFSET;

    raw_batch[batch_idx] = centered;
    ts_batch[batch_idx] = (uint32_t)micros();
    batch_idx++;

    // ── Máquina de Estados de Calibración en Vivo ──
    if (g_calib_state == CALIB_REST) {
      g_calib_sum += centered;
      g_calib_sum_sq += centered * centered;
      g_calib_samples_count++;

      if ((g_calib_samples_count % 250) == 0) {
        digitalWrite(PIN_LED, !digitalRead(PIN_LED));
      }

      if (g_calib_samples_count >= 2000) { // 2 segundos
        float mean = g_calib_sum / 2000.0f;
        float var = (g_calib_sum_sq / 2000.0f) - (mean * mean);
        float noise_std = sqrtf((var > 0.0f) ? var : 0.0f);
        
        g_active_noise = noise_std / ((g_active_mvc > 1.0f) ? g_active_mvc : 189.84f);
        
        Serial.printf("[CALIB] Fase 1 Completa -> Ruido Raw STD: %.3f\n", noise_std);
        Serial.println("[CALIB] Iniciando Fase 2: Contrae con fuerza por 2 segundos...");
        
        g_calib_sum = 0.0f;
        g_calib_sum_sq = 0.0f;
        g_calib_samples_count = 0;
        g_calib_state = CALIB_MVC;
        digitalWrite(PIN_LED, LOW);
      }
    } else if (g_calib_state == CALIB_MVC) {
      g_calib_sum_sq += centered * centered;
      g_calib_samples_count++;

      if ((g_calib_samples_count % 100) == 0) {
        float win_rms = sqrtf(g_calib_sum_sq / 100.0f);
        if (win_rms > g_calib_max_rms) {
          g_calib_max_rms = win_rms;
        }
        g_calib_sum_sq = 0.0f;
      }

      if (g_calib_samples_count >= 2000) {
        if (g_calib_max_rms < 10.0f) g_calib_max_rms = MODEL_MVC_VOLTAGE_V;
        
        // Factor de confort del 80%
        g_active_mvc = g_calib_max_rms * 0.80f;

        prefs.putFloat("mvc", g_active_mvc);
        prefs.putFloat("noise", g_active_noise);

        Serial.println("[CALIB] Calibración Completada con Éxito!");
        Serial.printf("[CALIB] -> MVC Pico: %.3f | MVC Activo (80%%): %.3f | Ruido: %.5f\n",
                      g_calib_max_rms, g_active_mvc, g_active_noise);

        for (int b = 0; b < 3; b++) {
          digitalWrite(PIN_LED, HIGH); vTaskDelay(pdMS_TO_TICKS(50));
          digitalWrite(PIN_LED, LOW);  vTaskDelay(pdMS_TO_TICKS(50));
        }
        digitalWrite(PIN_LED, HIGH);

        g_calib_state = CALIB_DONE;
      }
    }

    // ── Pipeline por Bloques ──
    if (batch_idx == WINDOW_STRIDE) {
      float filtered_batch[WINDOW_STRIDE];
      dsp_process_block(raw_batch, filtered_batch, WINDOW_STRIDE);

      uint32_t h = ring_head;
      if ((h - ring_tail) + WINDOW_STRIDE <= RING_SIZE) {
        for (int i = 0; i < WINDOW_STRIDE; i++) {
          ring[(h + i) & RING_MASK].timestamp_us = ts_batch[i];
          ring[(h + i) & RING_MASK].filtered = filtered_batch[i];
        }

        __asm__ volatile("" ::: "memory");
        ring_head = h + WINDOW_STRIDE;

        if ((ring_head - ring_tail) >= WINDOW_SIZE) {
          xTaskNotifyGive(hTaskInferencia);
        }
      } else {
        g_dropped += WINDOW_STRIDE;
      }
      batch_idx = 0;
    }

    vTaskDelayUntil(&xLastWakeTime, xPeriod);
  }
}

// ============================================================
// CORE 0 — taskInferencia (Windowing + Model + ESP-NOW + UDP)
// ============================================================
void taskInferencia(void *pvParameters) {
  if (!model_init()) {
    Serial.println("[FATAL] Falló la inicialización del motor de inferencia!");
    vTaskDelete(NULL);
  }

  static IPAddress udp_target_ip(255, 255, 255, 255);
  if (WiFi.status() == WL_CONNECTED) {
    udp.begin(UDP_PORT);
    Serial.printf("[WiFi] Servidor UDP listo en puerto %d\n", UDP_PORT);
  }

  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    // ── Receptor de Comandos UDP ──
    int rx_size = udp.parsePacket();
    if (rx_size > 0) {
      char rx_buf[32] = {0};
      int len = udp.read(rx_buf, sizeof(rx_buf) - 1);
      udp_target_ip = udp.remoteIP();

      if (len > 0) {
        if (strncmp(rx_buf, "CMD_CALIB", 9) == 0) {
          trigger_calibration();
        } else if (strncmp(rx_buf, "CMD_RESET", 9) == 0) {
          prefs.putFloat("mvc", MODEL_MVC_VOLTAGE_V);
          prefs.putFloat("noise", MODEL_NOISE_THRESHOLD);
          g_active_mvc = MODEL_MVC_VOLTAGE_V;
          g_active_noise = MODEL_NOISE_THRESHOLD;
          g_majority_voter.reset();
          Serial.println("[CALIB] Calibración restaurada a valores de fábrica.");
        }
      }
    }

    // ── Control Anti-Lag ──
    uint32_t cur_head = ring_head;
    if ((cur_head - ring_tail) > (WINDOW_SIZE + WINDOW_STRIDE)) {
      ring_tail = cur_head - WINDOW_SIZE;
    }

    while ((ring_head - ring_tail) >= WINDOW_SIZE) {
      uint32_t t = ring_tail;

      float current_mvc = (g_active_mvc > 1.0f) ? g_active_mvc : 189.84517f;
      for (int i = 0; i < WINDOW_SIZE; i++) {
        inferBuf[i] = ring[(t + i) & RING_MASK].filtered / current_mvc;
      }

      __asm__ volatile("" ::: "memory");
      ring_tail = t + WINDOW_STRIDE;

      // 1. Inferencia del Modelo
      float mav = 0.0f;
      float rms = 0.0f;
      int raw_prediction = model_predict(inferBuf, WINDOW_SIZE, mav, rms);

      // 2. Votación Mayoritaria (N=5)
      int filtered_prediction = g_majority_voter.update(raw_prediction);

      // 3. Transmisión Inalámbrica Ultrarrápida ESP-NOW (<1 ms) al DevKit de la Prótesis
      esp_now_send_gesture((uint8_t)filtered_prediction, (uint8_t)raw_prediction, mav, rms);

      // 4. Enviar Paquete UDP a la PC para el Monitor
      if (WiFi.status() == WL_CONNECTED) {
        struct __attribute__((packed)) {
          uint8_t filtered_prediction;
          uint8_t raw_prediction;
          float mav;
          float rms;
        } packet;

        packet.filtered_prediction = (uint8_t)filtered_prediction;
        packet.raw_prediction = (uint8_t)raw_prediction;
        packet.mav = mav;
        packet.rms = rms;

        udp.beginPacket(udp_target_ip, UDP_PORT);
        udp.write((const uint8_t *)&packet, sizeof(packet));
        udp.endPacket();
      }

      // 5. Depuración Serie
      if (Serial) {
        model_log_debug(raw_prediction, filtered_prediction, mav, rms);
      }
    }
  }
}
