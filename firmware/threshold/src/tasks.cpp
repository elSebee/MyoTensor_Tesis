/*
 * ============================================================
 *  tasks.cpp — Coordinador FreeRTOS con Control por Umbral (Threshold)
 * ============================================================
 *
 *  Core 1 — taskAcquisicion : ADC (MCP3208 SPI) + DSP IIR (1kHz) + Calibración
 *  Core 0 — taskThreshold  : Extracción MAV/RMS + Umbral + ESP-NOW + UDP
 * ============================================================
 */

#include "tasks.h"
#include "adc.h"
#include "config.h"
#include "dsp.h"
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
float *windowBuf = nullptr;
TaskHandle_t hTaskThreshold = nullptr;
WiFiUDP udp;
Preferences prefs;

// Parámetros de Calibración y Umbrales Activos (RAM + NVS Flash)
float g_active_mvc       = DEFAULT_MVC_VOLTAGE_V;
float g_active_noise     = DEFAULT_NOISE_THRESHOLD;
float g_active_threshold = DEFAULT_THRESHOLD_MAV;

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

// Estado actual de la prótesis (con histéresis)
static uint8_t s_current_hand_state = GESTURE_OPEN;

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
    Serial.println("[WARN] PSRAM no detectada. Usando SRAM interna para windowBuf.");
    windowBuf = (float *)malloc(WINDOW_SIZE * sizeof(float));
  } else {
    Serial.printf("[OK] PSRAM disponible: %u bytes\n", ESP.getPsramSize());
    windowBuf = (float *)ps_malloc(WINDOW_SIZE * sizeof(float));
  }

  if (!windowBuf) {
    Serial.println("[FATAL] Fallo al alojar memoria para windowBuf");
    return false;
  }
  memset(windowBuf, 0, WINDOW_SIZE * sizeof(float));

  // Cargar parámetros desde memoria Flash NVS (persistentes)
  prefs.begin("myotensor_th", false);
  g_active_mvc       = prefs.getFloat("mvc", DEFAULT_MVC_VOLTAGE_V);
  g_active_noise     = prefs.getFloat("noise", DEFAULT_NOISE_THRESHOLD);
  g_active_threshold = prefs.getFloat("thresh", DEFAULT_THRESHOLD_MAV);

  Serial.println("=================================================");
  Serial.println("   Configuración de Umbral (Threshold) Activa    ");
  Serial.println("=================================================");
  Serial.printf("   MVC Referencia : %.2f cuentas\n", g_active_mvc);
  Serial.printf("   Ruido Basal    : %.2f cuentas\n", g_active_noise);
  Serial.printf("   Umbral MAV     : %.2f cuentas\n", g_active_threshold);
  Serial.printf("   Histéresis     : %.2f (Abre < %.2f)\n", 
                THRESHOLD_HYSTERESIS, g_active_threshold * THRESHOLD_HYSTERESIS);
  Serial.println("=================================================");

  // Inicializar Emisor ESP-NOW Inalámbrico (<1 ms hacia la prótesis)
  esp_now_sender_init();

  // Inicializar filtros IIR SIMD
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
  Serial.println("\n[CALIB] >>> FASE 1: Mantén el brazo en REPOSO (2 segundos)...");
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
        
        g_active_noise = noise_std;
        
        Serial.printf("[CALIB] Fase 1 Completa -> Ruido Raw STD: %.2f cuentas\n", noise_std);
        Serial.println("[CALIB] >>> FASE 2: ¡¡CONTRAE CON FUERZA MÁXIMA (2 segundos)!!");
        
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

      if (g_calib_samples_count >= 2000) { // 2 segundos
        if (g_calib_max_rms < 10.0f) g_calib_max_rms = DEFAULT_MVC_VOLTAGE_V;
        
        g_active_mvc = g_calib_max_rms * 0.80f; // 80% del pico MVC
        
        // Umbral adaptativo: a mitad de camino entre el ruido y la contracción confortable
        float calculated_thresh = g_active_noise * 3.0f;
        if (calculated_thresh < (g_active_mvc * 0.25f)) {
          calculated_thresh = g_active_mvc * 0.25f;
        }
        g_active_threshold = calculated_thresh;

        // Guardar en NVS Flash
        prefs.putFloat("mvc", g_active_mvc);
        prefs.putFloat("noise", g_active_noise);
        prefs.putFloat("thresh", g_active_threshold);

        Serial.println("[CALIB] ¡Calibración Completada con Éxito!");
        Serial.printf("[CALIB] -> MVC Pico: %.2f | MVC Activo: %.2f | Ruido: %.2f\n",
                      g_calib_max_rms, g_active_mvc, g_active_noise);
        Serial.printf("[CALIB] -> Nuevo Umbral MAV Establecido: %.2f cuentas\n", g_active_threshold);

        for (int b = 0; b < 3; b++) {
          digitalWrite(PIN_LED, HIGH); vTaskDelay(pdMS_TO_TICKS(50));
          digitalWrite(PIN_LED, LOW);  vTaskDelay(pdMS_TO_TICKS(50));
        }
        digitalWrite(PIN_LED, HIGH);

        g_calib_state = CALIB_DONE;
      }
    }

    // ── Pipeline por Bloques (Notch + HPF + LPF) ──
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
          xTaskNotifyGive(hTaskThreshold);
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
// CORE 0 — taskThreshold (Extracción MAV/RMS + Umbral + Accionamiento)
// ============================================================
void taskThreshold(void *pvParameters) {
  static IPAddress udp_target_ip(255, 255, 255, 255);
  if (WiFi.status() == WL_CONNECTED) {
    udp.begin(UDP_PORT);
    Serial.printf("[WiFi] Servidor UDP listo en puerto %d\n", UDP_PORT);
  }

  uint32_t last_log_time = 0;

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
          prefs.putFloat("mvc", DEFAULT_MVC_VOLTAGE_V);
          prefs.putFloat("noise", DEFAULT_NOISE_THRESHOLD);
          prefs.putFloat("thresh", DEFAULT_THRESHOLD_MAV);
          g_active_mvc       = DEFAULT_MVC_VOLTAGE_V;
          g_active_noise     = DEFAULT_NOISE_THRESHOLD;
          g_active_threshold = DEFAULT_THRESHOLD_MAV;
          Serial.println("[CALIB] Umbral y parámetros restaurados a valores de fábrica.");
        }
      }
    }

    // ── Receptor de Comandos Serial ──
    if (Serial.available()) {
      char c = Serial.read();
      if (c == 'c' || c == 'C') {
        trigger_calibration();
      } else if (c == 'r' || c == 'R') {
        prefs.putFloat("mvc", DEFAULT_MVC_VOLTAGE_V);
        prefs.putFloat("noise", DEFAULT_NOISE_THRESHOLD);
        prefs.putFloat("thresh", DEFAULT_THRESHOLD_MAV);
        g_active_mvc       = DEFAULT_MVC_VOLTAGE_V;
        g_active_noise     = DEFAULT_NOISE_THRESHOLD;
        g_active_threshold = DEFAULT_THRESHOLD_MAV;
        Serial.println("[SERIAL] Restaurado a valores por defecto.");
      } else if (c == '+') {
        g_active_threshold += 5.0f;
        prefs.putFloat("thresh", g_active_threshold);
        Serial.printf("[SERIAL] Umbral aumentado a: %.2f cuentas\n", g_active_threshold);
      } else if (c == '-') {
        g_active_threshold = fmaxf(5.0f, g_active_threshold - 5.0f);
        prefs.putFloat("thresh", g_active_threshold);
        Serial.printf("[SERIAL] Umbral reducido a: %.2f cuentas\n", g_active_threshold);
      }
    }

    // ── Control Anti-Lag ──
    uint32_t cur_head = ring_head;
    if ((cur_head - ring_tail) > (WINDOW_SIZE + WINDOW_STRIDE)) {
      ring_tail = cur_head - WINDOW_SIZE;
    }

    while ((ring_head - ring_tail) >= WINDOW_SIZE) {
      uint32_t t = ring_tail;

      // 1. Extraer ventana y calcular MAV / RMS
      float sum_abs = 0.0f;
      float sum_sq  = 0.0f;

      for (int i = 0; i < WINDOW_SIZE; i++) {
        float val = ring[(t + i) & RING_MASK].filtered;
        windowBuf[i] = val;
        sum_abs += fabsf(val);
        sum_sq  += val * val;
      }

      __asm__ volatile("" ::: "memory");
      ring_tail = t + WINDOW_STRIDE;

      float mav = sum_abs / (float)WINDOW_SIZE;
      float rms = sqrtf(sum_sq / (float)WINDOW_SIZE);

      // 2. Decisión de Umbral con Histéresis
      // Estado Crudo (sin histéresis)
      uint8_t raw_state = (mav >= g_active_threshold) ? GESTURE_CLOSED : GESTURE_OPEN;

      // Estado con Histéresis:
      // Si la mano está abierta (0), requiere superar 'g_active_threshold' para cerrarse (2).
      // Si la mano está cerrada (2), requiere caer por debajo de 'g_active_threshold * THRESHOLD_HYSTERESIS' para abrirse (0).
      if (s_current_hand_state == GESTURE_OPEN) {
        if (mav >= g_active_threshold) {
          s_current_hand_state = GESTURE_CLOSED;
        }
      } else { // GESTURE_CLOSED
        if (mav < (g_active_threshold * THRESHOLD_HYSTERESIS)) {
          s_current_hand_state = GESTURE_OPEN;
        }
      }

      // 3. Transmisión Inalámbrica Ultrarrápida ESP-NOW (<1 ms) al Receptor de la Prótesis
      esp_now_send_gesture(s_current_hand_state, raw_state, mav, rms);

      // 4. Enviar Paquete UDP de Telemetría a la PC (Monitor)
      if (WiFi.status() == WL_CONNECTED) {
        struct __attribute__((packed)) {
          uint8_t filtered_prediction;
          uint8_t raw_prediction;
          float mav;
          float rms;
        } packet;

        packet.filtered_prediction = s_current_hand_state;
        packet.raw_prediction      = raw_state;
        packet.mav                 = mav;
        packet.rms                 = rms;

        udp.beginPacket(udp_target_ip, UDP_PORT);
        udp.write((const uint8_t *)&packet, sizeof(packet));
        udp.endPacket();
      }

      // 5. Salida de Depuración Serial Periódica
      uint32_t now = millis();
      if (now - last_log_time >= 100) { // 10 Hz
        last_log_time = now;
        const char *estado_str = (s_current_hand_state == GESTURE_CLOSED) ? ">> CERRADA [PUÑO] <<" : "   ABIERTA [REPOSO]  ";
        
        // Barra visual en terminal
        int bar_len = (int)(mav / (g_active_threshold * 2.0f) * 20.0f);
        if (bar_len < 0) bar_len = 0;
        if (bar_len > 20) bar_len = 20;
        char bar[22];
        for (int b = 0; b < bar_len; b++) bar[b] = '#';
        for (int b = bar_len; b < 20; b++) bar[b] = '.';
        bar[20] = '\0';

        Serial.printf("[THRESHOLD] MAV: %6.2f | Umbral: %5.2f | [%s] | Mano: %s\n",
                      mav, g_active_threshold, bar, estado_str);
      }
    }
  }
}
