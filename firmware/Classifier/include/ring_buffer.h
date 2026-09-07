#pragma once
/*
 * ============================================================
 *  ring_buffer.h — SPSC Lock-Free Ring Buffer (Core 1 → Core 0)
 * ============================================================
 */

#include <cstdint>
#include "config.h"

// Muestra individual con marca de tiempo de microsegundos
struct RingSample {
  uint32_t timestamp_us;
  float filtered;
};

// Estado global del buffer circular en SRAM interna
extern RingSample ring[RING_SIZE];
extern volatile uint32_t ring_head;
extern volatile uint32_t ring_tail;
extern volatile uint32_t g_dropped;
