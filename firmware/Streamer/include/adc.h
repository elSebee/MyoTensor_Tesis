#pragma once
/*
 * ============================================================
 *  adc.h — Interfaz del ADC externo MCP3208
 * ============================================================
 *  Chip : MCP3208 — ADC de 12 bits, 8 canales, SPI
 *  Bus  : SPI @ 1MHz, MSBFIRST, MODE0
 * ============================================================
 */

#include <Arduino.h>

// Inicializa el bus SPI y el pin CS
void adcInit();

// Devuelve la lectura cruda del canal indicado (0-7)
// Retorno: entero 0-4095 (12 bits)
int readADC(int channel);
