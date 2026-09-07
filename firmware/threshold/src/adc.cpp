/*
 * ============================================================
 *  adc.cpp — Implementacion del ADC externo MCP3208
 * ============================================================
 */

#include "adc.h"
#include "config.h"
#include <SPI.h>

static SPISettings SPI_SETTINGS(1000000, MSBFIRST, SPI_MODE0);

void adcInit() {
  SPI.begin(PIN_SPI_SCK, PIN_SPI_MISO, PIN_SPI_MOSI, PIN_CS);
  pinMode(PIN_CS, OUTPUT);
  digitalWrite(PIN_CS, HIGH);
}

int readADC(int channel) {
  // Protocolo MCP3208: Start bit + SGL + D2 D1 D0 del canal
  byte b1 = 0x06 | (channel >> 2);
  byte b2 = (channel & 0x03) << 6;

  SPI.beginTransaction(SPI_SETTINGS);
  digitalWrite(PIN_CS, LOW);
  SPI.transfer(b1);
  byte hi = SPI.transfer(b2);
  byte lo = SPI.transfer(0x00);
  digitalWrite(PIN_CS, HIGH);
  SPI.endTransaction();

  return ((hi & 0x0F) << 8) | lo;  // 0-4095
}
