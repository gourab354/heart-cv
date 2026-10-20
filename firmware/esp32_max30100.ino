/*
 * esp32_max30100.ino - MAX30100 Optical Finger PPG Streamer for ESP32
 * Prepared for Gourab - Build Club AOT
 *
 * Wiring (ESP32 DevKit to MAX30100 module):
 *   MAX30100 VIN -> ESP32 3V3
 *   MAX30100 GND -> ESP32 GND
 *   MAX30100 SDA -> ESP32 GPIO21 (I2C SDA)
 *   MAX30100 SCL -> ESP32 GPIO22 (I2C SCL)
 *   MAX30100 INT -> Leave open or GPIO19
 *
 * Output protocol (115200 baud):
 *   t_us,ir_raw,red_raw
 */

#include <Wire.h>
#include <Arduino.h>

#define MAX30100_ADDR 0x57

bool running = false;
uint32_t t0_us = 0;

void writeRegister(uint8_t address, uint8_t data) {
    Wire.beginTransmission(MAX30100_ADDR);
    Wire.write(address);
    Wire.write(data);
    Wire.endTransmission();
}

void initMAX30100() {
    Wire.begin(21, 22, 400000); // 400 kHz Fast I2C

    // Reset module
    writeRegister(0x06, 0x40);
    delay(100);

    // Mode: SpO2 mode (both Red and IR LEDs enabled)
    writeRegister(0x06, 0x03);

    // SpO2 config: 100 Hz sample rate, 1600 us pulse width (16-bit resolution)
    writeRegister(0x07, 0x07);

    // LED current: IR ~27mA, RED ~27mA
    writeRegister(0x09, 0x88);

    Serial.println("#READY esp32_max30100 100Hz");
}

void readFIFO(uint16_t &ir, uint16_t &red) {
    Wire.beginTransmission(MAX30100_ADDR);
    Wire.write(0x05); // FIFO data register
    Wire.endTransmission(false);

    Wire.requestFrom((uint8_t)MAX30100_ADDR, (uint8_t)4);
    if (Wire.available() >= 4) {
        ir = (Wire.read() << 8) | Wire.read();
        red = (Wire.read() << 8) | Wire.read();
    }
}

void setup() {
    Serial.begin(115200);
    delay(500);
    initMAX30100();
}

void loop() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == 'S') {
            running = true;
            t0_us = micros();
            Serial.println("#STARTED");
        } else if (c == 'X') {
            running = false;
            Serial.println("#STOPPED");
        }
    }

    if (!running) return;

    static uint32_t last_sample = 0;
    uint32_t now = micros();
    if (now - last_sample >= 10000) { // 100 Hz (10 ms)
        last_sample = now;

        uint16_t ir = 0, red = 0;
        readFIFO(ir, red);

        Serial.print(now - t0_us);
        Serial.print(',');
        Serial.print(ir);
        Serial.print(',');
        Serial.println(red);
    }
}
