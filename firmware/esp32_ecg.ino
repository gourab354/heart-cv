/*
 * esp32_ecg.ino - AD8232 ECG streamer for the rPPG-vs-ECG project
 * Prepared for Gourab - Build Club AOT
 *
 * Wiring (ESP32 DevKit):
 *   AD8232 GND    -> ESP32 GND
 *   AD8232 3.3V   -> ESP32 3V3  (NEVER 5V!)
 *   AD8232 OUTPUT -> ESP32 GPIO34 (ADC1 input-only, works with Wi-Fi off)
 *   AD8232 LO+    -> ESP32 GPIO32 (Lead-off detection)
 *   AD8232 LO-    -> ESP32 GPIO33 (Lead-off detection)
 *   AD8232 SDN    -> Leave open / unconnected (module is always ON)
 *
 * Protocol (115200 baud, USB serial):
 *   PC -> ESP32 : 'S' start streaming (t = 0 at this moment)
 *                 'X' stop streaming
 *   ESP32 -> PC : lines "#..." are comments (ignored by PC)
 *                 data lines: t_us,adc,leadoff
 *                 t_us    microseconds since 'S' was received
 *                 adc     raw 12-bit reading (0..4095)
 *                 leadoff 1 if an electrode is off, else 0
 *
 * Sampling rate: 250 Hz (timer-free, micros()-based scheduling).
 * Wi-Fi and Bluetooth kept OFF for lowest ADC noise.
 */

#include <Arduino.h>

const int ECG_PIN = 34;
const int LO_PLUS_PIN = 32;
const int LO_MINUS_PIN = 33;

const uint32_t FS = 250;                     // samples per second
const uint32_t PERIOD_US = 1000000UL / FS;   // 4000 us

bool running = false;
uint32_t t0_us = 0;
uint32_t next_us = 0;

void setup() {
    Serial.begin(115200);
    analogReadResolution(12);
    analogSetPinAttenuation(ECG_PIN, ADC_11db); // full 0..3.3 V range
    pinMode(LO_PLUS_PIN, INPUT);
    pinMode(LO_MINUS_PIN, INPUT);
    Serial.println("#READY esp32_ecg 250Hz");
}

void loop() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == 'S') {
            running = true;
            t0_us = micros();
            next_us = t0_us; // first sample immediately
            Serial.println("#STARTED");
        } else if (c == 'X') {
            running = false;
            Serial.println("#STOPPED");
        }
    }

    if (!running) return;

    uint32_t now = micros();
    if ((int32_t)(now - next_us) >= 0) { // wrap-safe comparison
        next_us += PERIOD_US;
        int leadoff = digitalRead(LO_PLUS_PIN) | digitalRead(LO_MINUS_PIN);
        int adc = analogRead(ECG_PIN);

        Serial.print(now - t0_us);
        Serial.print(',');
        Serial.print(adc);
        Serial.print(',');
        Serial.println(leadoff);
    }
}
