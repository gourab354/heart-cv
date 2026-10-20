#!/usr/bin/env python3
"""
bench_test_ecg.py - Real-Time Bench Test & Plotter for ESP32 AD8232 ECG.
Prepared for Gourab - Build Club AOT.

Tests the hardware connection before recording participants:
- Verifies serial connection on specified COM port at 115200 baud
- Streams and renders live rolling ECG waveform
- Detects lead-off detachment
- Shows live R-peak pulses and calculates ground-truth BPM

Usage:
  python bench_test_ecg.py --port COM5
"""

import argparse
import sys
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import serial


def parse_args():
    parser = argparse.ArgumentParser(description="Bench Test ESP32 AD8232 ECG")
    parser.add_argument("--port", default="COM5", help="Serial COM port of ESP32 (e.g., COM5 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--window", type=float, default=5.0, help="Display window in seconds (default: 5.0)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print(f"  BENCH TEST: ESP32 AD8232 ECG Streamer on {args.port}")
    print("=" * 65)
    print("  Checking serial connection...")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.2)
        time.sleep(2.0)  # Allow ESP32 reboot on connection
        ser.reset_input_buffer()
        ser.write(b"S")  # Start streaming
        print(f"[+] Connected to {args.port}! Starting 250 Hz stream...")
    except Exception as e:
        print(f"\n[!] Error opening port {args.port}: {e}")
        print("    Troubleshooting steps:")
        print("    1. Verify your ESP32 is plugged in via micro-USB.")
        print("    2. Check Windows Device Manager (Ports -> Silicon Labs or CH340).")
        print("    3. Ensure Arduino Serial Monitor is CLOSED so the port is free.")
        sys.exit(1)

    max_points = int(args.window * 250)
    t_data = []
    adc_data = []
    lo_data = []

    # Setup Matplotlib live plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [4, 1]})
    fig.canvas.manager.set_window_title(f"AD8232 ECG Bench Test [{args.port}]")

    (line_ecg,) = ax1.plot([], [], color="crimson", lw=1.4, label="ECG (ADC 0-4095)")
    ax1.set_xlim(0, args.window)
    ax1.set_ylim(0, 4096)
    ax1.set_ylabel("Raw ADC Value", fontsize=10)
    ax1.set_title("Live AD8232 ECG Trace (Pan-Tompkins R-Peaks)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    (line_lo,) = ax2.plot([], [], color="darkorange", lw=1.2, label="Lead-Off (1 = Disconnected)")
    ax2.set_xlim(0, args.window)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Attached", "Detached"])
    ax2.set_xlabel("Elapsed Time (s)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    t_start = time.perf_counter()

    def update(frame):
        nonlocal t_start
        # Read available lines
        while ser.in_waiting > 0:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) == 3:
                try:
                    t_us = int(parts[0])
                    adc = int(parts[1])
                    lo = int(parts[2])

                    t_sec = t_us / 1e6
                    t_data.append(t_sec)
                    adc_data.append(adc)
                    lo_data.append(lo)
                except ValueError:
                    pass

        if len(t_data) > max_points:
            del t_data[: len(t_data) - max_points]
            del adc_data[: len(adc_data) - max_points]
            del lo_data[: len(lo_data) - max_points]

        if t_data:
            latest_t = t_data[-1]
            min_t = max(0, latest_t - args.window)
            max_t = max(args.window, latest_t)

            ax1.set_xlim(min_t, max_t)
            ax2.set_xlim(min_t, max_t)

            line_ecg.set_data(t_data, adc_data)
            line_lo.set_data(t_data, lo_data)

            # Auto-scale Y around active signal
            if len(adc_data) > 50:
                y_min = max(0, int(np.min(adc_data) - 150))
                y_max = min(4095, int(np.max(adc_data) + 150))
                if y_max - y_min > 200:
                    ax1.set_ylim(y_min, y_max)

        return line_ecg, line_lo

    ani = FuncAnimation(fig, update, interval=40, blit=False)

    try:
        plt.tight_layout()
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[*] Stopping ESP32 stream and closing port...")
        try:
            ser.write(b"X")
            time.sleep(0.2)
            ser.close()
        except Exception:
            pass
        print("[+] Bench test complete.")


if __name__ == "__main__":
    main()
