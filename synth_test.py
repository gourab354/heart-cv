#!/usr/bin/env python3
"""
synth_test.py - Generate Synthetic ECG & rPPG Dataset (Testing Without Hardware).
Prepared for Gourab - Build Club AOT.

Generates 3 synthetic participants x 4 conditions (C1, C2, C3, C4) x 2 repeats.
Allows running the entire research analysis pipeline and generating Bland-Altman plots
without needing the physical ESP32 or webcam.

Run:
  python synth_test.py
  python analyze.py --data-dir synth_data --out synth_results
"""

import csv
import json
import os
import numpy as np


def generate_synthetic_recording(out_dir, participant, condition, repeat, duration_s=60.0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(hash((participant, condition, repeat)) % (2**32))

    # Base heart rate between 62 and 82 BPM, with subtle respiratory sinus arrhythmia (RSA)
    base_bpm = rng.uniform(64.0, 78.0)
    fs_video = 30.0
    fs_ecg = 250.0

    n_frames = int(duration_s * fs_video)
    t_frames = np.linspace(0.0, duration_s, n_frames)

    # Condition noise profiles:
    # C1: Bright, 40cm, Still (Low noise)
    # C2: Dim, 40cm, Still (Higher sensor noise)
    # C3: Bright, 70cm, Still (Slightly lower SNR)
    # C4: Bright, 40cm, Head Movement (Low-frequency motion modulation)
    noise_level = 0.04
    motion_amp = 0.0
    if condition == "C2":
        noise_level = 0.12
    elif condition == "C3":
        noise_level = 0.08
    elif condition == "C4":
        noise_level = 0.07
        motion_amp = 0.25

    # 1. Generate ECG Signal (250 Hz)
    n_ecg = int(duration_s * fs_ecg)
    t_ecg = np.linspace(0.0, duration_s, n_ecg)

    # Simulated R-peaks times using instantaneous pulse frequency
    # Add subtle heart rate variation over time (drift of ~3 BPM)
    t_pulse = t_ecg
    freq_ecg = (base_bpm + 2.5 * np.sin(2 * np.pi * 0.05 * t_pulse)) / 60.0

    # Phase accumulator for exact frequency
    phase = 2 * np.pi * np.cumsum(freq_ecg) / fs_ecg
    r_peak_locs = np.where((phase % (2 * np.pi) < (2 * np.pi * freq_ecg / fs_ecg)))[0]

    ecg_adc = np.full(n_ecg, 2048.0) + rng.normal(0, 15.0, n_ecg)
    for p in r_peak_locs:
        # Sharp QRS spike
        spike_w = 4
        start = max(0, p - spike_w)
        end = min(n_ecg, p + spike_w)
        ecg_adc[start:end] += 1200.0 * np.exp(-0.5 * ((np.arange(start, end) - p) / 1.5) ** 2)

    leadoff = np.zeros(n_ecg, dtype=int)
    t_us = (t_ecg * 1e6).astype(np.int64)

    # Write ecg.csv
    with open(os.path.join(out_dir, "ecg.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t_us", "adc", "leadoff", "host_t"])
        for i in range(n_ecg):
            wr.writerow([t_us[i], int(ecg_adc[i]), leadoff[i], round(t_ecg[i], 4)])

    # Write frame_times.csv
    with open(os.path.join(out_dir, "frame_times.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "t"])
        for i, t in enumerate(t_frames):
            wr.writerow([i, round(t, 4)])

    # 2. Generate rPPG RGB Signal (30 Hz)
    phase_v = 2 * np.pi * np.cumsum((base_bpm + 2.5 * np.sin(2 * np.pi * 0.05 * t_frames)) / 60.0) / fs_video
    cardiac_wave = np.sin(phase_v) + 0.3 * np.sin(2 * phase_v)

    # Motion artifact in C4
    motion_wave = motion_amp * np.sin(2 * np.pi * 0.35 * t_frames) if motion_amp > 0 else 0.0

    # Base skin tone
    base_r = 175.0
    base_g = 140.0
    base_b = 120.0

    amp = 1.8
    r_sig = base_r + (amp * 0.4 * cardiac_wave) + motion_wave + rng.normal(0, noise_level * 15.0, n_frames)
    g_sig = base_g + (amp * 1.0 * cardiac_wave) + motion_wave + rng.normal(0, noise_level * 12.0, n_frames)
    b_sig = base_b + (amp * 0.2 * cardiac_wave) + motion_wave + rng.normal(0, noise_level * 15.0, n_frames)

    # Write rgb.csv
    with open(os.path.join(out_dir, "rgb.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t", "R", "G", "B"])
        for i in range(n_frames):
            wr.writerow([round(t_frames[i], 4), round(r_sig[i], 3), round(g_sig[i], 3), round(b_sig[i], 3)])

    # Write meta.json
    meta = {
        "participant": participant,
        "condition": condition,
        "repeat": repeat,
        "frames": n_frames,
        "fps_measured": 30.0,
        "ecg_samples": n_ecg,
        "ecg_expected": n_ecg,
        "leadoff_fraction": 0.0,
        "codec": "SYNTHETIC",
        "started": "2026-10-20 12:00:00",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    print("=" * 65)
    print("  Generating Synthetic Benchmark Dataset (3 Participants x 4 Conditions x 2 Repeats)")
    print("=" * 65)

    base_dir = "synth_data"
    participants = ["P01", "P02", "P03"]
    conditions = ["C1", "C2", "C3", "C4"]
    repeats = [1, 2]

    count = 0
    for p in participants:
        for c in conditions:
            for r in repeats:
                folder = os.path.join(base_dir, p, f"{c}_R{r}")
                generate_synthetic_recording(folder, p, c, r, duration_s=60.0)
                count += 1
                print(f"  [{count:02d}/24] Created {folder}")

    print("\n[+] Synthetic dataset created in 'synth_data/'!")
    print("    Now test the complete analysis pipeline with:")
    print("    python analyze.py --data-dir synth_data --out synth_results\n")


if __name__ == "__main__":
    main()
