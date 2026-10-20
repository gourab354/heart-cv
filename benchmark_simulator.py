"""
Heart Vision Benchmark & Accuracy Validator.
Generates synthetic physiological pulse waveforms at ground-truth BPMs
to mathematically verify filtering, FFT, zero-padding, and parabolic interpolation accuracy.
"""

import time
import numpy as np
from rppg import RPPGProcessor


def run_benchmark(target_bpm: float = 72.4, duration_sec: float = 12.0, noise_std: float = 0.08):
    print("=" * 65)
    print(f"  BENCHMARK: Ground Truth Target Heart Rate = {target_bpm:.2f} BPM")
    print(f"  Duration: {duration_sec}s | Target FPS: 30.0 | Noise Level: {noise_std}")
    print("=" * 65)

    fs = 30.0
    processor = RPPGProcessor(
        buffer_seconds=9.0,
        min_seconds_for_bpm=3.5,
        target_fs=fs,
        algorithm="POS",
    )

    pulse_freq = target_bpm / 60.0  # In Hz
    total_samples = int(duration_sec * fs)

    # Base skin tone in RGB
    base_r = 175.0
    base_g = 140.0
    base_b = 120.0

    # PPG pulse amplitude modulation (capillary blood volume variations are ~0.5% - 2% of reflected light)
    amp = 1.6

    print("\nSimulating video frame stream...")
    start_time = 0.0

    final_metric = None
    for i in range(total_samples):
        t = i / fs

        # Realistic pulse signal: fundamental + dicrotic notch harmonic + slow DC respiration drift
        pulse = (
            np.sin(2 * np.pi * pulse_freq * t)
            + 0.35 * np.sin(4 * np.pi * pulse_freq * t + 0.5)
            + 0.5 * np.sin(2 * np.pi * 0.25 * t)  # 0.25 Hz respiration drift
            + np.random.normal(0, noise_std)
        )

        # Pulsatile variation is strongest in Green, moderate in Red, minimal in Blue
        r_val = base_r + (amp * 0.4 * pulse)
        g_val = base_g + (amp * 1.0 * pulse)
        b_val = base_b + (amp * 0.2 * pulse)

        metric = processor.update((r_val, g_val, b_val), timestamp=t, is_stable=True)
        final_metric = metric

        # Print progress every 3 seconds
        if i > 0 and i % int(3 * fs) == 0:
            print(f"  [{t:.1f}s] Status: {metric.status:10s} | Detected BPM: {metric.bpm:.2f} | Conf: {metric.confidence:.1f}%")

    print("\n" + "-" * 65)
    print("  FINAL ACCURACY REPORT")
    print("-" * 65)
    print(f"  Ground Truth Target : {target_bpm:.2f} BPM")
    print(f"  Measured BPM        : {final_metric.bpm:.2f} BPM")
    print(f"  Raw Peak BPM (FFT)  : {final_metric.bpm_raw:.2f} BPM")
    error = abs(final_metric.bpm - target_bpm)
    print(f"  Absolute Error      : {error:.2f} BPM")
    print(f"  Signal-to-Noise     : {final_metric.snr_db:+.1f} dB")
    print(f"  Confidence Score    : {final_metric.confidence:.1f}%")
    print(f"  Estimated HRV (RMSSD: {final_metric.hrv_rmssd:.1f} ms")
    print(f"  Accuracy Rating     : {'PASS (Clinical Precision < 1.0 BPM)' if error < 1.0 else 'CHECK PARAMETERS'}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="rPPG Heart Rate Benchmark")
    parser.add_argument("--bpm", type=float, default=74.2, help="Ground truth BPM to simulate")
    parser.add_argument("--noise", type=float, default=0.06, help="Noise standard deviation")
    args = parser.parse_args()

    run_benchmark(target_bpm=args.bpm, noise_std=args.noise)
