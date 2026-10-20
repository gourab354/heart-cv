# Webcam Heart Rate (rPPG) vs ECG

An experimental research project and real-time application comparing contactless webcam heart rate measurement (remote photoplethysmography / rPPG) against an ECG ground-truth reference using low-cost hardware.

Author: Gourab (Academy of Technology / Build Club)

---

## 1. Overview

The goal of this project is to evaluate how accurately a standard USB webcam can measure human heart rate from facial video compared to an electrical ECG reference, and to assess how factors like lighting, distance, and head motion impact measurement error.

### How rPPG Works
With each cardiac cycle, the heart pumps blood through the arterial vascular bed. The expanding capillary blood volume in facial skin increases light absorption, particularly in the green spectrum (~540-577 nm) due to oxyhemoglobin absorption peaks. A webcam captures these minute pulsatile intensity shifts. By isolating stable skin regions of interest (ROI) and filtering out sensor and motion noise, the volumetric blood pulse waveform is recovered without contacting the body.

---

## 2. Implemented Algorithms

The project implements and compares three rPPG algorithms under identical filtering bands (0.75 - 2.8 Hz / 45 - 168 BPM):

1. **POS (Plane-Orthogonal-to-Skin)**:
   Wang et al. (IEEE Trans. Biomed. Eng., 2017). Projects temporally normalized RGB color signals onto two orthogonal chrominance vectors on a 1.6-second sliding window with Hann overlap-add. Designed to cancel specular reflection and motion artifacts.
2. **CHROM (Chrominance-based method)**:
   de Haan and Jeanne (IEEE Trans. Biomed. Eng., 2013). Combines normalized chrominance difference signals with an adaptive standard deviation ratio.
3. **Green Channel Baseline**:
   Simple temporal bandpass on normalized green channel intensity.

### Signal Processing & Stability Pipeline
- **Temporal Landmark Smoothing**: 468 MediaPipe FaceMesh coordinates are smoothed with an exponential moving filter to prevent pixel-boundary flicker around the forehead and cheeks.
- **Uniform Timestamp Resampling**: Compensates for webcam frame rate jitter before filtering.
- **Zero-Phase Butterworth Filtering**: 3rd-order bandpass (0.75 to 2.8 Hz) using forward-backward filtering (`filtfilt`) to prevent phase distortion.
- **Harmonic Disambiguation**: Dicrotic notch reflections often create a prominent second harmonic ($2 \times f_0$). The peak detector evaluates candidate fundamental peaks ($f_0$) to prevent jumping between fundamental and harmonic rates (e.g., 60 BPM vs 120 BPM).
- **Physiological Slew-Rate Limiting**: Incorporates a rolling median window and limits rate-of-change to physiologically plausible resting limits (<= 2.5 BPM/sec), providing stable readings without lag.
- **Pulsatility Verification**: Multi-ROI cross-correlation between forehead and cheeks confirms arterial pulse synchronization and rejects static printed photos or frozen screens.

---

## 3. Hardware Architecture

For ground-truth validation, an ECG acquisition circuit was built using:
- **AD8232 Single-Lead ECG Front End**: Analog bio-potential amplifier with right-leg drive (RLD).
- **ESP32 DevKit**: Samples the AD8232 analog output at 250 Hz via 12-bit ADC (GPIO34) and streams microsecond timestamps and ADC values over USB Serial (115200 baud).
- **Webcam**: Standard 30 FPS USB camera (Zebronics HD).

### Wiring Diagram (AD8232 to ESP32)
- AD8232 GND -> ESP32 GND
- AD8232 3.3V -> ESP32 3V3 (Never 5V)
- AD8232 OUTPUT -> ESP32 GPIO34 (ADC1 input-only)
- AD8232 LO+ -> ESP32 GPIO32 (Lead-off detect)
- AD8232 LO- -> ESP32 GPIO33 (Lead-off detect)
- AD8232 SDN -> Unconnected (Module remains active)

> **Safety Warning**: The AD8232 electrodes touch the human body. During all recordings with electrodes attached, always disconnect the laptop from mains power and run on laptop battery to maintain galvanic isolation.

### Electrode Placement (Lead II Configuration)
- Red (RA): Right upper chest, below collarbone.
- Yellow (LA): Left upper chest, below collarbone.
- Green (RL): Lower right abdomen (reference / ground).

---

## 4. Repository Structure

```
├── firmware/
│   ├── esp32_ecg.ino        # ESP32 firmware for AD8232 streaming @ 250 Hz
│   └── esp32_max30100.ino   # Optional firmware for MAX30100 finger PPG sensor
├── main.py                  # Real-time OpenCV application with HUD & optional ECG stream
├── detector.py              # MediaPipe FaceMesh multi-ROI facial segmentation
├── rppg.py                  # rPPG signal engine, harmonic tracking, and filters
├── visualizer.py            # Real-time OpenCV HUD, oscilloscope, and telemetry
├── ecg_receiver.py          # Real-time background USB serial receiver for AD8232
├── rppg_lib.py              # Shared signal processing library (POS, CHROM, Pan-Tompkins)
├── capture.py               # Synchronized video + ECG recording script
├── extract_rppg.py          # Offline video processor: face video -> rgb.csv
├── analyze.py               # Statistical evaluation: MAE, RMSE, Bland-Altman, Wilcoxon
├── bench_test_ecg.py        # Standalone real-time ECG plotter for electrode testing
├── synth_test.py            # Synthetic dataset generator for pipeline validation
├── benchmark_simulator.py   # Sub-BPM mathematical precision validator
├── run.bat / run.ps1        # Quick-start launcher scripts for Windows
└── requirements.txt         # Python dependencies
```

---

## 5. Setup and Usage

### Prerequisites
Python 3.9 - 3.11 recommended.

Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Live Camera Detector
To run real-time camera tracking:
```bash
python main.py
```
*(Or double click `run.bat` on Windows)*

If the ESP32 ECG module is plugged into a USB port (e.g. COM5):
```bash
python main.py --ecg-port COM5
```
This enables dual-channel display: facial rPPG waveform and chest ECG reference are plotted simultaneously on the oscilloscope, showing real-time error between the two modalities.

#### Interactive Hotkeys:
- `R`: Reset buffer and recalibrate.
- `C`: Switch extraction algorithm (POS -> CHROM -> GREEN).
- `M`: Toggle ROI polygons overlay on/off.
- `S`: Save a timestamped snapshot of the telemetry view to `snapshots/`.
- `F`: Toggle fullscreen mode.
- `Q` or `ESC`: Exit.

---

## 6. Experimental Research Workflow

For collecting data across multiple subjects and conditions:

1. **Bench Test the Sensor**:
   ```bash
   python bench_test_ecg.py --port COM5
   ```
   Check that clean R-peaks appear without 50 Hz mains hum and lead-off is 0.

2. **Record Synchronized Session**:
   ```bash
   python capture.py --port COM5 --participant P01 --condition C1 --repeat 1 --duration 60
   ```
   Conditions:
   - C1: Bright light (lamp on), 40 cm distance, still
   - C2: Dim light, 40 cm distance, still
   - C3: Bright light, 70 cm distance, still
   - C4: Bright light, 40 cm distance, head movement

3. **Extract Facial Color Signals**:
   ```bash
   python extract_rppg.py data/P01/C1_R1 --preview
   ```

4. **Run Statistical Analysis & Plots**:
   ```bash
   python analyze.py --data-dir data --out results
   ```
   Generates:
   - `summary.csv`: MAE, RMSE, Pearson r, and bias across conditions.
   - `bland_altman.png`: Bland-Altman agreement plots with 95% limits of agreement.
   - `mae_by_condition.png`: Error comparisons by condition.
   - `stats.txt`: Wilcoxon signed-rank tests between methods.

### Testing Without Hardware
The entire analytical pipeline can be verified without connecting hardware using the built-in synthetic generator:
```bash
python synth_test.py
python analyze.py --data-dir synth_data --out synth_results
```

---

## 7. References

1. Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). Algorithmic principles of remote PPG. *IEEE Transactions on Biomedical Engineering*, 64(7), 1479-1491.
2. de Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. *IEEE Transactions on Biomedical Engineering*, 60(10), 2878-2886.
3. Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*, (3), 230-236.
4. Bland, J. M., & Altman, D. (1986). Statistical methods for assessing agreement between two methods of clinical measurement. *The Lancet*, 327(8476), 307-310.
