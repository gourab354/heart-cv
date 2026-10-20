# Heart Vision: Quick-Start Guide (Camera Mode)

A guide for setting up and running the camera-based heart rate measurement system.

---

## 1. How to Launch

### Option A: Windows Launcher
Double-click `run.bat` or `run.ps1`.

### Option B: Terminal Command
Open PowerShell or Command Prompt in the project folder and run:
```powershell
python main.py
```

---

## 2. On-Screen Display Overview

The application displays a split-view window (1280x720):

### Left: Live Camera View
- **Face Reticle**: Corner brackets indicate successful face tracking.
- **Vascular ROIs**: Shaded regions highlight the forehead and upper cheeks where blood volume variation is strongest.
- **Status Indicators (Top Banner)**:
  - `* ACQUIRING PULSE BUFFER...`: Collecting initial 4 to 5 seconds of frames to establish frequency baseline.
  - `* VERIFIED LIVE HUMAN PULSE`: Continuous cardiac pulse detected and validated.
  - `! FAKE / STATIC PHOTO DETECTED`: Anti-spoofing alert (triggered if a still image or phone screen without living pulse modulation is shown).
  - `! MOTION DETECTED - HOLD STILL`: Head movement flagged; reading stabilizes when movement stops.

### Right: Telemetry Panel
1. **Heart Rate Display**: Live BPM readout with an animated heart indicator pulsing on systolic peaks.
2. **Signal Quality**: Live Signal-to-Noise Ratio (SNR in dB) and confidence percentage.
3. **PPG Oscilloscope**: Scrolling waveform showing the filtered blood volume pulse in real time.
4. **Spectral Density (FFT)**: Frequency spectrum plot indicating the dominant cardiac peak across the 45-168 BPM band.
5. **Biometrics**: Heart Rate Variability (HRV / RMSSD in ms), Inter-Beat Interval (IBI in ms), and session Min/Avg/Max tracking.

---

## 3. Keyboard Hotkeys

| Key | Action | Description |
| :---: | :--- | :--- |
| **`R`** | Reset / Recalibrate | Clears the temporal buffer. Use when changing seats or lighting. |
| **`C`** | Cycle Algorithm | Toggles between POS, CHROM, and Green channel methods. |
| **`M`** | Toggle ROIs | Turns forehead and cheek overlay polygons on/off. |
| **`S`** | Save Snapshot | Saves current view and telemetry to the `snapshots/` folder. |
| **`F`** | Fullscreen | Toggles fullscreen display. |
| **`Q`** or **`ESC`** | Quit | Releases camera resources and exits cleanly. |

---

## 4. Best Practices for Accurate Readings

Because remote PPG detects light absorption changes of less than 1 percent:
- **Lighting**: Use consistent, diffuse lighting from the front. Avoid flickering fluorescent tubes or heavy backlighting.
- **Distance**: Sit approximately 40 to 70 cm from the webcam so the face occupies 30-40% of the frame.
- **Stillness**: Keep head movement minimal during measurements.
- **Buffer Warmup**: Allow 5 to 8 seconds of continuous tracking for the rolling filter to stabilize.

---

## 5. Connecting the Hardware Sensor Later

When the AD8232 ECG module and ESP32 are ready:
1. Flash `firmware/esp32_ecg.ino` to the ESP32.
2. Connect the ESP32 via USB and note the COM port (e.g. COM5).
3. Run:
   ```powershell
   python main.py --ecg-port COM5
   ```
   The application will display both the facial rPPG trace and the electrical ECG trace simultaneously and report live error between them.
