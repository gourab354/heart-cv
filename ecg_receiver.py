"""
Real-time ESP32 AD8232 ECG Receiver for Heart Vision.
Streams 250 Hz ground-truth ECG data over USB Serial and calculates real-time ECG BPM.
Prepared for Gourab - Build Club AOT.
"""

from collections import deque
import threading
import time
from typing import List, Optional, Tuple
import numpy as np
from scipy import signal
import serial


class ECGReceiver:
    """
    Background worker that streams 250 Hz ADC data from ESP32 AD8232 module
    and extracts ground-truth ECG R-peaks in real time.
    """

    def __init__(self, port: str, baud: int = 115200, fs: float = 250.0):
        self.port = port
        self.baud = baud
        self.fs = fs
        self.ser: Optional[serial.Serial] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.lock = threading.Lock()

        # Rolling buffers (stores last ~12 seconds of ECG @ 250 Hz = 3000 samples)
        self.buffer_len = int(12.0 * fs)
        self.timestamps: Deque[float] = deque(maxlen=self.buffer_len)
        self.raw_adc: Deque[int] = deque(maxlen=self.buffer_len)
        self.leadoff_flags: Deque[int] = deque(maxlen=self.buffer_len)

        # Real-time state
        self.latest_bpm: float = 0.0
        self.last_r_peak_time: float = 0.0
        self.is_connected = False
        self.is_leadoff = False
        self.total_samples = 0

    def start(self) -> bool:
        """Connects to the ESP32 and begins streaming."""
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.2)
            time.sleep(2.0)  # Wait for ESP32 reboot on DTR toggle
            self.ser.reset_input_buffer()
            self.ser.write(b"S")  # Trigger ESP32 streaming
            self.running = True
            self.is_connected = True

            self.thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.thread.start()
            print(f"[+] Connected to AD8232 ECG on {self.port} at 250 Hz.")
            return True
        except Exception as e:
            print(f"[!] Could not connect to ECG on {self.port}: {e}")
            self.is_connected = False
            return False

    def stop(self):
        """Stops streaming and releases the COM port."""
        self.running = False
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(b"X")
                time.sleep(0.2)
                self.ser.close()
            except Exception:
                pass
        self.is_connected = False

    def _reader_loop(self):
        t0 = time.perf_counter()
        while self.running and self.ser and self.ser.is_open:
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) != 3:
                    continue

                t_us = int(parts[0])
                adc = int(parts[1])
                leadoff = int(parts[2])

                t_now = t_us / 1e6

                with self.lock:
                    self.timestamps.append(t_now)
                    self.raw_adc.append(adc)
                    self.leadoff_flags.append(leadoff)
                    self.is_leadoff = leadoff == 1
                    self.total_samples += 1

            except Exception:
                time.sleep(0.01)

    def get_recent_waveform(self, num_samples: int = 250) -> List[float]:
        """Returns normalized filtered ECG trace for oscilloscope rendering."""
        with self.lock:
            if len(self.raw_adc) < 50:
                return []
            samples = list(self.raw_adc)[-num_samples:]

        arr = np.array(samples, dtype=float)
        # Quick baseline removal
        arr = arr - np.median(arr)
        # Normalization
        std = np.std(arr)
        if std > 1e-4:
            arr = arr / (std * 3.0)
        return arr.tolist()

    def get_bpm(self) -> Tuple[float, bool]:
        """
        Extracts current ground-truth ECG BPM using Pan-Tompkins QRS detection.
        Returns: (bpm, is_beat)
        """
        with self.lock:
            if len(self.raw_adc) < int(4.0 * self.fs):
                return 0.0, False
            t_arr = np.array(self.timestamps)
            adc_arr = np.array(self.raw_adc, dtype=float)
            lo_arr = np.array(self.leadoff_flags)

        # Discard if electrode detached > 15% of recent window
        if np.mean(lo_arr[-int(3.0 * self.fs) :]) > 0.15:
            return 0.0, False

        # Bandpass filter 5-15 Hz (isolate QRS complex)
        nyq = self.fs * 0.5
        b, a = signal.butter(2, [5.0 / nyq, 15.0 / nyq], btype="band")
        ecg_filt = signal.filtfilt(b, a, adc_arr - np.median(adc_arr))

        # Derivative + squaring + moving window integration
        d = np.gradient(ecg_filt)
        mwi_w = max(1, int(0.12 * self.fs))
        mwi = np.convolve(d**2, np.ones(mwi_w) / mwi_w, mode="same")

        thr = 0.35 * np.percentile(mwi, 98)
        peaks, _ = signal.find_peaks(mwi, height=thr, distance=int(0.3 * self.fs))

        if len(peaks) < 3:
            return 0.0, False

        # Compute BPM from recent RR intervals
        r_times = t_arr[peaks]
        rr = np.diff(r_times)
        valid_rr = rr[(rr >= 0.32) & (rr <= 1.4)]  # 43 to 187 BPM

        if len(valid_rr) < 2:
            return 0.0, False

        bpm = 60.0 / float(np.mean(valid_rr[-5:]))
        self.latest_bpm = bpm

        # Check if an R-peak occurred in the latest 10 samples
        is_beat = (len(adc_arr) - 1 - peaks[-1]) <= 6

        return round(bpm, 1), is_beat
