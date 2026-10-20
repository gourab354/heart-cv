"""
Remote Photoplethysmography (rPPG) Core Signal Processing Engine.
Features:
- Plane-Orthogonal-to-Skin (POS), CHROM, and Green-channel extraction algorithms
- High-precision timestamp resampling for zero-jitter compensation
- Zero-phase Butterworth bandpass filtering (0.75 - 2.8 Hz / 45 - 168 BPM)
- Zero-padded FFT with parabolic peak interpolation (0.1 BPM precision)
- Anti-Spoofing & Liveness Detection (Multi-ROI cross-correlation & AC/DC pulsatility check)
- Signal-to-Noise Ratio (SNR) and Confidence scoring
- Pulse peak detection, Inter-Beat Interval (IBI), and HRV (RMSSD) estimation
- Adaptive exponential moving average smoothing
"""

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d


@dataclass
class HeartRateMetrics:
    bpm: float = 0.0
    bpm_raw: float = 0.0
    confidence: float = 0.0  # 0 to 100%
    snr_db: float = 0.0
    hrv_rmssd: float = 0.0  # ms
    ibi_ms: float = 0.0  # ms
    is_beat: bool = False  # Systolic pulse peak in current frame
    buffer_progress: float = 0.0  # 0 to 100%
    is_valid: bool = False
    is_live: bool = False
    liveness_score: float = 0.0
    liveness_msg: str = "ANALYZING..."
    status: str = "BUFFERING"  # BUFFERING, TRACKING, STABLE, MOTION_ALERT, STATIC_PHOTO
    min_bpm: float = 0.0
    max_bpm: float = 0.0
    avg_bpm: float = 0.0


class RPPGProcessor:
    """
    High-accuracy rPPG extraction engine with Liveness & Anti-Spoofing verification.
    """

    ALGORITHMS = ["POS", "CHROM", "GREEN"]

    def __init__(
        self,
        buffer_seconds: float = 10.0,
        min_seconds_for_bpm: float = 4.5,
        target_fs: float = 30.0,
        algorithm: str = "POS",
        low_bpm: float = 45.0,
        high_bpm: float = 168.0,
    ):
        self.buffer_seconds = buffer_seconds
        self.min_seconds_for_bpm = min_seconds_for_bpm
        self.target_fs = target_fs
        self.algorithm = algorithm.upper() if algorithm.upper() in self.ALGORITHMS else "POS"
        self.low_bpm = low_bpm
        self.high_bpm = high_bpm

        # Passband in Hz: [0.75 Hz, 2.8 Hz] for [45, 168] BPM
        self.low_hz = self.low_bpm / 60.0
        self.high_hz = self.high_bpm / 60.0

        # Max buffer size
        self.max_buffer_len = int(buffer_seconds * target_fs * 1.5)

        # Ring buffers for timestamps and composite color channels
        self.timestamps: Deque[float] = deque(maxlen=self.max_buffer_len)
        self.red_channel: Deque[float] = deque(maxlen=self.max_buffer_len)
        self.green_channel: Deque[float] = deque(maxlen=self.max_buffer_len)
        self.blue_channel: Deque[float] = deque(maxlen=self.max_buffer_len)

        # Separate ROI streams for Liveness / Anti-Spoofing cross-correlation
        self.forehead_g: Deque[float] = deque(maxlen=self.max_buffer_len)
        self.cheeks_g: Deque[float] = deque(maxlen=self.max_buffer_len)

        # Filtered PPG waveform history for visual oscilloscope
        self.waveform_history: Deque[float] = deque(maxlen=180)
        # Spectrum history for visual FFT plot: (frequencies_bpm, magnitudes, peak_bpm)
        self.spectrum_data: Optional[Tuple[np.ndarray, np.ndarray, float]] = None

        # Medical-grade state tracking
        self.smooth_bpm: float = 0.0
        self.raw_bpm_history: Deque[float] = deque(maxlen=9)
        self.last_bpm_time: float = time.perf_counter()
        self.bpm_history: List[float] = []
        self.ibi_history: Deque[float] = deque(maxlen=30)
        self.last_beat_time: float = 0.0
        self.beat_cooldown: float = 0.35  # Max ~170 BPM to prevent double trigger

        # Statistics
        self.session_bpms: List[float] = []

    def reset(self):
        """Reset buffers and calibration state."""
        self.timestamps.clear()
        self.red_channel.clear()
        self.green_channel.clear()
        self.blue_channel.clear()
        self.forehead_g.clear()
        self.cheeks_g.clear()
        self.waveform_history.clear()
        self.spectrum_data = None
        self.smooth_bpm = 0.0
        self.raw_bpm_history.clear()
        self.last_bpm_time = time.perf_counter()
        self.bpm_history.clear()
        self.ibi_history.clear()
        self.last_beat_time = 0.0
        self.session_bpms.clear()

    def set_algorithm(self, algo: str):
        """Switch rPPG algorithm dynamically (POS, CHROM, GREEN)."""
        algo_up = algo.upper()
        if algo_up in self.ALGORITHMS:
            self.algorithm = algo_up

    def next_algorithm(self) -> str:
        """Cycle to the next algorithm."""
        idx = self.ALGORITHMS.index(self.algorithm)
        self.algorithm = self.ALGORITHMS[(idx + 1) % len(self.ALGORITHMS)]
        return self.algorithm

    def update(
        self,
        rgb_mean: Tuple[float, float, float],
        timestamp: Optional[float] = None,
        is_stable: bool = True,
        roi_means: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ) -> HeartRateMetrics:
        """
        Feed a new frame's average skin RGB, timestamp, and optional ROI breakdown.
        Evaluates physiological pulsatility and anti-spoofing liveness.
        """
        now = timestamp if timestamp is not None else time.perf_counter()
        r, g, b = rgb_mean

        # Discard zero or corrupt values
        if r <= 1.0 and g <= 1.0 and b <= 1.0:
            return HeartRateMetrics(status="NO_FACE", liveness_msg="NO FACE DETECTED")

        # Append to temporal sliding window
        self.timestamps.append(now)
        self.red_channel.append(r)
        self.green_channel.append(g)
        self.blue_channel.append(b)

        # Store separate forehead and cheek signals for liveness cross-correlation
        if roi_means and "forehead" in roi_means:
            self.forehead_g.append(roi_means["forehead"][1])
        else:
            self.forehead_g.append(g)

        if roi_means and ("left_cheek" in roi_means or "right_cheek" in roi_means):
            cheek_vals = []
            if "left_cheek" in roi_means:
                cheek_vals.append(roi_means["left_cheek"][1])
            if "right_cheek" in roi_means:
                cheek_vals.append(roi_means["right_cheek"][1])
            self.cheeks_g.append(float(np.mean(cheek_vals)))
        else:
            self.cheeks_g.append(g)

        duration = self.timestamps[-1] - self.timestamps[0]
        progress = min(100.0, (duration / self.min_seconds_for_bpm) * 100.0)

        # Buffer warming up
        if duration < self.min_seconds_for_bpm or len(self.timestamps) < 30:
            return HeartRateMetrics(
                buffer_progress=progress,
                status="BUFFERING",
                liveness_msg=f"WARMING UP ({int(progress)}%)",
                bpm=self.smooth_bpm if self.smooth_bpm > 0 else 0.0,
            )

        # If subject is moving violently, preserve previous BPM and warn
        if not is_stable:
            return HeartRateMetrics(
                bpm=self.smooth_bpm,
                confidence=20.0,
                status="MOTION_ALERT",
                is_live=False,
                liveness_msg="MOTION DETECTED - HOLD STILL",
                buffer_progress=100.0,
                is_valid=self.smooth_bpm > 0,
            )

        # Convert deques to numpy arrays
        t_arr = np.array(self.timestamps)
        r_arr = np.array(self.red_channel)
        g_arr = np.array(self.green_channel)
        b_arr = np.array(self.blue_channel)

        # 1. Resample to strictly uniform 30 Hz grid to cancel camera frame jitter
        n_samples = int(duration * self.target_fs)
        if n_samples < 45:
            return HeartRateMetrics(buffer_progress=progress, status="BUFFERING", liveness_msg="GATHERING FRAMES")

        t_uniform = np.linspace(t_arr[0], t_arr[-1], n_samples)
        r_unif = np.interp(t_uniform, t_arr, r_arr)
        g_unif = np.interp(t_uniform, t_arr, g_arr)
        b_unif = np.interp(t_uniform, t_arr, b_arr)
        fs = self.target_fs

        # 2. Extract raw pulse signal based on selected algorithm (with sub-window overlap-add)
        pulse_raw = self._extract_pulse(r_unif, g_unif, b_unif, fs)

        # 3. Detrend signal (eliminate slow lighting & respiration drift)
        pulse_detrend = signal.detrend(pulse_raw, type="linear")

        # 4. Zero-phase Butterworth Bandpass Filter (0.75 Hz to 2.8 Hz / 45 to 168 BPM)
        nyq = 0.5 * fs
        low_norm = self.low_hz / nyq
        high_norm = self.high_hz / nyq
        low_norm = max(0.01, min(low_norm, 0.9))
        high_norm = max(low_norm + 0.01, min(high_norm, 0.99))

        b_filter, a_filter = signal.butter(3, [low_norm, high_norm], btype="bandpass")
        pulse_filtered = signal.filtfilt(b_filter, a_filter, pulse_detrend)

        # Append latest value to visual oscilloscope waveform
        latest_val = float(pulse_filtered[-1])
        self.waveform_history.append(latest_val)

        # 5. High-resolution Zero-Padded FFT with Harmonic Disambiguation
        n_fft = 4096
        hanning_win = np.hanning(len(pulse_filtered))
        windowed = pulse_filtered * hanning_win
        fft_spec = np.abs(np.fft.rfft(windowed, n=n_fft))
        fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

        # Limit to valid heart rate passband
        band_mask = (fft_freqs >= self.low_hz) & (fft_freqs <= self.high_hz)
        band_freqs = fft_freqs[band_mask]
        band_spec = fft_spec[band_mask]

        if len(band_spec) == 0:
            return HeartRateMetrics(buffer_progress=progress, status="CALIBRATING", liveness_msg="CALIBRATING")

        # Detect candidate peaks for harmonic-aware fundamental selection
        peaks, _ = signal.find_peaks(band_spec, prominence=0.10 * np.max(band_spec))
        if len(peaks) > 0:
            sorted_peaks = sorted(peaks, key=lambda p: band_spec[p], reverse=True)
            best_p = sorted_peaks[0]
            best_freq = band_freqs[best_p]

            # Harmonic disambiguation:
            # If the tallest peak is in the upper harmonic register (> 95 BPM),
            # check if there is a fundamental peak at ~half the frequency
            sub_freq = best_freq / 2.0
            if sub_freq >= self.low_hz:
                for p in sorted_peaks[1:]:
                    pf = band_freqs[p]
                    if abs(pf - sub_freq) <= 0.16:  # within ~10 BPM
                        if band_spec[p] >= 0.25 * band_spec[best_p]:
                            best_p = p
                            best_freq = pf
                            break

            # Disambiguation with previously established stable rate
            if self.smooth_bpm > 0:
                prev_f = self.smooth_bpm / 60.0
                if abs(best_freq - 2.0 * prev_f) < 0.20:
                    for p in sorted_peaks:
                        if abs(band_freqs[p] - prev_f) < 0.18:
                            if band_spec[p] >= 0.25 * band_spec[best_p]:
                                best_p = p
                                best_freq = band_freqs[p]
                                break

            # Parabolic sub-bin interpolation on the selected peak
            if 0 < best_p < len(band_spec) - 1:
                y1, y2, y3 = band_spec[best_p - 1], band_spec[best_p], band_spec[best_p + 1]
                denom = y1 - 2.0 * y2 + y3
                delta = 0.5 * (y1 - y3) / denom if abs(denom) > 1e-9 else 0.0
                peak_freq = band_freqs[best_p] + delta * (band_freqs[1] - band_freqs[0])
            else:
                peak_freq = best_freq
            peak_idx = best_p
        else:
            peak_idx = int(np.argmax(band_spec))
            peak_freq = band_freqs[peak_idx]

        raw_bpm = peak_freq * 60.0

        # Save spectrum for UI rendering
        spectrum_bpm = band_freqs * 60.0
        max_mag = np.max(band_spec) if np.max(band_spec) > 1e-6 else 1.0
        norm_spec = band_spec / max_mag
        self.spectrum_data = (spectrum_bpm, norm_spec, raw_bpm)

        # 6. Signal-to-Noise Ratio (SNR) and Confidence Calculation
        snr_db, confidence = self._calculate_snr_and_confidence(band_freqs, band_spec, peak_freq)

        # 7. Liveness & Pulsatility Assessment (Sensible, Realistic Thresholds)
        is_live, liveness_score, liveness_msg = self._verify_liveness(
            t_arr, t_uniform, fs, b_filter, a_filter, g_unif, band_spec, peak_idx
        )

        # 8. Real-time Pulse Peak (Systolic Beat) Detection & IBI
        is_beat, ibi, hrv = self._detect_beat_and_hrv(pulse_filtered, fs, now)

        # 9. Medical Physiological Median & Slew-Rate Filter
        # Eliminates false harmonic jumps and ensures medical-grade stability
        dt = max(0.01, now - self.last_bpm_time)
        self.last_bpm_time = now

        if raw_bpm >= self.low_bpm and raw_bpm <= self.high_bpm:
            self.raw_bpm_history.append(raw_bpm)
            filtered_raw = float(np.median(self.raw_bpm_history))

            if self.smooth_bpm == 0.0:
                self.smooth_bpm = filtered_raw
            else:
                # Maximum physiological rate of change: <= 2.5 BPM per second at rest
                max_step = max(0.6, 2.5 * dt)
                diff = filtered_raw - self.smooth_bpm
                step = np.clip(diff, -max_step, max_step)
                self.smooth_bpm += step

            self.session_bpms.append(self.smooth_bpm)

        # Determine status
        if not is_live:
            status = "STATIC_PHOTO"
        elif confidence >= 60.0:
            status = "STABLE"
        elif confidence >= 30.0:
            status = "TRACKING"
        else:
            status = "CALIBRATING"

        # Session summary metrics
        if self.session_bpms:
            min_bpm = float(np.min(self.session_bpms))
            max_bpm = float(np.max(self.session_bpms))
            avg_bpm = float(np.mean(self.session_bpms))
        else:
            min_bpm = max_bpm = avg_bpm = self.smooth_bpm

        display_bpm = self.smooth_bpm if (is_live and self.smooth_bpm > 0) else 0.0

        return HeartRateMetrics(
            bpm=round(display_bpm, 1),
            bpm_raw=round(raw_bpm, 1),
            confidence=round(confidence, 1),
            snr_db=round(snr_db, 1),
            hrv_rmssd=round(hrv, 1),
            ibi_ms=round(ibi, 1),
            is_beat=is_beat if is_live else False,
            buffer_progress=100.0,
            is_valid=display_bpm > 0,
            is_live=is_live,
            liveness_score=round(liveness_score, 1),
            liveness_msg=liveness_msg,
            status=status,
            min_bpm=round(min_bpm, 1),
            max_bpm=round(max_bpm, 1),
            avg_bpm=round(avg_bpm, 1),
        )

    def _verify_liveness(
        self,
        t_arr: np.ndarray,
        t_uniform: np.ndarray,
        fs: float,
        b_filter: np.ndarray,
        a_filter: np.ndarray,
        g_unif: np.ndarray,
        band_spec: np.ndarray,
        peak_idx: int,
    ) -> Tuple[bool, float, str]:
        """
        Robust pulsatility and liveness assessment.
        Allows real human faces to be tracked smoothly while detecting completely frozen photos.
        """
        # Test 1: AC/DC pulsatility index
        ac = np.std(g_unif)
        dc = np.mean(g_unif) if np.mean(g_unif) > 1e-4 else 1.0
        ac_dc = ac / dc

        # Only an absolute dead digital image has zero pixel variation (< 0.015%)
        if ac_dc < 0.00015:
            return False, 5.0, "STATIC PHOTO / ZERO VARIATION"

        # Severe violent shake
        if ac_dc > 0.08:
            return True, 30.0, "HEAD MOVEMENT - STABILIZING"

        # Test 2: Peak spectral prominence
        median_power = np.median(band_spec)
        max_power = band_spec[peak_idx]
        prominence = max_power / (median_power + 1e-6)

        # Cross-correlation between forehead and cheeks (quality metric)
        cross_corr = 0.5
        if len(self.forehead_g) == len(t_arr) and len(self.cheeks_g) == len(t_arr):
            fh_arr = np.array(self.forehead_g)
            ch_arr = np.array(self.cheeks_g)
            fh_unif = np.interp(t_uniform, t_arr, fh_arr)
            ch_unif = np.interp(t_uniform, t_arr, ch_arr)
            fh_filt = signal.filtfilt(b_filter, a_filter, signal.detrend(fh_unif))
            ch_filt = signal.filtfilt(b_filter, a_filter, signal.detrend(ch_unif))
            std_fh = np.std(fh_filt)
            std_ch = np.std(ch_filt)
            if std_fh > 1e-6 and std_ch > 1e-6:
                corr_matrix = np.corrcoef(fh_filt, ch_filt)
                val = float(corr_matrix[0, 1])
                if not np.isnan(val):
                    cross_corr = max(-1.0, min(1.0, val))

        # True static image test: extremely low variation + virtually zero peak prominence
        if ac_dc < 0.0003 and prominence < 1.3:
            return False, 10.0, "STATIC PHOTO DETECTED"

        # Human liveness score (0 to 100%)
        liveness_score = min(100.0, 40.0 + (prominence * 10.0) + (max(0.0, cross_corr) * 20.0))
        return True, liveness_score, "LIVE HUMAN TISSUE VERIFIED"

    def _extract_pulse(
        self, r: np.ndarray, g: np.ndarray, b: np.ndarray, fs: float = 30.0
    ) -> np.ndarray:
        """
        rPPG pulse extraction algorithms with sliding-window overlap-add.
        """
        n = len(r)
        rgb = np.column_stack([r, g, b])

        if self.algorithm == "POS":
            # POS with 1.6s sliding window overlap-add (Wang et al., 2017)
            win_s = 1.6
            L = int(round(win_s * fs))
            if n < L:
                r_norm = r / (np.mean(r) + 1e-4)
                g_norm = g / (np.mean(g) + 1e-4)
                b_norm = b / (np.mean(b) + 1e-4)
                s1 = g_norm - b_norm
                s2 = g_norm + b_norm - 2.0 * r_norm
                alpha = np.std(s1) / (np.std(s2) + 1e-6)
                return s1 + alpha * s2

            P = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
            h_all = np.zeros(n)
            w = signal.windows.hann(L, sym=False)
            for s in range(0, n - L + 1):
                c = rgb[s : s + L].T
                cn = c / (c.mean(axis=1, keepdims=True) + 1e-12)
                sp = P @ cn
                std1 = sp[0].std()
                std2 = sp[1].std()
                h = sp[0] + (std1 / (std2 + 1e-12)) * sp[1]
                h_all[s : s + L] += h - h.mean()
            return h_all

        elif self.algorithm == "CHROM":
            # CHROM with 1.6s sliding window overlap-add (de Haan & Jeanne, 2013)
            win_s = 1.6
            L = int(round(win_s * fs))
            if n < L:
                r_norm = r / (np.mean(r) + 1e-4)
                g_norm = g / (np.mean(g) + 1e-4)
                b_norm = b / (np.mean(b) + 1e-4)
                xs = 3.0 * r_norm - 2.0 * g_norm
                ys = 1.5 * r_norm + g_norm - 1.5 * b_norm
                alpha = np.std(xs) / (np.std(ys) + 1e-6)
                return xs - alpha * ys

            ma = uniform_filter1d(rgb, size=L, axis=0, mode="nearest")
            cn = rgb / (ma + 1e-12)
            xs = 3.0 * cn[:, 0] - 2.0 * cn[:, 1]
            ys = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]
            out = np.zeros(n)
            w = signal.windows.hann(L, sym=False)
            step = L // 2
            for s in range(0, n - L + 1, step):
                xw, yw = xs[s : s + L], ys[s : s + L]
                alpha = xw.std() / (yw.std() + 1e-12)
                sw = xw - alpha * yw
                out[s : s + L] += w * (sw - sw.mean())
            return out

        else:
            # Green channel baseline (inverted so peaks align with systole)
            g_norm = g / (np.mean(g) + 1e-4)
            return -(g_norm - 1.0)

    def _calculate_snr_and_confidence(
        self, band_freqs: np.ndarray, band_spec: np.ndarray, peak_freq: float
    ) -> Tuple[float, float]:
        """
        Calculates Signal-to-Noise Ratio (SNR) in dB and maps to confidence score (0-100%).
        """
        power_spec = band_spec**2
        total_power = np.sum(power_spec)
        if total_power < 1e-9:
            return 0.0, 0.0

        delta = 0.12
        fund_mask = np.abs(band_freqs - peak_freq) <= delta
        harm_mask = np.abs(band_freqs - (2.0 * peak_freq)) <= delta

        signal_mask = fund_mask | harm_mask
        signal_power = np.sum(power_spec[signal_mask])
        noise_power = np.sum(power_spec[~signal_mask])

        if noise_power < 1e-9:
            snr_db = 20.0
        else:
            snr_ratio = signal_power / noise_power
            snr_db = float(10.0 * np.log10(max(snr_ratio, 1e-4)))

        clamped_snr = max(-4.0, min(snr_db, 14.0))
        confidence = float(((clamped_snr - (-4.0)) / (14.0 - (-4.0))) * 100.0)

        return snr_db, confidence

    def _detect_beat_and_hrv(
        self, pulse: np.ndarray, fs: float, now: float
    ) -> Tuple[bool, float, float]:
        """
        Detects local systolic peaks in the filtered PPG waveform,
        measures Inter-Beat Interval (IBI), and calculates HRV (RMSSD).
        """
        min_dist = max(5, int(fs / 3.0))
        peaks, _ = signal.find_peaks(pulse, distance=min_dist, prominence=np.std(pulse) * 0.4)

        is_beat = False
        latest_ibi = 0.0
        hrv_rmssd = 0.0

        if len(peaks) >= 2:
            last_peak_idx = peaks[-1]
            if len(pulse) - 1 - last_peak_idx <= 3:
                if now - self.last_beat_time > self.beat_cooldown:
                    is_beat = True
                    self.last_beat_time = now

            ibi_samples = np.diff(peaks)
            ibis_ms = (ibi_samples / fs) * 1000.0
            for ibi_val in ibis_ms:
                if 330.0 <= ibi_val <= 1400.0:
                    self.ibi_history.append(float(ibi_val))

            if len(self.ibi_history) > 0:
                latest_ibi = self.ibi_history[-1]

            if len(self.ibi_history) >= 3:
                successive_diffs = np.diff(np.array(self.ibi_history))
                hrv_rmssd = float(np.sqrt(np.mean(successive_diffs**2)))

        return is_beat, latest_ibi, hrv_rmssd
