"""
Biomedical UI & Visualizer for OpenCV rPPG Heart Rate Detector.
Renders an interactive high-contrast HUD with:
- Dual-pane layout (Webcam feed with biometric tracking on left, Telemetry on right)
- Large glowing BPM readout with animated beating heart
- Live real-time PPG oscilloscope waveform
- Live FFT Power Spectral Density plot
- Signal quality & confidence meter (SNR in dB)
- Liveness & Anti-Spoofing indicator (rejects static photos & phone screens)
- Biometric stats (HRV/RMSSD, IBI, Min/Max/Avg BPM)
- Keyboard shortcut help panel
"""

import math
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from detector import DetectionResult
from rppg import HeartRateMetrics


class UIColors:
    # BGR format for OpenCV
    BG_DARK = (18, 14, 11)  # Deep slate #0B0E12
    PANEL_BG = (28, 22, 17)  # Card dark blue-gray
    PANEL_BORDER = (55, 45, 38)
    TEXT_WHITE = (245, 245, 245)
    TEXT_MUTED = (160, 150, 140)
    TEXT_DARK = (90, 80, 70)

    # Accents
    CYAN_ACCENT = (235, 200, 30)  # Bright cyan
    GREEN_ACCENT = (110, 230, 80)  # Emerald pulse green
    AMBER_ACCENT = (40, 165, 255)  # Warning amber
    RED_ACCENT = (60, 60, 245)  # Alert red / fake indicator
    PURPLE_ACCENT = (235, 110, 180)  # High-tech violet

    GRID_LINE = (38, 30, 24)
    OVERLAY_ROI = (240, 180, 50)  # Translucent cyan-gold for ROIs


class HUDVisualizer:
    """
    Renders high-FPS HUD dashboard directly on OpenCV canvas.
    """

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height

        # Layout geometry
        self.sidebar_width = 440
        self.feed_width = self.width - self.sidebar_width
        self.feed_height = self.height

        # Dynamic animation states
        self.heart_scale = 1.0
        self.heart_target_scale = 1.0
        self.last_anim_time = time.perf_counter()

        # Pre-computed unit heart polygon (parametric curve)
        self.unit_heart = self._generate_unit_heart(points=40)

        # UI toggle states
        self.show_rois = True
        self.show_help = True

    def _generate_unit_heart(self, points: int = 40) -> np.ndarray:
        """Parametric 2D heart contour normalized to unit circle."""
        t = np.linspace(0, 2 * np.pi, points)
        x = 16.0 * (np.sin(t) ** 3)
        y = -(13.0 * np.cos(t) - 5.0 * np.cos(2 * t) - 2.0 * np.cos(3 * t) - np.cos(4 * t))
        pts = np.column_stack([x / 16.0, y / 16.0])
        return pts

    def render(
        self,
        frame: np.ndarray,
        detection: DetectionResult,
        metrics: HeartRateMetrics,
        algorithm_name: str,
        waveform: List[float],
        spectrum_data: Optional[Tuple[np.ndarray, np.ndarray, float]],
        fps: float = 30.0,
        ecg_bpm: float = 0.0,
        ecg_waveform: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Render full application frame with camera feed and biometric telemetry.
        """
        now = time.perf_counter()
        dt = max(0.001, now - self.last_anim_time)
        self.last_anim_time = now

        # Update heart animation decay (only pulses if valid live beat)
        if metrics.is_beat and metrics.is_live:
            self.heart_scale = 1.38
        else:
            self.heart_scale += (1.0 - self.heart_scale) * min(1.0, dt * 10.0)

        # 1. Blank canvas
        canvas = np.full((self.height, self.width, 3), UIColors.BG_DARK, dtype=np.uint8)

        # 2. Camera feed with tracking graphics on left
        self._render_camera_feed(canvas, frame, detection, metrics)

        # 3. Sidebar telemetry on right
        self._render_sidebar(
            canvas,
            metrics,
            algorithm_name,
            waveform,
            spectrum_data,
            fps,
            detection.detected,
            ecg_bpm=ecg_bpm,
            ecg_waveform=ecg_waveform,
        )

        return canvas

    def _render_camera_feed(
        self,
        canvas: np.ndarray,
        frame: np.ndarray,
        detection: DetectionResult,
        metrics: HeartRateMetrics,
    ):
        """Fit webcam feed into the left viewport with tracking graphics."""
        fh, fw, _ = frame.shape
        target_w = self.feed_width
        target_h = self.feed_height

        scale = min(target_w / fw, target_h / fh)
        scaled_w = int(fw * scale)
        scaled_h = int(fh * scale)
        offset_x = (target_w - scaled_w) // 2
        offset_y = (target_h - scaled_h) // 2

        resized = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        feed_view = resized.copy()

        if detection.detected and detection.face_bbox is not None:
            sx = scale
            sy = scale

            # Draw ROIs
            if self.show_rois and detection.roi_polygons:
                roi_overlay = feed_view.copy()
                roi_color = UIColors.OVERLAY_ROI if metrics.is_live else (60, 60, 180)
                for name, poly in detection.roi_polygons.items():
                    if len(poly) >= 3:
                        disp_poly = (poly * [sx, sy]).astype(np.int32)
                        cv2.fillPoly(roi_overlay, [disp_poly], roi_color)
                        cv2.polylines(
                            feed_view,
                            [disp_poly],
                            isClosed=True,
                            color=UIColors.CYAN_ACCENT if metrics.is_live else UIColors.RED_ACCENT,
                            thickness=1,
                            lineType=cv2.LINE_AA,
                        )
                cv2.addWeighted(roi_overlay, 0.25, feed_view, 0.75, 0, feed_view)

            # Draw corner brackets
            bx, by, bw, bh = detection.face_bbox
            dbx = int(bx * sx)
            dby = int(by * sy)
            dbw = int(bw * sx)
            dbh = int(bh * sy)

            if not metrics.is_live and metrics.status == "STATIC_PHOTO":
                bracket_color = UIColors.RED_ACCENT
            elif detection.is_stable:
                bracket_color = UIColors.GREEN_ACCENT
            else:
                bracket_color = UIColors.AMBER_ACCENT

            self._draw_corner_brackets(feed_view, dbx, dby, dbw, dbh, color=bracket_color)

        canvas[offset_y : offset_y + scaled_h, offset_x : offset_x + scaled_w] = feed_view
        self._render_feed_header(canvas, detection, metrics)

    def _render_feed_header(
        self, canvas: np.ndarray, detection: DetectionResult, metrics: HeartRateMetrics
    ):
        """Draw top banner with mode and stability/liveness pill."""
        cv2.rectangle(canvas, (0, 0), (self.feed_width, 48), UIColors.BG_DARK, -1)
        cv2.line(canvas, (0, 48), (self.feed_width, 48), UIColors.PANEL_BORDER, 1)

        cv2.putText(
            canvas,
            "HEART VISION  |  rPPG PULSE DETECTOR",
            (20, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            UIColors.TEXT_WHITE,
            2,
            cv2.LINE_AA,
        )

        # Liveness & Tracking Status Pill
        if not detection.detected:
            pill_text = "SEARCHING FOR FACE"
            pill_color = UIColors.AMBER_ACCENT
            pill_width = 240
        elif metrics.status == "STATIC_PHOTO":
            pill_text = "! FAKE / STATIC PHOTO DETECTED"
            pill_color = UIColors.RED_ACCENT
            pill_width = 310
        elif not detection.is_stable:
            pill_text = "! MOTION DETECTED - HOLD STILL"
            pill_color = UIColors.AMBER_ACCENT
            pill_width = 300
        elif metrics.is_live:
            pill_text = "* VERIFIED LIVE HUMAN PULSE"
            pill_color = UIColors.GREEN_ACCENT
            pill_width = 280
        else:
            pill_text = "* ACQUIRING PULSE BUFFER..."
            pill_color = UIColors.CYAN_ACCENT
            pill_width = 280

        px = self.feed_width - pill_width - 15
        cv2.rectangle(canvas, (px, 10), (self.feed_width - 15, 38), UIColors.PANEL_BG, -1)
        cv2.rectangle(canvas, (px, 10), (self.feed_width - 15, 38), pill_color, 1)
        cv2.putText(
            canvas,
            pill_text,
            (px + 10, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            pill_color,
            1,
            cv2.LINE_AA,
        )

    def _render_sidebar(
        self,
        canvas: np.ndarray,
        metrics: HeartRateMetrics,
        algorithm_name: str,
        waveform: List[float],
        spectrum_data: Optional[Tuple[np.ndarray, np.ndarray, float]],
        fps: float,
        face_detected: bool,
        ecg_bpm: float = 0.0,
        ecg_waveform: Optional[List[float]] = None,
    ):
        """Renders dark glassmorphic telemetry cards on the right."""
        sx = self.feed_width
        sw = self.sidebar_width

        cv2.line(canvas, (sx, 0), (sx, self.height), UIColors.PANEL_BORDER, 2)

        # Top section: Heart Rate primary display (Card 1)
        self._render_bpm_card(canvas, sx + 15, 15, sw - 30, 160, metrics, face_detected, ecg_bpm)

        # Card 2: Liveness & Signal Quality
        self._render_quality_card(canvas, sx + 15, 185, sw - 30, 85, metrics, algorithm_name)

        # Card 3: Real-time PPG (+ optional ECG) Waveform Oscilloscope
        self._render_waveform_card(canvas, sx + 15, 280, sw - 30, 135, waveform, metrics.is_live, ecg_waveform)

        # Card 4: FFT Power Spectrum
        self._render_spectrum_card(canvas, sx + 15, 425, sw - 30, 130, spectrum_data, metrics)

        # Card 5: Biometrics & Telemetry Stats
        self._render_stats_card(canvas, sx + 15, 565, sw - 30, 75, metrics, ecg_bpm)

        # Bottom section: Keyboard Hotkeys & FPS
        self._render_footer(canvas, sx + 15, 650, sw - 30, 58, fps)

    def _render_bpm_card(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        metrics: HeartRateMetrics,
        face_detected: bool,
        ecg_bpm: float = 0.0,
    ):
        """Card 1: Giant BPM Readout with Beating Heart and Ground Truth reference."""
        self._draw_panel(canvas, x, y, w, h)

        title = "HEART RATE (rPPG vs ECG)" if ecg_bpm > 0 else "HEART RATE"
        cv2.putText(
            canvas,
            title,
            (x + 18, y + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            UIColors.TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

        heart_center = (x + w - 55, y + 68)
        heart_radius = int(24 * self.heart_scale)
        heart_color = UIColors.RED_ACCENT if (metrics.is_live and metrics.is_valid) else UIColors.TEXT_DARK
        self._draw_heart(canvas, heart_center[0], heart_center[1], heart_radius, heart_color)

        if not face_detected:
            bpm_str = "--"
            status_desc = "No face in view"
            val_color = UIColors.TEXT_MUTED
        elif metrics.status == "STATIC_PHOTO":
            bpm_str = "--"
            status_desc = f"REJECTED: {metrics.liveness_msg}"
            val_color = UIColors.RED_ACCENT
        elif metrics.status == "BUFFERING":
            bpm_str = f"{int(metrics.buffer_progress)}%"
            status_desc = "Gathering pulse buffer..."
            val_color = UIColors.CYAN_ACCENT
        elif metrics.bpm > 0:
            bpm_str = f"{metrics.bpm:.0f}"
            if ecg_bpm > 0:
                err = abs(metrics.bpm - ecg_bpm)
                status_desc = f"ECG Ground Truth: {ecg_bpm:.0f} BPM | Error: {err:.1f} BPM"
                val_color = UIColors.GREEN_ACCENT if err < 3.0 else UIColors.AMBER_ACCENT
            else:
                if metrics.bpm < 60:
                    status_desc = "Resting / Athletic Bradycardia"
                    val_color = UIColors.CYAN_ACCENT
                elif metrics.bpm <= 100:
                    status_desc = "Normal Resting Pulse (Live)"
                    val_color = UIColors.GREEN_ACCENT
                else:
                    status_desc = "Elevated Pulse Rate (Live)"
                    val_color = UIColors.AMBER_ACCENT
        else:
            bpm_str = "--"
            status_desc = "Calibrating pulse..."
            val_color = UIColors.CYAN_ACCENT

        cv2.putText(
            canvas,
            bpm_str,
            (x + 18, y + 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.2,
            val_color,
            4,
            cv2.LINE_AA,
        )

        if bpm_str != "--" and "%" not in bpm_str:
            unit_x = x + 18 + (len(bpm_str) * 44) + 10
            cv2.putText(
                canvas,
                "BPM",
                (unit_x, y + 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                UIColors.TEXT_MUTED,
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            status_desc,
            (x + 20, y + 138),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            val_color if not metrics.is_live else UIColors.TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

    def _render_quality_card(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        metrics: HeartRateMetrics,
        algorithm_name: str,
    ):
        """Card 2: Liveness status, confidence bar, and SNR."""
        self._draw_panel(canvas, x, y, w, h)

        if metrics.is_live:
            live_label = "[x] VERIFIED LIVE"
            live_color = UIColors.GREEN_ACCENT
        elif metrics.status == "STATIC_PHOTO":
            live_label = "[!] SPOOF / STATIC PHOTO"
            live_color = UIColors.RED_ACCENT
        else:
            live_label = "[?] ANALYZING PULSE"
            live_color = UIColors.CYAN_ACCENT

        cv2.putText(
            canvas,
            live_label,
            (x + 18, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            live_color,
            1,
            cv2.LINE_AA,
        )

        algo_badge = f"ALGO: {algorithm_name}"
        cv2.putText(
            canvas,
            algo_badge,
            (x + w - 110, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            UIColors.CYAN_ACCENT,
            1,
            cv2.LINE_AA,
        )

        bar_x = x + 18
        bar_y = y + 36
        bar_w = w - 36
        bar_h = 10
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 32, 26), -1)

        fill_pct = max(0.0, min(100.0, metrics.confidence)) / 100.0
        fill_w = int(bar_w * fill_pct)
        if not metrics.is_live:
            bar_color = UIColors.RED_ACCENT
        elif fill_pct >= 0.65:
            bar_color = UIColors.GREEN_ACCENT
        elif fill_pct >= 0.40:
            bar_color = UIColors.AMBER_ACCENT
        else:
            bar_color = UIColors.RED_ACCENT

        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

        snr_txt = f"SNR: {metrics.snr_db:+.1f} dB"
        conf_txt = f"Quality: {metrics.confidence:.0f}%"
        cv2.putText(
            canvas,
            conf_txt,
            (x + 18, y + 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            UIColors.TEXT_WHITE,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            snr_txt,
            (x + w - 110, y + 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            UIColors.TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

    def _render_waveform_card(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        waveform: List[float],
        is_live: bool,
        ecg_waveform: Optional[List[float]] = None,
    ):
        """Card 3: Real-Time PPG (+ optional ECG) Oscilloscope."""
        self._draw_panel(canvas, x, y, w, h)

        title = "PULSE WAVEFORM (Green: rPPG | Gold: ECG)" if (ecg_waveform and len(ecg_waveform) > 5) else "PULSE WAVEFORM (PPG OSCILLOSCOPE)"
        cv2.putText(
            canvas,
            title,
            (x + 18, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            UIColors.TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

        gx = x + 18
        gy = y + 36
        gw = w - 36
        gh = h - 48

        cv2.rectangle(canvas, (gx, gy), (gx + gw, gy + gh), (15, 12, 10), -1)
        cv2.rectangle(canvas, (gx, gy), (gx + gw, gy + gh), UIColors.PANEL_BORDER, 1)

        mid_y = gy + (gh // 2)
        cv2.line(canvas, (gx, mid_y), (gx + gw, mid_y), UIColors.GRID_LINE, 1)

        # Draw ECG trace in background if available
        if ecg_waveform and len(ecg_waveform) >= 10:
            ecg_arr = np.array(ecg_waveform)
            ecg_pts = []
            n_ecg = len(ecg_arr)
            for i, val in enumerate(ecg_arr):
                px = gx + int((i / max(1, n_ecg - 1)) * gw)
                norm_y = max(-1.0, min(1.0, val))
                py = mid_y - int(norm_y * (gh * 0.40))
                ecg_pts.append((px, py))
            ecg_pts_np = np.array(ecg_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [ecg_pts_np], isClosed=False, color=(50, 180, 240), thickness=1, lineType=cv2.LINE_AA)

        # Draw rPPG trace
        if len(waveform) >= 5 and is_live:
            wave_arr = np.array(waveform)
            std = np.std(wave_arr)
            scale_factor = 2.8 * std if std > 1e-4 else 1.0

            pts = []
            n_pts = len(wave_arr)
            for i, val in enumerate(wave_arr):
                px = gx + int((i / max(1, n_pts - 1)) * gw)
                norm_y = max(-1.0, min(1.0, val / scale_factor))
                py = mid_y - int(norm_y * (gh * 0.42))
                pts.append((px, py))

            pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts_np], isClosed=False, color=UIColors.GREEN_ACCENT, thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(canvas, pts[-1], 4, UIColors.CYAN_ACCENT, -1, lineType=cv2.LINE_AA)
        else:
            msg = "Acquiring blood volume pulse..." if is_live else "No living pulsatile waveform"
            cv2.putText(
                canvas,
                msg,
                (gx + 30, mid_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                UIColors.TEXT_MUTED if is_live else UIColors.RED_ACCENT,
                1,
                cv2.LINE_AA,
            )

    def _render_spectrum_card(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        spectrum_data: Optional[Tuple[np.ndarray, np.ndarray, float]],
        metrics: HeartRateMetrics,
    ):
        """Card 4: FFT Power Spectral Density Graph with Peak Marker."""
        self._draw_panel(canvas, x, y, w, h)

        cv2.putText(
            canvas,
            "SPECTRAL DENSITY (FFT)",
            (x + 18, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            UIColors.TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

        gx = x + 18
        gy = y + 34
        gw = w - 36
        gh = h - 54

        cv2.rectangle(canvas, (gx, gy), (gx + gw, gy + gh), (15, 12, 10), -1)
        cv2.rectangle(canvas, (gx, gy), (gx + gw, gy + gh), UIColors.PANEL_BORDER, 1)

        if spectrum_data is not None and metrics.is_live:
            freqs_bpm, norm_spec, peak_bpm = spectrum_data
            min_bpm = 45.0
            max_bpm = 168.0

            n_bins = len(freqs_bpm)
            step = max(1, n_bins // gw)

            for i in range(0, n_bins, step):
                f_bpm = freqs_bpm[i]
                mag = norm_spec[i]
                bx = gx + int(((f_bpm - min_bpm) / (max_bpm - min_bpm)) * gw)
                bar_height = int(mag * (gh - 4))
                by = gy + gh - bar_height

                if abs(f_bpm - peak_bpm) < 2.5:
                    col = UIColors.CYAN_ACCENT
                else:
                    col = (70, 55, 45)

                cv2.line(canvas, (bx, gy + gh - 1), (bx, by), col, 1)

            peak_x = gx + int(((peak_bpm - min_bpm) / (max_bpm - min_bpm)) * gw)
            peak_x = max(gx, min(gx + gw, peak_x))
            cv2.line(canvas, (peak_x, gy), (peak_x, gy + gh), UIColors.CYAN_ACCENT, 1, lineType=cv2.LINE_AA)

            peak_lbl = f"Peak: {peak_bpm:.1f} BPM"
            cv2.putText(
                canvas,
                peak_lbl,
                (x + w - 120, y + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                UIColors.CYAN_ACCENT,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(canvas, "45", (gx, gy + gh + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, UIColors.TEXT_MUTED, 1)
        cv2.putText(canvas, "100", (gx + (gw // 2) - 12, gy + gh + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, UIColors.TEXT_MUTED, 1)
        cv2.putText(canvas, "168 BPM", (gx + gw - 46, gy + gh + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, UIColors.TEXT_MUTED, 1)

    def _render_stats_card(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        metrics: HeartRateMetrics,
        ecg_bpm: float = 0.0,
    ):
        """Card 5: Biometric Telemetry (HRV, IBI, Min/Max/Avg or ECG comparison)."""
        self._draw_panel(canvas, x, y, w, h)

        col_w = (w - 20) // 3

        if ecg_bpm > 0:
            c1_x = x + 15
            cv2.putText(canvas, "ECG GROUND TRUTH", (c1_x, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.36, UIColors.AMBER_ACCENT, 1)
            cv2.putText(canvas, f"{ecg_bpm:.1f} BPM", (c1_x, y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.60, UIColors.TEXT_WHITE, 2, cv2.LINE_AA)

            c2_x = c1_x + col_w
            cv2.putText(canvas, "rPPG ESTIMATE", (c2_x, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.36, UIColors.GREEN_ACCENT, 1)
            rppg_str = f"{metrics.bpm:.1f} BPM" if metrics.is_live else "--"
            cv2.putText(canvas, rppg_str, (c2_x, y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.60, UIColors.GREEN_ACCENT, 2, cv2.LINE_AA)

            c3_x = c2_x + col_w
            cv2.putText(canvas, "DELTA ERROR", (c3_x, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.36, UIColors.CYAN_ACCENT, 1)
            err_str = f"{abs(metrics.bpm - ecg_bpm):.1f} BPM" if metrics.is_live else "--"
            cv2.putText(canvas, err_str, (c3_x, y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.60, UIColors.CYAN_ACCENT, 2, cv2.LINE_AA)
        else:
            c1_x = x + 15
            cv2.putText(canvas, "HRV (RMSSD)", (c1_x, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, UIColors.TEXT_MUTED, 1)
            hrv_str = f"{metrics.hrv_rmssd:.0f} ms" if (metrics.hrv_rmssd > 0 and metrics.is_live) else "--"
            cv2.putText(canvas, hrv_str, (c1_x, y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, UIColors.TEXT_WHITE, 2, cv2.LINE_AA)

            c2_x = c1_x + col_w
            cv2.putText(canvas, "PULSE IBI", (c2_x, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, UIColors.TEXT_MUTED, 1)
            ibi_str = f"{metrics.ibi_ms:.0f} ms" if (metrics.ibi_ms > 0 and metrics.is_live) else "--"
            cv2.putText(canvas, ibi_str, (c2_x, y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, UIColors.TEXT_WHITE, 2, cv2.LINE_AA)

            c3_x = c2_x + col_w
            cv2.putText(canvas, "MIN / AVG / MAX", (c3_x, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, UIColors.TEXT_MUTED, 1)
            if metrics.avg_bpm > 0 and metrics.is_live:
                range_str = f"{metrics.min_bpm:.0f} / {metrics.avg_bpm:.0f} / {metrics.max_bpm:.0f}"
            else:
                range_str = "-- / -- / --"
            cv2.putText(canvas, range_str, (c3_x, y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, UIColors.CYAN_ACCENT, 1, cv2.LINE_AA)

    def _render_footer(
        self, canvas: np.ndarray, x: int, y: int, w: int, h: int, fps: float
    ):
        """Footer: Keyboard Hotkeys and Frame Rate."""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), UIColors.PANEL_BG, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), UIColors.PANEL_BORDER, 1)

        line1 = "[R] Reset   [C] Switch Algo   [M] Toggle ROIs"
        line2 = "[S] Snapshot   [Q/ESC] Quit"
        cv2.putText(canvas, line1, (x + 12, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, UIColors.TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, line2, (x + 12, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, UIColors.TEXT_MUTED, 1, cv2.LINE_AA)

        fps_str = f"{fps:.0f} FPS"
        cv2.putText(canvas, fps_str, (x + w - 70, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.50, UIColors.GREEN_ACCENT, 1, cv2.LINE_AA)

    def _draw_panel(self, canvas: np.ndarray, x: int, y: int, w: int, h: int):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), UIColors.PANEL_BG, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), UIColors.PANEL_BORDER, 1)

    def _draw_heart(self, canvas: np.ndarray, cx: int, cy: int, radius: int, color: Tuple[int, int, int]):
        pts = (self.unit_heart * radius + [cx, cy]).astype(np.int32)
        cv2.fillPoly(canvas, [pts], color, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [pts], isClosed=True, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    def _draw_corner_brackets(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        color: Tuple[int, int, int],
        length: int = 20,
        thickness: int = 2,
    ):
        cl = min(length, min(w, h) // 3)
        cv2.line(frame, (x, y), (x + cl, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + cl), color, thickness)
        cv2.line(frame, (x + w, y), (x + w - cl, y), color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + cl), color, thickness)
        cv2.line(frame, (x, y + h), (x + cl, y + h), color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - cl), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w - cl, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - cl), color, thickness)
