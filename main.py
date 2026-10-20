"""
Heart Vision: Real-Time Remote Photoplethysmography (rPPG) Heart Rate Detector.
Detects heart rate from webcam video using OpenCV, MediaPipe FaceMesh, and the
state-of-the-art Plane-Orthogonal-to-Skin (POS) rPPG algorithm.

Controls:
  [Q] or [ESC] : Quit application
  [R]           : Reset buffers & recalibrate
  [C]           : Switch algorithm (POS -> CHROM -> GREEN)
  [M]           : Toggle vascular ROI mask visualization
  [S]           : Save timestamped snapshot with telemetry
  [F]           : Toggle fullscreen window
"""

import argparse
import datetime
import os
import sys
import time
import cv2
import numpy as np

from detector import FaceDetector
from rppg import RPPGProcessor
from visualizer import HUDVisualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Heart Vision: Accurate real-time webcam heart rate detector via rPPG."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional path to video file instead of live camera",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera capture width (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera capture height (default: 480)",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="POS",
        choices=["POS", "CHROM", "GREEN"],
        help="rPPG extraction algorithm (default: POS)",
    )
    parser.add_argument(
        "--ecg-port",
        type=str,
        default=None,
        help="Optional serial port for live AD8232 ECG streaming (e.g., COM5)",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Launch window in fullscreen mode",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("  HEART VISION - Real-Time rPPG Heart Rate Detector")
    print("  Powered by OpenCV, MediaPipe FaceMesh & POS Algorithm")
    print("=" * 65)

    # 1. Initialize Video Capture
    source = args.video if args.video else args.camera
    print(f"[*] Opening video source: {source}...")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[!] Error: Could not open video source {source}.")
        print("    If using camera, ensure no other application is using it.")
        sys.exit(1)

    # Configure camera properties
    if args.video is None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[+] Capture initialized: {actual_w}x{actual_h} @ ~{actual_fps:.1f} FPS")

    # 2. Initialize Core Subsystems
    print("[*] Loading MediaPipe FaceMesh model...")
    detector = FaceDetector(
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
        motion_threshold=14.0,
    )

    print(f"[*] Initializing rPPG processor with [{args.algo}] algorithm...")
    processor = RPPGProcessor(
        buffer_seconds=9.0,
        min_seconds_for_bpm=3.5,
        target_fs=30.0,
        algorithm=args.algo,
    )

    visualizer = HUDVisualizer(width=1280, height=720)

    # Optional Live AD8232 ECG Receiver
    ecg_receiver = None
    if args.ecg_port:
        print(f"[*] Initializing live AD8232 ECG streaming on {args.ecg_port}...")
        try:
            from ecg_receiver import ECGReceiver

            ecg_receiver = ECGReceiver(port=args.ecg_port)
            ecg_receiver.start()
        except Exception as e:
            print(f"[!] Could not start ECG receiver: {e}")

    # Create Snapshot Directory
    snapshot_dir = os.path.join(os.getcwd(), "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    # 3. Setup Window
    window_name = "Heart Vision - Contactless Pulse Detector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    is_fullscreen = False

    if args.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        is_fullscreen = True

    print("\n[+] System ready!")
    print("-----------------------------------------------------------------")
    print("  Controls:")
    print("    [Q] / [ESC] : Quit")
    print("    [R]         : Reset buffer / Recalibrate")
    print("    [C]         : Cycle Algorithm (POS -> CHROM -> GREEN)")
    print("    [M]         : Toggle Vascular ROIs")
    print("    [S]         : Save Telemetry Snapshot")
    print("    [F]         : Toggle Fullscreen")
    print("-----------------------------------------------------------------\n")

    # FPS Calculation tracking
    fps_history = []
    prev_frame_time = time.perf_counter()

    try:
        while True:
            t_frame_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                if args.video is not None:
                    # Loop video file for continuous demonstration
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[!] Video capture frame dropped or disconnected.")
                    break

            # Mirror webcam feed horizontally for intuitive natural interaction
            if args.video is None:
                frame = cv2.flip(frame, 1)

            # 1. Detect Face & Extract Skin ROIs
            detection = detector.process(frame)

            # 2. Update rPPG Signal Engine (with multi-ROI liveness verification)
            metrics = processor.update(
                rgb_mean=detection.rgb_mean,
                timestamp=t_frame_start,
                is_stable=detection.is_stable,
                roi_means=detection.roi_means,
            )

            # Read live ECG if connected
            ecg_bpm = 0.0
            ecg_waveform = None
            if ecg_receiver and ecg_receiver.is_connected:
                ecg_bpm, _ = ecg_receiver.get_bpm()
                ecg_waveform = ecg_receiver.get_recent_waveform(200)

            # 3. Calculate Smooth FPS
            frame_dt = max(1e-4, t_frame_start - prev_frame_time)
            prev_frame_time = t_frame_start
            instant_fps = 1.0 / frame_dt
            fps_history.append(instant_fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            rolling_fps = sum(fps_history) / len(fps_history)

            # 4. Render Rich HUD Canvas (with optional ECG ground truth)
            canvas = visualizer.render(
                frame=frame,
                detection=detection,
                metrics=metrics,
                algorithm_name=processor.algorithm,
                waveform=list(processor.waveform_history),
                spectrum_data=processor.spectrum_data,
                fps=rolling_fps,
                ecg_bpm=ecg_bpm,
                ecg_waveform=ecg_waveform,
            )

            # 5. Display Canvas
            cv2.imshow(window_name, canvas)

            # 6. Keyboard Hotkey Event Handler
            key = cv2.waitKey(1) & 0xFF

            if key in [ord("q"), ord("Q"), 27]:  # 27 = ESC
                print("[*] Exit signal received. Closing...")
                break

            elif key in [ord("r"), ord("R")]:
                print("[*] Recalibrating: Cleared signal buffer.")
                processor.reset()

            elif key in [ord("c"), ord("C")]:
                new_algo = processor.next_algorithm()
                print(f"[*] Switched rPPG Algorithm to: [{new_algo}]")

            elif key in [ord("m"), ord("M")]:
                visualizer.show_rois = not visualizer.show_rois
                state_str = "ON" if visualizer.show_rois else "OFF"
                print(f"[*] Vascular ROI Overlay: {state_str}")

            elif key in [ord("s"), ord("S")]:
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"snapshot_{timestamp_str}_{int(metrics.bpm)}bpm.png"
                filepath = os.path.join(snapshot_dir, filename)
                cv2.imwrite(filepath, canvas)
                print(f"[+] Snapshot saved: {filepath}")

            elif key in [ord("f"), ord("F")]:
                is_fullscreen = not is_fullscreen
                prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user.")

    finally:
        print("[*] Releasing resources and shutting down...")
        if ecg_receiver:
            try:
                ecg_receiver.stop()
            except Exception:
                pass
        try:
            cap.release()
        except Exception:
            pass
        try:
            detector.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[+] Done. Thank you for using Heart Vision!")


if __name__ == "__main__":
    main()
