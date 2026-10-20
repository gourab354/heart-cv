#!/usr/bin/env python3
"""
capture.py - Record Webcam Video and ESP32 ECG together (One Recording).
Prepared for Gourab - Build Club AOT.

Creates: data/<participant>/<condition>_R<repeat>/
  video.avi        lossless (FFV1) if available, else MJPG
  frame_times.csv  frame,t (t = seconds since start signal)
  ecg.csv          t_us,adc,leadoff,host_t
  meta.json        settings + quick quality numbers

Usage:
  python capture.py --port COM5 --participant P01 --condition C1 --repeat 1 --duration 60
"""

import argparse
import csv
import json
import os
import threading
import time
import cv2
import serial


def parse_args():
    ap = argparse.ArgumentParser(description="Synchronous Webcam + ESP32 ECG Data Recorder")
    ap.add_argument("--port", required=True, help="Serial port of the ESP32 (e.g., COM5)")
    ap.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    ap.add_argument("--cam", type=int, default=0, help="Camera index (default: 0)")
    ap.add_argument("--participant", required=True, help="Participant ID e.g. P01 (no real names!)")
    ap.add_argument("--condition", required=True, help="Condition code e.g. C1, C2, C3, C4")
    ap.add_argument("--repeat", type=int, default=1, help="Repeat number (default: 1)")
    ap.add_argument("--duration", type=float, default=60.0, help="Recording duration in seconds (default: 60)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--no-preview", action="store_true", help="Disable display window")
    return ap.parse_args()


def configure_camera(cap, a):
    """Try to lock exposure and white balance."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    cap.set(cv2.CAP_PROP_FPS, a.fps)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    for v in (1, 0.25):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, v)
        if abs(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) - v) < 1e-6:
            break

    info = {
        "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "fps_reported": cap.get(cv2.CAP_PROP_FPS),
        "auto_exposure": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
        "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
        "auto_wb": cap.get(cv2.CAP_PROP_AUTO_WB),
    }
    print("Camera configuration:", info)
    return info


def open_writer(path_no_ext, fps, size):
    for codec, ext in (("FFV1", ".avi"), ("MJPG", ".avi")):
        w = cv2.VideoWriter(path_no_ext + ext, cv2.VideoWriter_fourcc(*codec), fps, size)
        if w.isOpened():
            print("Video codec selected:", codec)
            return w, codec, path_no_ext + ext
    raise RuntimeError("Could not open VideoWriter (tried FFV1 and MJPG)")


def main():
    a = parse_args()
    out = os.path.join(a.data_dir, a.participant, f"{a.condition}_R{a.repeat}")
    os.makedirs(out, exist_ok=True)

    print(f"[*] Connecting to ESP32 on {a.port} @ {a.baud} baud...")
    ser = serial.Serial(a.port, a.baud, timeout=0.2)
    time.sleep(2.0)  # ESP32 auto-resets when port opens
    ser.reset_input_buffer()

    cap = cv2.VideoCapture(a.cam)
    if not cap.isOpened():
        raise SystemExit(f"Camera index {a.cam} not found - try --cam 1")

    cam_info = configure_camera(cap, a)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read initial frame from camera.")

    h, w = frame.shape[:2]
    writer, codec, vpath = open_writer(os.path.join(out, "video"), a.fps, (w, h))

    # Warm-up camera sensor
    for _ in range(15):
        cap.read()

    ecg_rows = []
    t0_box = [None]
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            line = ser.readline().decode(errors="ignore").strip()
            if not line or line.startswith("#") or t0_box[0] is None:
                continue
            p = line.split(",")
            if len(p) != 3:
                continue
            try:
                ecg_rows.append(
                    (
                        int(p[0]),
                        int(p[1]),
                        int(p[2]),
                        time.perf_counter() - t0_box[0],
                    )
                )
            except ValueError:
                pass

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    input(
        f"\n[{a.participant} {a.condition} R{a.repeat}] Participant ready?\n"
        f"Press ENTER to record for {a.duration:.0f} s ... "
    )

    frame_rows = []
    n = 0
    t0_box[0] = time.perf_counter()
    t0 = t0_box[0]
    ser.write(b"S")  # ESP32 starts its microsecond clock now

    print(f"[*] Recording started: {a.duration:.0f}s timer running...")

    try:
        while True:
            ok, frame = cap.read()
            t = time.perf_counter() - t0
            if not ok:
                print("Warning: frame grab failed")
                if t > a.duration:
                    break
                continue

            writer.write(frame)
            frame_rows.append((n, t))
            n += 1

            if not a.no_preview:
                disp = frame.copy()
                rem = max(0.0, a.duration - t)
                cv2.putText(
                    disp,
                    f"REC: {rem:.1f}s | Frames: {n}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Recording (q = abort)", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Aborted by user.")
                    break

            if t >= a.duration:
                break

    finally:
        ser.write(b"X")
        time.sleep(0.3)
        stop.set()
        th.join(1.0)
        writer.release()
        cap.release()
        ser.close()
        cv2.destroyAllWindows()

    with open(os.path.join(out, "frame_times.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "t"])
        wr.writerows(frame_rows)

    with open(os.path.join(out, "ecg.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t_us", "adc", "leadoff", "host_t"])
        wr.writerows(ecg_rows)

    dur = frame_rows[-1][1] if frame_rows else 0.0
    fps_real = len(frame_rows) / dur if dur > 0 else 0.0
    lo_frac = (sum(r[2] for r in ecg_rows) / len(ecg_rows)) if ecg_rows else 1.0

    meta = {
        "participant": a.participant,
        "condition": a.condition,
        "repeat": a.repeat,
        "frames": len(frame_rows),
        "fps_measured": round(fps_real, 2),
        "ecg_samples": len(ecg_rows),
        "ecg_expected": int(250 * dur),
        "leadoff_fraction": round(lo_frac, 4),
        "codec": codec,
        "camera": cam_info,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[+] Saved to {out}")
    print(f"    Frames: {len(frame_rows)} | Actual FPS: {fps_real:.1f}")
    print(f"    ECG samples: {len(ecg_rows)} (expected ~{int(250 * dur)})")
    print(f"    Electrode Lead-off: {lo_frac * 100:.1f}%")

    if fps_real < 27:
        print("  [!] WARNING: fps well below 30 - close other programs or lower resolution")
    if len(ecg_rows) < 0.95 * 250 * dur:
        print("  [!] WARNING: ECG samples missing - check USB cable / baud rate")
    if lo_frac > 0.05:
        print("  [!] WARNING: Electrode off for >5% of the time - REDO this recording")


if __name__ == "__main__":
    main()
