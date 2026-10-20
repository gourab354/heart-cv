#!/usr/bin/env python3
"""
extract_rppg.py - Face ROI -> Mean RGB trace per frame (saved as rgb.csv).
Prepared for Gourab - Build Club AOT.

Usage:
  python extract_rppg.py data/P01/C1_R1 [--preview]
"""

import csv
import os
import sys
import cv2
import numpy as np

# MediaPipe FaceMesh landmark indices (outline polygons)
FOREHEAD = [67, 109, 10, 338, 297, 299, 9, 69]
L_CHEEK = [50, 101, 118, 117, 116, 123, 187, 205]
R_CHEEK = [280, 330, 347, 346, 345, 352, 411, 425]


def make_mesh():
    try:
        import mediapipe as mp

        return mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception:
        return None


def roi_mask_mesh(mesh, frame):
    h, w = frame.shape[:2]
    res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    pts = np.array([(p.x * w, p.y * h) for p in lm])
    mask = np.zeros((h, w), np.uint8)
    for idx in (FOREHEAD, L_CHEEK, R_CHEEK):
        cv2.fillPoly(mask, [pts[idx].astype(np.int32)], 255)
    return mask


class HaarROI:
    def __init__(self):
        self.det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.box = None  # smoothed (x, y, w, h)

    def mask(self, frame):
        h, w = frame.shape[:2]
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f = self.det.detectMultiScale(g, 1.2, 5, minSize=(80, 80))
        if len(f):
            b = np.array(max(f, key=lambda r: r[2] * r[3]), float)
            self.box = b if self.box is None else 0.9 * self.box + 0.1 * b
        if self.box is None:
            return None
        x, y, bw, bh = self.box
        m = np.zeros((h, w), np.uint8)

        def rect(fx0, fy0, fx1, fy1):
            cv2.rectangle(
                m,
                (int(x + fx0 * bw), int(y + fy0 * bh)),
                (int(x + fx1 * bw), int(y + fy1 * bh)),
                255,
                -1,
            )

        rect(0.25, 0.08, 0.75, 0.22)  # forehead
        rect(0.15, 0.50, 0.38, 0.72)  # left cheek
        rect(0.62, 0.50, 0.85, 0.72)  # right cheek
        return m


def main(folder, preview=False):
    vpath = os.path.join(folder, "video.avi")
    if not os.path.exists(vpath):
        raise SystemExit(f"Video file not found: {vpath}")

    ft_path = os.path.join(folder, "frame_times.csv")
    if not os.path.exists(ft_path):
        raise SystemExit(f"Frame times file not found: {ft_path}")

    ft = np.loadtxt(ft_path, delimiter=",", skiprows=1)
    cap = cv2.VideoCapture(vpath)

    mesh = make_mesh()
    if mesh:
        haar = None
        print("ROI extraction method: MediaPipe FaceMesh")
    else:
        if not hasattr(cv2, "CascadeClassifier"):
            raise SystemExit(
                "No face detector available. Install opencv-python and mediapipe."
            )
        haar = HaarROI()
        print("ROI extraction method: Haar Cascade fallback")

    rows, miss = [], 0
    last = None
    i = 0

    while True:
        ok, frame = cap.read()
        if not ok or i >= len(ft):
            break

        m = roi_mask_mesh(mesh, frame) if mesh else haar.mask(frame)
        if m is None or m.sum() == 0:
            rgb = last
            miss += 1
        else:
            b, g, r = cv2.mean(frame, mask=m)[:3]
            rgb = (r, g, b)
            last = rgb

        if rgb is not None:
            rows.append((ft[i, 1], *rgb))

        if preview and m is not None:
            vis = frame.copy()
            vis[m == 0] = vis[m == 0] // 3
            cv2.imshow("ROI (q quits)", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        i += 1

    cap.release()
    cv2.destroyAllWindows()

    out = os.path.join(folder, "rgb.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "R", "G", "B"])
        w.writerows(rows)

    lost_pct = 100 * miss / max(i, 1)
    print(f"[+] Saved {out}: {len(rows)} frames, face lost in {miss} frames ({lost_pct:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python extract_rppg.py <recording_folder> [--preview]")
    main(sys.argv[1], preview="--preview" in sys.argv)
