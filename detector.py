"""
Face and Region-of-Interest (ROI) Detector for Remote Photoplethysmography (rPPG).
Extracts anatomical regions (forehead, left cheek, right cheek) with MediaPipe FaceMesh
and applies adaptive skin pixel filtering for high signal-to-noise ratio.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np


@dataclass
class DetectionResult:
    detected: bool = False
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    roi_polygons: Dict[str, np.ndarray] = field(default_factory=dict)
    rgb_mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (R, G, B)
    roi_means: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    motion_score: float = 0.0
    is_stable: bool = True
    landmarks: Optional[np.ndarray] = None


class FaceDetector:
    """
    Tracks facial landmarks and extracts precise vascular skin ROIs
    using MediaPipe FaceMesh (468 landmarks).
    """

    # Anatomical landmark indices for rPPG ROIs
    # Forehead: between eyebrows and hairline
    FOREHEAD_LANDMARKS = [67, 109, 10, 338, 297, 334, 9, 105]

    # Left Cheek (anatomical right of the subject, left on camera image)
    LEFT_CHEEK_LANDMARKS = [116, 117, 118, 101, 50, 205, 206, 207, 187, 123]

    # Right Cheek (anatomical left of the subject, right on camera image)
    RIGHT_CHEEK_LANDMARKS = [345, 346, 347, 330, 280, 425, 426, 427, 411, 352]

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        motion_threshold: float = 12.0,
    ):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.motion_threshold = motion_threshold
        self.prev_centroid: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None

    def process(self, frame: np.ndarray) -> DetectionResult:
        """
        Process a BGR video frame and extract facial ROIs and average skin RGB values.
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            self.prev_centroid = None
            self.prev_pts = None
            return DetectionResult(detected=False)

        # Take primary face
        face_landmarks = results.multi_face_landmarks[0]
        raw_pts = np.array(
            [[lm.x * w, lm.y * h] for lm in face_landmarks.landmark],
            dtype=np.float32,
        )

        # Temporal landmark smoothing (eliminates 1-2 pixel boundary jitter)
        if self.prev_pts is not None and len(self.prev_pts) == len(raw_pts):
            pts = 0.80 * self.prev_pts + 0.20 * raw_pts
        else:
            pts = raw_pts
        self.prev_pts = pts

        # Bounding box of full face
        min_x = int(np.clip(np.min(pts[:, 0]), 0, w - 1))
        max_x = int(np.clip(np.max(pts[:, 0]), 0, w - 1))
        min_y = int(np.clip(np.min(pts[:, 1]), 0, h - 1))
        max_y = int(np.clip(np.max(pts[:, 1]), 0, h - 1))
        face_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)

        # Motion detection: track nose bridge / face centroid displacement
        centroid = np.mean(pts, axis=0)
        if self.prev_centroid is not None:
            displacement = np.linalg.norm(centroid - self.prev_centroid)
        else:
            displacement = 0.0
        self.prev_centroid = centroid

        is_stable = displacement < self.motion_threshold

        # Extract ROIs using smoothed coordinates
        roi_polys = {
            "forehead": self._get_poly_coords(pts, self.FOREHEAD_LANDMARKS, w, h),
            "left_cheek": self._get_poly_coords(pts, self.LEFT_CHEEK_LANDMARKS, w, h),
            "right_cheek": self._get_poly_coords(pts, self.RIGHT_CHEEK_LANDMARKS, w, h),
        }

        # Extract skin RGB from each ROI
        roi_means = {}
        all_roi_pixels = []

        for name, poly in roi_polys.items():
            if len(poly) >= 3:
                pixels = self._extract_roi_pixels(frame, poly)
                if len(pixels) > 20:
                    filtered_pixels = self._filter_skin_outliers(pixels)
                    if len(filtered_pixels) > 0:
                        all_roi_pixels.append(filtered_pixels)
                        # OpenCV is BGR -> convert to RGB for rPPG
                        mean_bgr = np.mean(filtered_pixels, axis=0)
                        roi_means[name] = (float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0]))

        if all_roi_pixels:
            combined = np.vstack(all_roi_pixels)
            mean_bgr = np.mean(combined, axis=0)
            rgb_mean = (float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0]))
        else:
            rgb_mean = (0.0, 0.0, 0.0)

        return DetectionResult(
            detected=True,
            face_bbox=face_bbox,
            roi_polygons=roi_polys,
            rgb_mean=rgb_mean,
            roi_means=roi_means,
            motion_score=float(displacement),
            is_stable=is_stable,
            landmarks=pts,
        )

    def _get_poly_coords(
        self, pts: np.ndarray, indices: List[int], w: int, h: int
    ) -> np.ndarray:
        poly = pts[indices].astype(np.int32)
        poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
        return poly

    def _extract_roi_pixels(self, frame: np.ndarray, poly: np.ndarray) -> np.ndarray:
        """
        Fast polygon pixel extraction using minimal bounding sub-rectangle mask.
        """
        x, y, bw, bh = cv2.boundingRect(poly)
        if bw <= 0 or bh <= 0:
            return np.empty((0, 3), dtype=np.uint8)

        sub_frame = frame[y : y + bh, x : x + bw]
        rel_poly = poly - [x, y]

        mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.fillPoly(mask, [rel_poly], 255)

        return sub_frame[mask > 0]

    def _filter_skin_outliers(self, pixels: np.ndarray) -> np.ndarray:
        """
        Discards specular glare (top 5%) and shadow/hair artifacts (bottom 5%)
        using relative luminance clipping.
        """
        if len(pixels) < 20:
            return pixels

        # Fast perceptual luminance: Y = 0.299R + 0.587G + 0.114B
        lum = 0.299 * pixels[:, 2] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 0]
        low_th = np.percentile(lum, 7)
        high_th = np.percentile(lum, 93)

        mask = (lum >= low_th) & (lum <= high_th)
        filtered = pixels[mask]
        return filtered if len(filtered) > 10 else pixels

    def close(self):
        self.face_mesh.close()
