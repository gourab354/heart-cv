"""
Face and Region-of-Interest (ROI) Detector for Remote Photoplethysmography (rPPG).
Extracts anatomical regions (forehead, left cheek, right cheek) with MediaPipe FaceMesh
and applies adaptive skin pixel filtering for high signal-to-noise ratio.

Motion robustness:
  - Adaptive temporal landmark smoothing (fast catch-up during large motion)
  - Three-tier motion classification: 'none' | 'mild' | 'heavy'
  - Optical-flow ROI tracking fallback when MediaPipe loses the face
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
    motion_level: str = "none"  # 'none' | 'mild' | 'heavy'
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

    # Motion level thresholds (pixels of centroid displacement per frame)
    # mild_threshold: start flagging but keep processing
    # heavy_threshold: severe shake, warn user
    MILD_MOTION_THRESHOLD = 8.0
    HEAVY_MOTION_THRESHOLD = 22.0

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        motion_threshold: float = 22.0,  # kept for API compat, mapped to heavy
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

        # Optical-flow fallback state
        self.prev_gray: Optional[np.ndarray] = None
        self.of_pts: Optional[np.ndarray] = None  # flow tracking points (Nx1x2)
        self.of_bbox: Optional[Tuple[int, int, int, int]] = None  # last known face bbox
        self._of_miss_count: int = 0  # consecutive frames without MediaPipe
        self._OF_MAX_MISS = 10  # stop flow fallback after 10 misses

    def process(self, frame: np.ndarray) -> DetectionResult:
        """
        Process a BGR video frame and extract facial ROIs and average skin RGB values.
        Motion robustness:
          - Adaptive smoothing: fast catch-up for large displacements
          - Three-tier motion classification (none / mild / heavy)
          - Optical-flow ROI tracking fallback when MediaPipe fails
        """
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        # ------------------------------------------------------------------ #
        # OPTICAL FLOW FALLBACK: if MediaPipe lost face, try to track ROI     #
        # ------------------------------------------------------------------ #
        if not results.multi_face_landmarks:
            if (
                self.prev_gray is not None
                and self.of_pts is not None
                and self._of_miss_count < self._OF_MAX_MISS
                and self.of_bbox is not None
            ):
                new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.of_pts, None,
                    winSize=(21, 21), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
                )
                good_mask = status.ravel() == 1
                if good_mask.sum() >= 4:
                    # Estimate translation from tracked points
                    delta = (new_pts[good_mask] - self.of_pts[good_mask]).mean(axis=0)
                    dx, dy = int(delta[0, 0]), int(delta[0, 1])
                    bx, by, bw, bh = self.of_bbox
                    bx = int(np.clip(bx + dx, 0, w - 1))
                    by = int(np.clip(by + dy, 0, h - 1))
                    bw = int(np.clip(bw, 1, w - bx))
                    bh = int(np.clip(bh, 1, h - by))
                    self.of_bbox = (bx, by, bw, bh)
                    self.of_pts = new_pts[good_mask].reshape(-1, 1, 2)

                    # Extract RGB from tracked bbox region
                    roi_frame = frame[by : by + bh, bx : bx + bw]
                    if roi_frame.size > 0:
                        mean_bgr = np.mean(roi_frame.reshape(-1, 3), axis=0)
                        rgb_mean = (float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0]))
                    else:
                        rgb_mean = (0.0, 0.0, 0.0)

                    self._of_miss_count += 1
                    self.prev_gray = gray.copy()
                    return DetectionResult(
                        detected=True,
                        face_bbox=self.of_bbox,
                        rgb_mean=rgb_mean,
                        motion_score=float(np.linalg.norm([dx, dy])),
                        is_stable=False,
                        motion_level="mild",
                    )

            # Truly lost: reset all flow state
            self.prev_centroid = None
            self.prev_pts = None
            self.of_pts = None
            self.of_bbox = None
            self._of_miss_count = 0
            self.prev_gray = gray.copy()
            return DetectionResult(detected=False)

        # MediaPipe found a face — reset flow miss counter
        self._of_miss_count = 0

        # Take primary face
        face_landmarks = results.multi_face_landmarks[0]
        raw_pts = np.array(
            [[lm.x * w, lm.y * h] for lm in face_landmarks.landmark],
            dtype=np.float32,
        )

        # ------------------------------------------------------------------ #
        # ADAPTIVE TEMPORAL LANDMARK SMOOTHING                                #
        # Large displacements → more weight on new data (fast catch-up)       #
        # Small jitter        → more weight on history  (noise rejection)     #
        # ------------------------------------------------------------------ #
        if self.prev_pts is not None and len(self.prev_pts) == len(raw_pts):
            raw_centroid = raw_pts.mean(axis=0)
            if self.prev_centroid is not None:
                pre_disp = float(np.linalg.norm(raw_centroid - self.prev_centroid))
            else:
                pre_disp = 0.0

            # alpha = weight given to NEW observation (0.20 baseline → 1.0 at heavy motion)
            alpha = float(np.clip(0.20 + (pre_disp / self.HEAVY_MOTION_THRESHOLD) * 0.80, 0.20, 1.0))
            pts = (1.0 - alpha) * self.prev_pts + alpha * raw_pts
        else:
            pts = raw_pts
        self.prev_pts = pts

        # Bounding box of full face
        min_x = int(np.clip(np.min(pts[:, 0]), 0, w - 1))
        max_x = int(np.clip(np.max(pts[:, 0]), 0, w - 1))
        min_y = int(np.clip(np.min(pts[:, 1]), 0, h - 1))
        max_y = int(np.clip(np.max(pts[:, 1]), 0, h - 1))
        face_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)

        # ------------------------------------------------------------------ #
        # MOTION CLASSIFICATION (three tiers)                                 #
        # ------------------------------------------------------------------ #
        centroid = np.mean(pts, axis=0)
        if self.prev_centroid is not None:
            displacement = float(np.linalg.norm(centroid - self.prev_centroid))
        else:
            displacement = 0.0
        self.prev_centroid = centroid

        if displacement < self.MILD_MOTION_THRESHOLD:
            motion_level = "none"
            is_stable = True
        elif displacement < self.HEAVY_MOTION_THRESHOLD:
            motion_level = "mild"
            is_stable = True   # still usable for rPPG, just deweighted
        else:
            motion_level = "heavy"
            is_stable = False  # severe shake — warn user

        # Store optical-flow seed points for fallback
        bbox_frame = frame[min_y:max_y, min_x:max_x]
        if bbox_frame.size > 0:
            bbox_gray = gray[min_y:max_y, min_x:max_x]
            corners = cv2.goodFeaturesToTrack(bbox_gray, maxCorners=40, qualityLevel=0.05,
                                              minDistance=5, blockSize=5)
            if corners is not None and len(corners) >= 4:
                # Convert from bbox-relative to full-frame coords
                corners[:, :, 0] += min_x
                corners[:, :, 1] += min_y
                self.of_pts = corners.astype(np.float32)
                self.of_bbox = face_bbox
        self.prev_gray = gray.copy()

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
            motion_level=motion_level,
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
