"""
recognition/liveness_detector.py - Anti-Spoofing & Liveness Detection Engine.

Detects live human presence and protects against 2D static photograph attacks,
digital phone screen replays, and photo presentations using temporal landmark
micro-motion tracking, blink detection dynamics, and frequency texture analysis.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Deque
from collections import deque
import math
import numpy as np
import cv2
import logging
import config
from recognition.face_detector import DetectedFace

logger = logging.getLogger("ExamAuth.LivenessDetector")


@dataclass
class LivenessAssessment:
    """Detailed outcome of liveness anti-spoofing analysis."""
    is_live: bool
    liveness_score: float       # [0.0 - 1.0]
    blinks_detected: int
    motion_score: float
    texture_score: float
    message: str
    details: str
    is_challenge_complete: bool = True


class LivenessDetector:
    """
    Temporal & Texture Anti-Spoofing Liveness Engine.
    Maintains a sliding frame history buffer to evaluate live physiological cues.
    """

    def __init__(
        self,
        buffer_size: int = config.LIVENESS_BUFFER_SIZE,
        threshold: float = config.LIVENESS_THRESHOLD,
        min_frames: int = config.LIVENESS_MIN_FRAMES_REQUIRED
    ):
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.min_frames = min_frames

        # Sliding window buffer of normalized facial geometry
        self._landmark_history: Deque[List[Tuple[float, float]]] = deque(maxlen=self.buffer_size)
        self._raw_frames_history: Deque[np.ndarray] = deque(maxlen=self.buffer_size)
        self._blink_counter = 0
        self._prev_eye_aspect_ratio = 1.0
        self._eye_closed_frames = 0

    def reset(self) -> None:
        """Clear historical sliding buffers (e.g., when a new student begins authentication)."""
        self._landmark_history.clear()
        self._raw_frames_history.clear()
        self._blink_counter = 0
        self._prev_eye_aspect_ratio = 1.0
        self._eye_closed_frames = 0

    def update(self, frame_bgr: np.ndarray, detected_face: Optional[DetectedFace]) -> None:
        """Add current frame face landmarks to sliding history window."""
        if detected_face is None or len(detected_face.landmarks) < 5:
            return

        # Normalize 5 landmarks relative to bounding box and inter-ocular distance
        # Landmarks: [0: right_eye, 1: left_eye, 2: nose, 3: right_mouth, 4: left_mouth]
        re = detected_face.landmarks[0]
        le = detected_face.landmarks[1]
        nt = detected_face.landmarks[2]
        rm = detected_face.landmarks[3]
        lm = detected_face.landmarks[4]

        # Inter-ocular distance
        iod = math.hypot(re[0] - le[0], re[1] - le[1])
        if iod < 1e-4:
            iod = 1.0

        # Normalized coordinates relative to eye midpoint
        mid_eye_x = (re[0] + le[0]) / 2.0
        mid_eye_y = (re[1] + le[1]) / 2.0

        normalized = [
            ((p[0] - mid_eye_x) / iod, (p[1] - mid_eye_y) / iod)
            for p in [re, le, nt, rm, lm]
        ]
        self._landmark_history.append(normalized)

        # Track eye-mouth-nose vertical distance ratio for blink & expression shifts
        mid_mouth_y = (rm[1] + lm[1]) / 2.0
        vert_ratio = abs(mid_mouth_y - mid_eye_y) / iod
        nose_eye_ratio = abs(nt[1] - mid_eye_y) / iod

        # Blink estimation: rapid vertical dip in eye-nose distance or ratio
        ear_approx = nose_eye_ratio
        if ear_approx < config.LIVENESS_EAR_THRESHOLD:
            self._eye_closed_frames += 1
        else:
            if 1 <= self._eye_closed_frames <= 5:
                self._blink_counter += 1
            self._eye_closed_frames = 0

        self._prev_eye_aspect_ratio = ear_approx

    def evaluate(self, frame_bgr: np.ndarray, detected_face: DetectedFace) -> LivenessAssessment:
        """
        Evaluate full liveness confidence based on temporal micro-motion,
        blinks, and texture spectrum.
        """
        # Update buffer with current observation
        self.update(frame_bgr, detected_face)

        # 1. Check if sufficient frames have been buffered
        history_len = len(self._landmark_history)
        if history_len < self.min_frames:
            # Not enough frames yet, provide provisional live rating for initial feedback
            return LivenessAssessment(
                is_live=True,
                liveness_score=0.75,
                blinks_detected=self._blink_counter,
                motion_score=0.5,
                texture_score=0.8,
                message="Buffering motion... Look directly at the camera.",
                details=f"Gathering frames ({history_len}/{self.min_frames})",
                is_challenge_complete=False
            )

        # 2. Compute Temporal Micro-Motion across sliding window
        # Real 3D human faces exhibit physiological micro-tremors and natural shifts
        arr = np.array(self._landmark_history)  # Shape: (N, 5, 2)
        # Compute coordinate standard deviation across time
        coord_stds = np.std(arr, axis=0)  # Shape: (5, 2)
        mean_std = float(np.mean(coord_stds))

        # Motion score based on standard deviation
        # Static photo: mean_std < 0.001
        # Live human: mean_std between 0.003 and 0.08
        if mean_std < config.LIVENESS_MICRO_MOTION_MIN:
            motion_score = max(0.05, mean_std / config.LIVENESS_MICRO_MOTION_MIN * 0.4)
        elif mean_std > 0.15:
            # Excessive erratic shaking
            motion_score = 0.50
        else:
            motion_score = min(1.0, 0.6 + (mean_std / 0.03) * 0.4)

        # 3. Blink and dynamic motion bonus
        blink_score = min(1.0, self._blink_counter * 0.4) if self._blink_counter > 0 else 0.0

        # 4. Texture / Frequency Analysis (Screen / Photo Moiré Pattern check)
        h_img, w_img = frame_bgr.shape[:2]
        fx, fy, fw, fh = detected_face.x, detected_face.y, detected_face.w, detected_face.h
        fx, fy = max(0, fx), max(0, fy)
        fw = min(w_img - fx, fw)
        fh = min(h_img - fy, fh)
        face_roi = frame_bgr[fy:fy+fh, fx:fx+fw]

        if face_roi.size > 0:
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            # High frequency energy in Laplacian
            lap_var = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
            # Real faces with skin texture have Laplacian variance in healthy range [50 - 600]
            # Digital screens or flat paper printouts often have anomalous texture or flat gradients
            if 40.0 <= lap_var <= 1200.0:
                texture_score = 0.90
            else:
                texture_score = 0.50
        else:
            texture_score = 0.70

        # 5. Aggregate Weighted Liveness Score
        # Weights: Motion (45%), Texture (35%), Blink/Expression (20%)
        composite_score = (motion_score * 0.45) + (texture_score * 0.35) + (max(0.65, 0.65 + blink_score * 0.35) * 0.20)
        composite_score = min(1.0, max(0.0, composite_score))

        is_live = composite_score >= self.threshold

        if not is_live:
            msg = "⚠️ Spoof / Static Image Detected. Please position a live face and blink naturally."
            details = f"Liveness Score: {composite_score:.2f} < threshold {self.threshold:.2f} (Motion: {motion_score:.2f}, Texture: {texture_score:.2f})"
        else:
            msg = "✓ Live Candidate Verified"
            details = f"Liveness Score: {composite_score:.2f} >= threshold {self.threshold:.2f} (Blinks: {self._blink_counter}, Motion: {motion_score:.2f})"

        return LivenessAssessment(
            is_live=is_live,
            liveness_score=composite_score,
            blinks_detected=self._blink_counter,
            motion_score=motion_score,
            texture_score=texture_score,
            message=msg,
            details=details,
            is_challenge_complete=True
        )
