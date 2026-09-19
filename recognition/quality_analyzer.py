"""
recognition/quality_analyzer.py - Real-time Face Image Quality Assessment.

Evaluates facial illumination (mean brightness), focus sharpness (Laplacian variance),
and face scale/coverage to guide students for optimal biometric registration and verification.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np
import config


@dataclass
class QualityAssessment:
    """Outcome of real-time face image quality analysis."""
    is_acceptable: bool
    status_code: str       # "GOOD", "TOO_DARK", "TOO_BRIGHT", "BLURRY", "TOO_FAR", "TOO_CLOSE"
    message: str           # User-friendly guidance text
    brightness: float      # Mean pixel intensity
    sharpness: float       # Laplacian variance
    face_area_ratio: float # Fraction of frame covered by face
    badge_color: str       # Hex color for UI pill


class QualityAnalyzer:
    """Evaluates facial image quality for registration and authentication."""

    def __init__(
        self,
        min_brightness: float = config.QUALITY_MIN_BRIGHTNESS,
        max_brightness: float = config.QUALITY_MAX_BRIGHTNESS,
        min_sharpness: float = config.QUALITY_MIN_SHARPNESS,
        min_area_ratio: float = config.QUALITY_MIN_FACE_AREA_RATIO,
        max_area_ratio: float = config.QUALITY_MAX_FACE_AREA_RATIO
    ):
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_sharpness = min_sharpness
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def evaluate_frame(
        self,
        frame_bgr: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> QualityAssessment:
        """
        Evaluate brightness, sharpness, and face distance for a frame.
        If face_bbox (x, y, w, h) is provided, analyzes the localized face ROI.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return QualityAssessment(
                is_acceptable=False,
                status_code="NO_FRAME",
                message="No camera frame available.",
                brightness=0.0,
                sharpness=0.0,
                face_area_ratio=0.0,
                badge_color="#DC2626"
            )

        h_img, w_img = frame_bgr.shape[:2]
        frame_area = float(w_img * h_img)

        # Determine ROI (Face crop or whole image)
        if face_bbox:
            fx, fy, fw, fh = face_bbox
            fx = max(0, min(w_img - 1, fx))
            fy = max(0, min(h_img - 1, fy))
            fw = max(1, min(w_img - fx, fw))
            fh = max(1, min(h_img - fy, fh))
            roi = frame_bgr[fy:fy+fh, fx:fx+fw]
            face_area_ratio = float(fw * fh) / frame_area if frame_area > 0 else 0.0
        else:
            roi = frame_bgr
            face_area_ratio = 0.0

        if roi.size == 0:
            roi = frame_bgr

        # Convert to grayscale for metric calculations
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 1. Brightness / Illumination: Mean pixel intensity [0 - 255]
        brightness = float(np.mean(gray))

        # 2. Focus Sharpness: Variance of the Laplacian operator
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Quality Checks Priority
        if brightness < self.min_brightness:
            return QualityAssessment(
                is_acceptable=False,
                status_code="TOO_DARK",
                message="⚠️ Low Lighting: Face is in shadow. Move to a better-lit area.",
                brightness=brightness,
                sharpness=sharpness,
                face_area_ratio=face_area_ratio,
                badge_color="#D97706"
            )

        if brightness > self.max_brightness:
            return QualityAssessment(
                is_acceptable=False,
                status_code="TOO_BRIGHT",
                message="⚠️ Overexposed: Strong glare on face. Adjust lighting.",
                brightness=brightness,
                sharpness=sharpness,
                face_area_ratio=face_area_ratio,
                badge_color="#D97706"
            )

        if face_bbox and face_area_ratio < self.min_area_ratio:
            return QualityAssessment(
                is_acceptable=False,
                status_code="TOO_FAR",
                message="⚠️ Too Far: Please step closer to the camera.",
                brightness=brightness,
                sharpness=sharpness,
                face_area_ratio=face_area_ratio,
                badge_color="#D97706"
            )

        if face_bbox and face_area_ratio > self.max_area_ratio:
            return QualityAssessment(
                is_acceptable=False,
                status_code="TOO_CLOSE",
                message="⚠️ Too Close: Please step back slightly.",
                brightness=brightness,
                sharpness=sharpness,
                face_area_ratio=face_area_ratio,
                badge_color="#D97706"
            )

        if sharpness < self.min_sharpness:
            return QualityAssessment(
                is_acceptable=False,
                status_code="BLURRY",
                message="⚠️ Motion Blur: Please hold still and look at the camera.",
                brightness=brightness,
                sharpness=sharpness,
                face_area_ratio=face_area_ratio,
                badge_color="#D97706"
            )

        # All quality checks passed!
        return QualityAssessment(
            is_acceptable=True,
            status_code="GOOD",
            message="✓ Optimal Lighting & Focus",
            brightness=brightness,
            sharpness=sharpness,
            face_area_ratio=face_area_ratio,
            badge_color="#059669"
        )
