"""
recognition/face_detector.py - Deep Learning Face Detection using YuNet ONNX.

Face Detection locates the presence and coordinates of human faces and key landmarks
within an image. This is explicitly distinct from Face Recognition, which identifies
who the person is.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2
import numpy as np
import logging
import config
from recognition.model_utils import ensure_models_downloaded

logger = logging.getLogger("ExamAuth.FaceDetector")


@dataclass
class DetectedFace:
    """Structured representation of a single detected face."""
    x: int
    y: int
    w: int
    h: int
    confidence: float
    landmarks: List[Tuple[int, int]]  # 5 landmarks: right eye, left eye, nose, right mouth, left mouth
    raw_face: np.ndarray             # 15-element array required by SFace alignCrop


class FaceDetector:
    """
    YuNet Deep Neural Network Face Detector.
    Detects face bounding boxes and 5 facial landmarks (right eye, left eye, nose tip,
    right mouth corner, left mouth corner) at high speed.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        score_threshold: float = config.FACE_DETECTION_SCORE_THRESHOLD,
        nms_threshold: float = config.FACE_DETECTION_NMS_THRESHOLD
    ):
        ensure_models_downloaded()
        self.model_path = model_path or config.YUNET_MODEL_PATH
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.current_input_size = config.FACE_DETECTION_DEFAULT_SIZE

        logger.info(f"Initializing YuNet FaceDetector from: {self.model_path}")
        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=self.current_input_size,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=5000
        )

    def detect(self, image_bgr: np.ndarray) -> List[DetectedFace]:
        """
        Detect faces in a BGR image.
        Returns a list of DetectedFace objects, sorted by area (largest face first).
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        h, w = image_bgr.shape[:2]
        if (w, h) != self.current_input_size:
            self.detector.setInputSize((w, h))
            self.current_input_size = (w, h)

        _, faces = self.detector.detect(image_bgr)
        if faces is None or len(faces) == 0:
            return []

        results: List[DetectedFace] = []
        for face in faces:
            # Face format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
            x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            score = float(face[-1])

            landmarks = [
                (int(face[4]), int(face[5])),    # Right eye
                (int(face[6]), int(face[7])),    # Left eye
                (int(face[8]), int(face[9])),    # Nose tip
                (int(face[10]), int(face[11])),  # Right mouth corner
                (int(face[12]), int(face[13]))   # Left mouth corner
            ]

            results.append(
                DetectedFace(
                    x=max(0, x),
                    y=max(0, y),
                    w=max(1, fw),
                    h=max(1, fh),
                    confidence=score,
                    landmarks=landmarks,
                    raw_face=face
                )
            )

        # Sort by area descending (largest/closest face first)
        results.sort(key=lambda f: f.w * f.h, reverse=True)
        return results

    def draw_faces(
        self,
        image_bgr: np.ndarray,
        faces: List[DetectedFace],
        draw_landmarks: bool = True
    ) -> np.ndarray:
        """Draw bounding boxes and landmarks on a copy of the image for visual feedback."""
        canvas = image_bgr.copy()
        for idx, face in enumerate(faces):
            # Choose color: Emerald green for primary face, Orange/Red if multiple faces
            box_color = (0, 200, 0) if len(faces) == 1 else (0, 165, 255)

            # Draw bounding box
            cv2.rectangle(canvas, (face.x, face.y), (face.x + face.w, face.y + face.h), box_color, 2)

            # Confidence label
            label = f"Face {idx+1}: {face.confidence*100:.1f}%"
            cv2.putText(
                canvas,
                label,
                (face.x, max(20, face.y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2,
                cv2.LINE_AA
            )

            # Landmarks
            if draw_landmarks:
                landmark_colors = [
                    (255, 0, 0),   # Right eye (Blue)
                    (0, 0, 255),   # Left eye (Red)
                    (0, 255, 255), # Nose tip (Yellow)
                    (255, 0, 255), # Right mouth (Magenta)
                    (0, 255, 0)    # Left mouth (Green)
                ]
                for (lx, ly), color in zip(face.landmarks, landmark_colors):
                    cv2.circle(canvas, (lx, ly), 3, color, -1, cv2.LINE_AA)

        return canvas
