"""
services/registration_service.py - Student Registration Business Logic.

Coordinates face detection, face encoding extraction, demographic validation,
and database storage during manual webcam registration.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import logging
from database.db import Database
from recognition.face_detector import FaceDetector, DetectedFace
from recognition.face_encoder import FaceEncoder

logger = logging.getLogger("ExamAuth.RegistrationService")


@dataclass
class RegistrationResult:
    """Outcome of a student registration attempt."""
    success: bool
    message: str
    student_id: Optional[int] = None
    detected_face: Optional[DetectedFace] = None


class RegistrationService:
    """Handles end-to-end student enrollment and biometric encoding."""

    def __init__(
        self,
        db: Optional[Database] = None,
        detector: Optional[FaceDetector] = None,
        encoder: Optional[FaceEncoder] = None
    ):
        self.db = db or Database()
        self.detector = detector or FaceDetector()
        self.encoder = encoder or FaceEncoder()

    def register_student(
        self,
        name: str,
        roll_number: str,
        frame_bgr: np.ndarray,
        course: str = "",
        registration_type: str = "WEBCAM"
    ) -> RegistrationResult:
        """
        Validate inputs, detect face, generate 128-d embedding, and save student record.
        """
        # 1. Validate demographic fields
        name = name.strip()
        roll_number = roll_number.strip().upper()
        course = course.strip()

        if not name:
            return RegistrationResult(success=False, message="Student Name is required.")
        if not roll_number:
            return RegistrationResult(success=False, message="Roll Number is required.")

        # 2. Check for duplicate roll number
        if self.db.student_exists(roll_number):
            logger.warning(f"Registration rejected: Roll number '{roll_number}' is already registered.")
            return RegistrationResult(
                success=False,
                message=f"Roll number '{roll_number}' is already registered in the system."
            )

        # 3. Check frame availability
        if frame_bgr is None or frame_bgr.size == 0:
            logger.error("Registration failed: No camera frame provided.")
            return RegistrationResult(
                success=False,
                message="Camera could not be accessed. No frame available for capture."
            )

        # 4. Face Detection
        faces = self.detector.detect(frame_bgr)
        face_count = len(faces)

        if face_count == 0:
            logger.info(f"Registration attempt for '{roll_number}': No face detected.")
            return RegistrationResult(
                success=False,
                message="No face detected. Please position your face in front of the camera."
            )

        if face_count > 1:
            logger.warning(f"Registration attempt for '{roll_number}': Multiple faces detected ({face_count}).")
            return RegistrationResult(
                success=False,
                message="Multiple faces detected. Please ensure only one student is visible."
            )

        primary_face = faces[0]

        # 5. Face Encoding (Feature Extraction)
        try:
            embedding = self.encoder.encode_face(frame_bgr, primary_face)
            encoding_bytes = self.encoder.serialize_embedding(embedding)
        except Exception as e:
            logger.error(f"Failed to generate face encoding: {e}", exc_info=True)
            return RegistrationResult(
                success=False,
                message=f"Biometric encoding failure: {e}"
            )

        # 6. Database Storage
        try:
            student_id = self.db.add_student(
                name=name,
                roll_number=roll_number,
                face_encoding_bytes=encoding_bytes,
                course=course,
                registration_type=registration_type
            )
            logger.info(f"Student Registered Successfully: id={student_id}, roll={roll_number}")
            return RegistrationResult(
                success=True,
                message="Student Registered Successfully",
                student_id=student_id,
                detected_face=primary_face
            )
        except Exception as e:
            logger.error(f"Database error during student registration: {e}", exc_info=True)
            return RegistrationResult(
                success=False,
                message=f"Database error while saving student: {e}"
            )
