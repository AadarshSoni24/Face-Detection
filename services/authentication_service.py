"""
services/authentication_service.py - Exam Authentication Decision Pipeline.

Verifies a student's live face against their registered biometric reference.
If verified: marks attendance and grants exam access.
If failed: logs attempt, denies access, and prevents attendance.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import logging
from database.db import Database
from recognition.face_detector import FaceDetector, DetectedFace
from recognition.face_encoder import FaceEncoder
from recognition.face_matcher import FaceMatcher, MatchResult
from services.attendance_service import AttendanceService

logger = logging.getLogger("ExamAuth.AuthenticationService")


@dataclass
class AuthenticationResult:
    """Detailed outcome of an exam authentication attempt."""
    success: bool
    status_title: str          # "IDENTITY VERIFIED" or "AUTHENTICATION FAILED"
    student_name: str
    roll_number: str
    face_match_status: str     # "SUCCESS" or "FAILED"
    attendance_status: str     # "PRESENT", "ALREADY MARKED", or "NOT MARKED"
    exam_access: str           # "GRANTED" or "DENIED"
    message: str
    similarity_score: Optional[float] = None
    threshold: Optional[float] = None
    detected_face: Optional[DetectedFace] = None


class AuthenticationService:
    """Executes the complete verification workflow on exam day."""

    def __init__(
        self,
        db: Optional[Database] = None,
        detector: Optional[FaceDetector] = None,
        encoder: Optional[FaceEncoder] = None,
        matcher: Optional[FaceMatcher] = None,
        attendance_service: Optional[AttendanceService] = None
    ):
        self.db = db or Database()
        self.detector = detector or FaceDetector()
        self.encoder = encoder or FaceEncoder()
        self.matcher = matcher or FaceMatcher()
        self.attendance_service = attendance_service or AttendanceService(self.db)

    def authenticate(self, roll_number: str, frame_bgr: np.ndarray) -> AuthenticationResult:
        """
        Verify student identity for the given roll number using the live camera frame.
        """
        roll_clean = roll_number.strip().upper() if roll_number else ""

        # 1. Validate Roll Number input
        if not roll_clean:
            return AuthenticationResult(
                success=False,
                status_title="AUTHENTICATION FAILED",
                student_name="Unknown",
                roll_number="N/A",
                face_match_status="FAILED",
                attendance_status="NOT MARKED",
                exam_access="DENIED",
                message="Please enter your roll number."
            )

        # 2. Look up student in database
        student = self.db.get_student_by_roll(roll_clean)
        if not student:
            logger.warning(f"Authentication attempt failed: Student '{roll_clean}' not found.")
            self.db.log_auth_attempt(
                student_id=None,
                roll_number=roll_clean,
                result="NOT_FOUND",
                details="Student not found."
            )
            return AuthenticationResult(
                success=False,
                status_title="AUTHENTICATION FAILED",
                student_name="Not Found",
                roll_number=roll_clean,
                face_match_status="FAILED",
                attendance_status="NOT MARKED",
                exam_access="DENIED",
                message="Student not found."
            )

        student_id = student["id"]
        student_name = student["name"]

        # 3. Check frame availability
        if frame_bgr is None or frame_bgr.size == 0:
            logger.error(f"Authentication error for '{roll_clean}': Camera frame unavailable.")
            self.db.log_auth_attempt(
                student_id=student_id,
                roll_number=roll_clean,
                result="CAMERA_ERROR",
                details="Camera could not be accessed."
            )
            return AuthenticationResult(
                success=False,
                status_title="AUTHENTICATION FAILED",
                student_name=student_name,
                roll_number=roll_clean,
                face_match_status="FAILED",
                attendance_status="NOT MARKED",
                exam_access="DENIED",
                message="Camera could not be accessed."
            )

        # 4. Detect face
        faces = self.detector.detect(frame_bgr)
        face_count = len(faces)

        if face_count == 0:
            logger.info(f"Authentication attempt for '{roll_clean}': No face detected.")
            self.db.log_auth_attempt(
                student_id=student_id,
                roll_number=roll_clean,
                result="NO_FACE",
                details="No face detected in live feed."
            )
            return AuthenticationResult(
                success=False,
                status_title="AUTHENTICATION FAILED",
                student_name=student_name,
                roll_number=roll_clean,
                face_match_status="FAILED",
                attendance_status="NOT MARKED",
                exam_access="DENIED",
                message="No face detected. Please position your face in front of the camera."
            )

        if face_count > 1:
            logger.warning(f"Authentication attempt for '{roll_clean}': Multiple faces ({face_count}).")
            self.db.log_auth_attempt(
                student_id=student_id,
                roll_number=roll_clean,
                result="MULTI_FACE",
                details=f"Multiple faces detected ({face_count})."
            )
            return AuthenticationResult(
                success=False,
                status_title="AUTHENTICATION FAILED",
                student_name=student_name,
                roll_number=roll_clean,
                face_match_status="FAILED",
                attendance_status="NOT MARKED",
                exam_access="DENIED",
                message="Multiple faces detected. Please ensure only one student is visible."
            )

        live_face = faces[0]

        # 5. Extract live face embedding
        try:
            live_embedding = self.encoder.encode_face(frame_bgr, live_face)
        except Exception as e:
            logger.error(f"Live encoding error for '{roll_clean}': {e}", exc_info=True)
            self.db.log_auth_attempt(
                student_id=student_id,
                roll_number=roll_clean,
                result="ENCODING_ERROR",
                details=str(e)
            )
            return AuthenticationResult(
                success=False,
                status_title="AUTHENTICATION FAILED",
                student_name=student_name,
                roll_number=roll_clean,
                face_match_status="FAILED",
                attendance_status="NOT MARKED",
                exam_access="DENIED",
                message=f"Feature extraction error: {e}"
            )

        # 6. Fetch registered reference encodings for this student
        stored_blobs = self.db.get_student_encodings(student_id)
        if not stored_blobs:
            logger.error(f"No biometric templates found for student '{roll_clean}' (id={student_id})")
            return AuthenticationResult(
                success=False,
                status_title="AUTHENTICATION FAILED",
                student_name=student_name,
                roll_number=roll_clean,
                face_match_status="FAILED",
                attendance_status="NOT MARKED",
                exam_access="DENIED",
                message="No registered biometric profile found for this student."
            )

        ref_embeddings = [self.encoder.deserialize_embedding(b) for b in stored_blobs]

        # 7. Compare embeddings
        match_result = self.matcher.compare_multi(live_embedding, ref_embeddings)

        # 8. Decision logic
        if match_result.is_match:
            # Face Match SUCCESS!
            # Mark attendance
            att_success, att_msg = self.attendance_service.record_attendance(
                student_id=student_id,
                roll_number=roll_clean
            )
            att_status_text = "PRESENT" if att_success else "PRESENT (ALREADY MARKED)"

            # Log audit record
            self.db.log_auth_attempt(
                student_id=student_id,
                roll_number=roll_clean,
                result="SUCCESS",
                similarity_score=match_result.similarity_score,
                details=match_result.details
            )

            logger.info(
                f"Exam access GRANTED for '{roll_clean}' ({student_name}). "
                f"Similarity: {match_result.similarity_score:.3f}"
            )

            return AuthenticationResult(
                success=True,
                status_title="IDENTITY VERIFIED",
                student_name=student_name,
                roll_number=roll_clean,
                face_match_status="SUCCESS",
                attendance_status=att_status_text,
                exam_access="GRANTED",
                message="Identity verified successfully. Exam access granted.",
                similarity_score=match_result.similarity_score,
                threshold=match_result.threshold,
                detected_face=live_face
            )
        else:
            # Face Match FAILED!
            # Do NOT mark attendance
            self.db.log_auth_attempt(
                student_id=student_id,
                roll_number=roll_clean,
                result="FAILED",
                similarity_score=match_result.similarity_score,
                details=match_result.details
            )

            logger.warning(
                f"Exam access DENIED for '{roll_clean}' ({student_name}). "
                f"Similarity: {match_result.similarity_score:.3f} < threshold {match_result.threshold:.3f}"
            )

            return AuthenticationResult(
                success=False,
                status_title="AUTHENTICATION FAILED",
                student_name=student_name,
                roll_number=roll_clean,
                face_match_status="FAILED",
                attendance_status="NOT MARKED",
                exam_access="DENIED",
                message="Face does not match the registered student.",
                similarity_score=match_result.similarity_score,
                threshold=match_result.threshold,
                detected_face=live_face
            )

    def identify_face(self, frame_bgr: np.ndarray) -> AuthenticationResult:
        """
        1:N Automatic Face Identification:
        Scans live camera face against ALL registered students in the database.
        If a face matches above threshold, automatically returns the student's
        name, roll number, marks attendance, and grants exam access without
        requiring manual roll number entry!
        """
        # 1. Check frame availability
        if frame_bgr is None or frame_bgr.size == 0:
            return AuthenticationResult(
                success=False,
                status_title="AUTHENTICATION FAILED",
                student_name="Unknown",
                roll_number="N/A",
                face_match_status="FAILED",
                attendance_status="NOT MARKED",
                exam_access="DENIED",
                message="Camera could not be accessed."
            )

        # 2. Detect face
        faces = self.detector.detect(frame_bgr)
        face_count = len(faces)

        if face_count == 0:
            return AuthenticationResult(
                success=False,
                status_title="AUTHENTICATION FAILED",
                student_name="Unknown",
                roll_number="N/A",
                face_match_status="FAILED",
                attendance_status="NOT MARKED",
                exam_access="DENIED",
                message="No face detected. Please position your face in front of the camera."
            )

        if face_count > 1:
            return AuthenticationResult(
                success=False,
                status_title="AUTHENTICATION FAILED",
                student_name="Multiple",
                roll_number="N/A",
                face_match_status="FAILED",
                attendance_status="NOT MARKED",
                exam_access="DENIED",
                message="Multiple faces detected. Please ensure only one student is visible."
            )

        live_face = faces[0]

        # 3. Extract live face embedding
        try:
            live_embedding = self.encoder.encode_face(frame_bgr, live_face)
        except Exception as e:
            logger.error(f"Live encoding error during auto-identification: {e}", exc_info=True)
            return AuthenticationResult(
                success=False,
                status_title="AUTHENTICATION FAILED",
                student_name="Error",
                roll_number="N/A",
                face_match_status="FAILED",
                attendance_status="NOT MARKED",
                exam_access="DENIED",
                message=f"Feature extraction error: {e}"
            )

        # 4. Fetch all registered students from SQLite
        students_with_encs = self.db.get_all_students_with_encodings()
        if not students_with_encs:
            return AuthenticationResult(
                success=False,
                status_title="AUTHENTICATION FAILED",
                student_name="No Records",
                roll_number="N/A",
                face_match_status="FAILED",
                attendance_status="NOT MARKED",
                exam_access="DENIED",
                message="No registered students found in database."
            )

        # 5. Search for highest matching student (1:N search)
        best_student = None
        best_score = -1.0

        for s in students_with_encs:
            if not s["encodings"]:
                continue
            ref_embeddings = [self.encoder.deserialize_embedding(b) for b in s["encodings"]]
            match_res = self.matcher.compare_multi(live_embedding, ref_embeddings)
            if match_res.similarity_score > best_score:
                best_score = match_res.similarity_score
                best_student = s

        # 6. Evaluate match against threshold
        if best_student is not None and best_score >= self.matcher.threshold:
            # Student matched!
            matched_id = best_student["id"]
            matched_roll = best_student["roll_number"]
            matched_name = best_student["name"]

            att_success, att_msg = self.attendance_service.record_attendance(
                student_id=matched_id,
                roll_number=matched_roll
            )
            att_status_text = "PRESENT" if att_success else "PRESENT (ALREADY MARKED)"

            self.db.log_auth_attempt(
                student_id=matched_id,
                roll_number=matched_roll,
                result="SUCCESS",
                similarity_score=best_score,
                details=f"Auto-identified student {matched_name} ({matched_roll})"
            )

            logger.info(
                f"Auto-identified student '{matched_roll}' ({matched_name}). "
                f"Similarity: {best_score:.3f} >= {self.matcher.threshold:.3f}"
            )

            return AuthenticationResult(
                success=True,
                status_title="IDENTITY VERIFIED (AUTO-DETECTED)",
                student_name=matched_name,
                roll_number=matched_roll,
                face_match_status="SUCCESS",
                attendance_status=att_status_text,
                exam_access="GRANTED",
                message=f"Face recognized! Automatically identified {matched_name} ({matched_roll}).",
                similarity_score=best_score,
                threshold=self.matcher.threshold,
                detected_face=live_face
            )
        else:
            # Unrecognized face
            self.db.log_auth_attempt(
                student_id=None,
                roll_number="UNKNOWN",
                result="FAILED",
                similarity_score=best_score if best_score > -1 else 0.0,
                details="Unrecognized face during auto-identification"
            )

            logger.warning(
                f"Auto-identification failed. Best similarity: {best_score:.3f} < {self.matcher.threshold:.3f}"
            )

            return AuthenticationResult(
                success=False,
                status_title="AUTHENTICATION FAILED",
                student_name="Unrecognized Face",
                roll_number="Unknown",
                face_match_status="FAILED",
                attendance_status="NOT MARKED",
                exam_access="DENIED",
                message="Unrecognized face. No matching student record found in database.",
                similarity_score=best_score if best_score > -1 else 0.0,
                threshold=self.matcher.threshold,
                detected_face=live_face
            )
