"""
tests/test_services.py - Integration tests for RegistrationService, AuthenticationService, and AttendanceService.
"""

import unittest
import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from database.db import Database
from recognition.face_detector import FaceDetector
from recognition.face_encoder import FaceEncoder
from recognition.face_matcher import FaceMatcher
from services.registration_service import RegistrationService
from services.authentication_service import AuthenticationService
from services.attendance_service import AttendanceService


class TestServicesWorkflow(unittest.TestCase):
    """End-to-end integration tests for student enrollment, authentication, and attendance."""

    def setUp(self):
        self.test_db_path = str(BASE_DIR / "data" / "test_services_db.db")
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

        self.db = Database(self.test_db_path)
        self.detector = FaceDetector()
        self.encoder = FaceEncoder()
        self.matcher = FaceMatcher()
        self.att_service = AttendanceService(self.db)
        self.reg_service = RegistrationService(self.db, self.detector, self.encoder)
        self.auth_service = AuthenticationService(
            self.db, self.detector, self.encoder, self.matcher, self.att_service
        )

        assets_dir = BASE_DIR / "data" / "sample_import"
        self.face1_img = cv2.imread(str(assets_dir / "rahul_face.jpg"))
        self.face2_img = cv2.imread(str(assets_dir / "priya_face.jpg"))
        self.noface_img = cv2.imread(str(assets_dir / "no_face.jpg"))
        self.multiface_img = cv2.imread(str(assets_dir / "multi_face.jpg"))

    def tearDown(self):
        del self.db
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except Exception:
                pass

    def test_01_valid_registration(self):
        """Test registering a valid student with live face capture."""
        res = self.reg_service.register_student(
            name="Rahul Kumar",
            roll_number="101",
            frame_bgr=self.face1_img,
            course="Computer Science"
        )
        self.assertTrue(res.success)
        self.assertEqual(res.message, "Student Registered Successfully")
        self.assertIsNotNone(res.student_id)

        # Verify record exists in DB
        student = self.db.get_student_by_roll("101")
        self.assertIsNotNone(student)
        self.assertEqual(student["name"], "Rahul Kumar")

    def test_02_duplicate_roll_registration(self):
        """Test that registering an existing roll number fails with clear message."""
        self.reg_service.register_student("Rahul Kumar", "101", self.face1_img)
        res = self.reg_service.register_student("Duplicate Rahul", "101", self.face1_img)
        self.assertFalse(res.success)
        self.assertIn("already registered", res.message)

    def test_03_missing_student_info_registration(self):
        """Test registration validation when name or roll number is omitted."""
        res1 = self.reg_service.register_student("", "101", self.face1_img)
        self.assertFalse(res1.success)
        self.assertIn("Name is required", res1.message)

        res2 = self.reg_service.register_student("Rahul Kumar", "", self.face1_img)
        self.assertFalse(res2.success)
        self.assertIn("Roll Number is required", res2.message)

    def test_04_no_face_during_registration(self):
        """Test registration failure when no face is present in camera frame."""
        res = self.reg_service.register_student("Amit Verma", "103", self.noface_img)
        self.assertFalse(res.success)
        self.assertIn("No face detected", res.message)

    def test_05_multiple_faces_during_registration(self):
        """Test registration failure when more than 1 face is visible."""
        res = self.reg_service.register_student("Sneha Patel", "104", self.multiface_img)
        self.assertFalse(res.success)
        self.assertIn("Multiple faces detected", res.message)

    def test_06_valid_face_authentication_and_attendance(self):
        """Test successful authentication of registered student matching live face."""
        # 1. Register student Rahul Kumar (roll 101) with Face 1
        reg_res = self.reg_service.register_student("Rahul Kumar", "101", self.face1_img)
        self.assertTrue(reg_res.success)

        # 2. Authenticate using Face 1
        auth_res = self.auth_service.authenticate("101", self.face1_img)
        self.assertTrue(auth_res.success)
        self.assertEqual(auth_res.status_title, "IDENTITY VERIFIED")
        self.assertEqual(auth_res.face_match_status, "SUCCESS")
        self.assertEqual(auth_res.attendance_status, "PRESENT")
        self.assertEqual(auth_res.exam_access, "GRANTED")

        # 3. Verify attendance was recorded in DB
        records = self.db.get_attendance_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["roll_number"], "101")
        self.assertEqual(records[0]["status"], "PRESENT")

    def test_07_face_mismatch_authentication(self):
        """Test authentication denial when presenting a different person's face."""
        # Register Rahul with Face 1
        self.reg_service.register_student("Rahul Kumar", "101", self.face1_img)

        # Attempt auth with Face 2 (Priya / Impostor)
        auth_res = self.auth_service.authenticate("101", self.face2_img)
        self.assertFalse(auth_res.success)
        self.assertEqual(auth_res.status_title, "AUTHENTICATION FAILED")
        self.assertEqual(auth_res.face_match_status, "FAILED")
        self.assertEqual(auth_res.attendance_status, "NOT MARKED")
        self.assertEqual(auth_res.exam_access, "DENIED")
        self.assertIn("does not match", auth_res.message)

        # Verify attendance was NOT recorded
        records = self.db.get_attendance_records()
        self.assertEqual(len(records), 0)

    def test_08_unknown_roll_number(self):
        """Test authentication with roll number that does not exist in the database."""
        auth_res = self.auth_service.authenticate("999", self.face1_img)
        self.assertFalse(auth_res.success)
        self.assertEqual(auth_res.message, "Student not found.")
        self.assertEqual(auth_res.exam_access, "DENIED")

    def test_09_no_face_during_authentication(self):
        """Test authentication failure when live webcam has no face in frame."""
        self.reg_service.register_student("Rahul Kumar", "101", self.face1_img)
        auth_res = self.auth_service.authenticate("101", self.noface_img)
        self.assertFalse(auth_res.success)
        self.assertIn("No face detected", auth_res.message)
        self.assertEqual(auth_res.exam_access, "DENIED")

    def test_10_multiple_faces_during_authentication(self):
        """Test authentication failure when multiple faces are detected."""
        self.reg_service.register_student("Rahul Kumar", "101", self.face1_img)
        auth_res = self.auth_service.authenticate("101", self.multiface_img)
        self.assertFalse(auth_res.success)
        self.assertIn("Multiple faces detected", auth_res.message)
        self.assertEqual(auth_res.exam_access, "DENIED")

    def test_11_duplicate_attendance_attempt(self):
        """Test that re-authenticating the same student on the same day does not duplicate attendance."""
        self.reg_service.register_student("Rahul Kumar", "101", self.face1_img)

        # First auth
        res1 = self.auth_service.authenticate("101", self.face1_img)
        self.assertEqual(res1.attendance_status, "PRESENT")

        # Second auth today
        res2 = self.auth_service.authenticate("101", self.face1_img)
        self.assertTrue(res2.success)
        self.assertIn("ALREADY MARKED", res2.attendance_status)

        # Verify only 1 record in database
        records = self.db.get_attendance_records()
        self.assertEqual(len(records), 1)

    def test_12_camera_frame_unavailable(self):
        """Test graceful handling when camera frame is None."""
        self.reg_service.register_student("Rahul Kumar", "101", self.face1_img)
        auth_res = self.auth_service.authenticate("101", None)
        self.assertFalse(auth_res.success)
        self.assertIn("Camera could not be accessed", auth_res.message)

    def test_13_auto_identify_face_without_roll_number(self):
        """Test 1:N automatic identification where student face is scanned and roll number is auto-filled."""
        # 1. Register two distinct students
        self.reg_service.register_student("Rahul Kumar", "101", self.face1_img, "Computer Science")
        self.reg_service.register_student("Priya Sharma", "102", self.face2_img, "IT")

        # 2. Auto-identify Rahul's face without supplying roll number
        res_rahul = self.auth_service.identify_face(self.face1_img)
        self.assertTrue(res_rahul.success)
        self.assertEqual(res_rahul.roll_number, "101")
        self.assertEqual(res_rahul.student_name, "Rahul Kumar")
        self.assertEqual(res_rahul.face_match_status, "SUCCESS")
        self.assertEqual(res_rahul.attendance_status, "PRESENT")
        self.assertEqual(res_rahul.exam_access, "GRANTED")

        # 3. Auto-identify Priya's face without supplying roll number
        res_priya = self.auth_service.identify_face(self.face2_img)
        self.assertTrue(res_priya.success)
        self.assertEqual(res_priya.roll_number, "102")
        self.assertEqual(res_priya.student_name, "Priya Sharma")
        self.assertEqual(res_priya.exam_access, "GRANTED")

        # 4. Check that attendance was marked for both automatically
        records = self.db.get_attendance_records()
        self.assertEqual(len(records), 2)
        rolls_marked = {r["roll_number"] for r in records}
        self.assertEqual(rolls_marked, {"101", "102"})


if __name__ == "__main__":
    unittest.main()
