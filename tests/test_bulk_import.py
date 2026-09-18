"""
tests/test_bulk_import.py - Unit and integration tests for Bulk Student Photo & CSV Import.
"""

import unittest
import os
import sys
import shutil
from pathlib import Path
import cv2

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from database.db import Database
from recognition.face_detector import FaceDetector
from recognition.face_encoder import FaceEncoder
from recognition.face_matcher import FaceMatcher
from services.import_service import ImportService
from services.authentication_service import AuthenticationService
from services.attendance_service import AttendanceService


class TestBulkPhotoImport(unittest.TestCase):
    """Test suite for batch photo validation, CSV mapping, and subsequent authentication."""

    def setUp(self):
        self.test_db_path = str(BASE_DIR / "data" / "test_import_db.db")
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

        self.db = Database(self.test_db_path)
        self.detector = FaceDetector()
        self.encoder = FaceEncoder()
        self.matcher = FaceMatcher()
        self.import_service = ImportService(self.db, self.detector, self.encoder)

        self.sample_dir = str(BASE_DIR / "data" / "sample_import")
        self.csv_path = str(BASE_DIR / "data" / "sample_import" / "students_mapping.csv")

    def tearDown(self):
        del self.db
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except Exception:
                pass

    def test_01_bulk_import_with_csv(self):
        """Test full bulk import workflow with CSV mapping and diverse image qualities."""
        summary = self.import_service.run_import(
            photos_folder=self.sample_dir,
            csv_mapping_path=self.csv_path
        )

        print(
            f"\n[Bulk Import Execution Results]\n"
            f"Total files: {summary.total_files}\n"
            f"Processed successfully: {summary.processed_successfully}\n"
            f"Needs review: {summary.needs_review}\n"
            f"Failed: {summary.failed}"
        )

        # 101 (Rahul) and 102 (Priya) have valid single faces -> Processed
        self.assertGreaterEqual(summary.processed_successfully, 2)

        # 103 (no_face) and 104 (multi_face) -> Needs Review
        self.assertGreaterEqual(summary.needs_review, 2)

        # 105 (corrupt_image) -> Failed
        self.assertGreaterEqual(summary.failed, 1)

        # Verify students registered in database
        s1 = self.db.get_student_by_roll("101")
        self.assertIsNotNone(s1)
        self.assertEqual(s1["name"], "Rahul Kumar")
        self.assertEqual(s1["registration_type"], "BULK_IMPORT")

        s2 = self.db.get_student_by_roll("102")
        self.assertIsNotNone(s2)
        self.assertEqual(s2["name"], "Priya Sharma")

    def test_02_imported_student_authenticates_successfully(self):
        """Verify that an imported student authenticates identically to a manually registered student."""
        # 1. Run import
        self.import_service.run_import(
            photos_folder=self.sample_dir,
            csv_mapping_path=self.csv_path
        )

        # 2. Authenticate student 101 with Face 1 frame
        att_service = AttendanceService(self.db)
        auth_service = AuthenticationService(
            self.db, self.detector, self.encoder, self.matcher, att_service
        )

        face1_img = cv2.imread(os.path.join(self.sample_dir, "rahul_face.jpg"))
        auth_res = auth_service.authenticate("101", face1_img)

        self.assertTrue(auth_res.success)
        self.assertEqual(auth_res.status_title, "IDENTITY VERIFIED")
        self.assertEqual(auth_res.face_match_status, "SUCCESS")
        self.assertEqual(auth_res.attendance_status, "PRESENT")
        self.assertEqual(auth_res.exam_access, "GRANTED")

        # 3. Authenticate with wrong face (Priya's face against roll 101) -> Denied
        face2_img = cv2.imread(os.path.join(self.sample_dir, "priya_face.jpg"))
        mismatch_res = auth_service.authenticate("101", face2_img)
        self.assertFalse(mismatch_res.success)
        self.assertEqual(mismatch_res.status_title, "AUTHENTICATION FAILED")
        self.assertEqual(mismatch_res.exam_access, "DENIED")

    def test_03_missing_csv_unreliable_filenames_requires_review(self):
        """Verify that without CSV, ambiguous filenames are flagged for Manual Review rather than guessing."""
        temp_test_dir = str(BASE_DIR / "data" / "temp_ambiguous_test")
        os.makedirs(temp_test_dir, exist_ok=True)
        try:
            # Copy a face with an ambiguous filename like "IMG_0099.jpg"
            shutil.copy(
                os.path.join(self.sample_dir, "rahul_face.jpg"),
                os.path.join(temp_test_dir, "IMG_0099.jpg")
            )
            summary = self.import_service.run_import(photos_folder=temp_test_dir, csv_mapping_path=None)
            self.assertEqual(summary.needs_review, 1)
            self.assertEqual(summary.processed_successfully, 0)
            self.assertIn("Manual Review Required", summary.items[0].message)
        finally:
            shutil.rmtree(temp_test_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
