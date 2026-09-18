"""
tests/test_database.py - Unit tests for SQLite database schema, queries, and constraints.
"""

import unittest
import os
import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from database.db import Database


class TestDatabaseOperations(unittest.TestCase):
    """Test suite for database layer CRUD, constraints, and audit logging."""

    def setUp(self):
        self.test_db_path = str(BASE_DIR / "data" / "test_unit_db.db")
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        self.db = Database(self.test_db_path)
        self.dummy_embedding = np.ones(128, dtype=np.float32).tobytes()

    def tearDown(self):
        # Force release and remove test database
        del self.db
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except Exception:
                pass

    def test_01_add_and_retrieve_student(self):
        """Test inserting a new student and retrieving by roll number and ID."""
        sid = self.db.add_student(
            name="Rahul Kumar",
            roll_number="101",
            face_encoding_bytes=self.dummy_embedding,
            course="CS"
        )
        self.assertGreater(sid, 0)

        # Retrieve by roll
        student = self.db.get_student_by_roll("101")
        self.assertIsNotNone(student)
        self.assertEqual(student["name"], "Rahul Kumar")
        self.assertEqual(student["roll_number"], "101")
        self.assertEqual(student["course"], "CS")

        # Retrieve by ID
        student_by_id = self.db.get_student_by_id(sid)
        self.assertIsNotNone(student_by_id)
        self.assertEqual(student_by_id["roll_number"], "101")

    def test_02_duplicate_roll_number_rejection(self):
        """Test that duplicate roll numbers are strictly rejected by SQLite UNIQUE constraint."""
        self.db.add_student("Rahul Kumar", "101", self.dummy_embedding, "CS")
        import sqlite3
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.add_student("Duplicate Rahul", "101", self.dummy_embedding, "IT")

    def test_03_student_exists(self):
        """Test existence check for roll numbers."""
        self.db.add_student("Rahul Kumar", "101", self.dummy_embedding)
        self.assertTrue(self.db.student_exists("101"))
        self.assertTrue(self.db.student_exists("101 "))   # whitespace stripped
        self.assertFalse(self.db.student_exists("999"))

    def test_04_multiple_reference_encodings(self):
        """Test storing and retrieving multiple reference face encodings per student."""
        sid = self.db.add_student("Rahul Kumar", "101", self.dummy_embedding)
        dummy_ref2 = (np.ones(128, dtype=np.float32) * 2).tobytes()
        dummy_ref3 = (np.ones(128, dtype=np.float32) * 3).tobytes()

        self.db.add_reference_encoding(sid, dummy_ref2, "photo2.jpg")
        self.db.add_reference_encoding(sid, dummy_ref3, "photo3.jpg")

        encodings = self.db.get_student_encodings(sid)
        self.assertEqual(len(encodings), 3)  # Primary + 2 references

    def test_05_mark_attendance_and_prevent_duplicate(self):
        """Test marking attendance and enforcing duplicate prevention on the same date."""
        sid = self.db.add_student("Rahul Kumar", "101", self.dummy_embedding)

        # First marking: should succeed
        success, msg = self.db.mark_attendance(sid, "101", date_str="2026-09-17")
        self.assertTrue(success)
        self.assertIn("Successfully", msg)

        # Duplicate marking on same date: should be blocked!
        dup_success, dup_msg = self.db.mark_attendance(sid, "101", date_str="2026-09-17")
        self.assertFalse(dup_success)
        self.assertIn("already marked", dup_msg.lower())

        # Attendance on a different date: should succeed!
        next_day_success, _ = self.db.mark_attendance(sid, "101", date_str="2026-09-18")
        self.assertTrue(next_day_success)

    def test_06_delete_student_cascade(self):
        """Test deleting student record and cascading removal of attendance/references."""
        sid = self.db.add_student("Rahul Kumar", "101", self.dummy_embedding)
        self.db.mark_attendance(sid, "101")
        self.db.add_reference_encoding(sid, self.dummy_embedding)

        deleted = self.db.delete_student(sid)
        self.assertTrue(deleted)
        self.assertIsNone(self.db.get_student_by_roll("101"))
        self.assertEqual(len(self.db.get_student_encodings(sid)), 0)

    def test_07_audit_logging_and_dashboard_stats(self):
        """Test logging authentication attempts and computing real-time dashboard analytics."""
        sid = self.db.add_student("Rahul Kumar", "101", self.dummy_embedding)
        self.db.mark_attendance(sid, "101")

        # Log 1 success and 2 failures
        self.db.log_auth_attempt(sid, "101", "SUCCESS", 0.85, "Verified")
        self.db.log_auth_attempt(sid, "101", "FAILED", 0.12, "Face mismatch")
        self.db.log_auth_attempt(None, "999", "NOT_FOUND", None, "Student not found")

        stats = self.db.get_dashboard_stats()
        self.assertEqual(stats["total_students"], 1)
        self.assertEqual(stats["attendance_today"], 1)
        self.assertEqual(stats["failed_auth_today"], 2)
        self.assertEqual(stats["total_auth_today"], 3)


if __name__ == "__main__":
    unittest.main()
