"""
tests/test_sessions.py - Unit tests for Exam Session Management and Session-Linked Attendance.
"""

import unittest
import os
import shutil
from database.db import Database
import config


class TestExamSessions(unittest.TestCase):
    """Test suite for exam session creation, querying, deletion, and session attendance."""

    def setUp(self):
        self.test_db_path = os.path.join(config.DATA_DIR, "test_sessions_db.db")
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        self.db = Database(self.test_db_path)

    def tearDown(self):
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

    def test_01_create_and_get_session(self):
        """Test creating an exam session and retrieving by ID."""
        sid = self.db.create_exam_session(
            session_code="CS101-MID",
            course_name="Computer Science",
            exam_title="Data Structures & Algorithms",
            hall_number="Hall 302",
            exam_date="2026-10-15",
            start_time="09:30 AM",
            end_time="12:30 PM"
        )
        self.assertGreater(sid, 0)

        session = self.db.get_session_by_id(sid)
        self.assertIsNotNone(session)
        self.assertEqual(session["session_code"], "CS101-MID")
        self.assertEqual(session["course_name"], "Computer Science")
        self.assertEqual(session["status"], "ACTIVE")

    def test_02_filter_active_sessions(self):
        """Test filtering active sessions and search queries."""
        self.db.create_exam_session("PHY201", "Physics", "Mechanics", "Hall A", "2026-10-16", "10:00 AM", "01:00 PM")
        self.db.create_exam_session("CHEM301", "Chemistry", "Organic Chem", "Hall B", "2026-10-17", "02:00 PM", "05:00 PM", status="COMPLETED")

        active = self.db.get_active_sessions()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["session_code"], "PHY201")

        search_res = self.db.get_all_sessions(search_query="Organic")
        self.assertEqual(len(search_res), 1)
        self.assertEqual(search_res[0]["session_code"], "CHEM301")

    def test_03_session_linked_attendance(self):
        """Test marking attendance with session_id and pass_code."""
        # Add student
        dummy_enc = b"\x00" * 512
        stu_id = self.db.add_student("Aadarsh Soni", "CS202601", dummy_enc, "Computer Science")

        # Create session
        sid = self.db.create_exam_session("CS101-FIN", "CS", "Finals", "Hall 1", "2026-10-20", "09:00 AM", "12:00 PM")

        # Mark attendance linked with session
        success, msg = self.db.mark_attendance(
            student_id=stu_id,
            roll_number="CS202601",
            date_str="2026-10-20",
            time_str="09:05:00",
            status="PRESENT",
            session_id=sid,
            pass_code="PASS-CS202601-ABC"
        )
        self.assertTrue(success)

        # Retrieve records by session_id
        records = self.db.get_attendance_records(session_id=sid)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["roll_number"], "CS202601")
        self.assertEqual(records[0]["session_code"], "CS101-FIN")
        self.assertEqual(records[0]["pass_code"], "PASS-CS202601-ABC")


if __name__ == "__main__":
    unittest.main()
