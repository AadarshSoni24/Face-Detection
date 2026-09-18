"""
services/attendance_service.py - Exam Attendance Business Logic.

Enforces attendance rules:
- Attendance is recorded ONLY after successful biometric authentication.
- Duplicate attendance for the same student on the same date/session is strictly prevented.
- Provides queries for auditing and reporting attendance records.
"""

from typing import Tuple, List, Dict, Any, Optional
import datetime
import logging
from database.db import Database

logger = logging.getLogger("ExamAuth.AttendanceService")


class AttendanceService:
    """Manages recording and querying attendance with duplicate prevention."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()

    def record_attendance(
        self,
        student_id: int,
        roll_number: str,
        date_str: Optional[str] = None,
        time_str: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Record attendance for verified student.
        Returns: (success: bool, message: str)
        """
        today = date_str or datetime.date.today().strftime("%Y-%m-%d")
        now_time = time_str or datetime.datetime.now().strftime("%H:%M:%S")

        success, message = self.db.mark_attendance(
            student_id=student_id,
            roll_number=roll_number,
            date_str=today,
            time_str=now_time,
            status="PRESENT"
        )
        return success, message

    def is_already_marked(self, student_id: int) -> bool:
        """Check if student has already been marked present today."""
        return self.db.is_attendance_marked_today(student_id)

    def get_attendance_list(
        self,
        date_str: Optional[str] = None,
        search_query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve attendance records with optional date and name/roll filters."""
        return self.db.get_attendance_records(date_str=date_str, search_query=search_query)
