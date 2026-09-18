"""
database/db.py - Database connection management and parameterized CRUD operations.
"""

import sqlite3
import datetime
from typing import Optional, List, Dict, Any, Tuple
import logging
from contextlib import contextmanager
import config
from database.schema import init_db

logger = logging.getLogger("ExamAuth.Database")


class Database:
    """Manages SQLite database connections and thread-safe parameterized operations."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DB_PATH
        # Automatically initialize schema if not present
        init_db(self.db_path)

    @contextmanager
    def get_connection(self):
        """Create and yield a configured SQLite connection, ensuring proper closure."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---------------------------------------------------------
    # Student Operations
    # ---------------------------------------------------------
    def add_student(
        self,
        name: str,
        roll_number: str,
        face_encoding_bytes: bytes,
        course: str = "",
        registration_type: str = "WEBCAM"
    ) -> int:
        """
        Insert a new student with primary face encoding.
        Returns newly inserted student ID.
        Raises sqlite3.IntegrityError if roll_number already exists.
        """
        name = name.strip()
        roll_number = roll_number.strip().upper()
        course = course.strip()

        query = """
            INSERT INTO students (name, roll_number, course, face_encoding, registration_type)
            VALUES (?, ?, ?, ?, ?);
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (name, roll_number, course, face_encoding_bytes, registration_type))
            student_id = cursor.lastrowid
            logger.info(f"Registered student id={student_id}, roll='{roll_number}', name='{name}'")
            return student_id

    def add_reference_encoding(
        self,
        student_id: int,
        face_encoding_bytes: bytes,
        source_label: str = "reference"
    ) -> int:
        """Insert an additional reference face encoding for an existing student."""
        query = """
            INSERT INTO student_reference_encodings (student_id, face_encoding, source_label)
            VALUES (?, ?, ?);
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (student_id, face_encoding_bytes, source_label))
            return cursor.lastrowid

    def get_student_by_roll(self, roll_number: str) -> Optional[Dict[str, Any]]:
        """Retrieve student record by roll number (case-insensitive)."""
        roll_number = roll_number.strip().upper()
        query = "SELECT * FROM students WHERE roll_number = ? COLLATE NOCASE;"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (roll_number,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_student_by_id(self, student_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve student record by student ID."""
        query = "SELECT * FROM students WHERE id = ?;"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (student_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def student_exists(self, roll_number: str) -> bool:
        """Check if a student with given roll number already exists."""
        return self.get_student_by_roll(roll_number) is not None

    def get_student_encodings(self, student_id: int) -> List[bytes]:
        """
        Retrieve all approved face encodings for a student:
        Primary encoding from students table + any secondary reference encodings.
        """
        encodings = []
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Primary encoding
            cursor.execute("SELECT face_encoding FROM students WHERE id = ?;", (student_id,))
            row = cursor.fetchone()
            if row and row["face_encoding"]:
                encodings.append(row["face_encoding"])

            # Reference encodings
            cursor.execute(
                "SELECT face_encoding FROM student_reference_encodings WHERE student_id = ? ORDER BY id ASC;",
                (student_id,)
            )
            for r in cursor.fetchall():
                encodings.append(r["face_encoding"])
        return encodings

    def get_all_students_with_encodings(self) -> List[Dict[str, Any]]:
        """
        Retrieve all registered students along with their approved face encoding BLOBs.
        Used for 1:N identification (automatic recognition without typing roll number).
        """
        results = []
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, roll_number, name, course, face_encoding FROM students;")
            students = cursor.fetchall()
            for s in students:
                s_dict = dict(s)
                sid = s_dict["id"]
                encs = [s_dict["face_encoding"]] if s_dict["face_encoding"] else []

                # Also grab any secondary reference encodings
                cursor.execute(
                    "SELECT face_encoding FROM student_reference_encodings WHERE student_id = ?;",
                    (sid,)
                )
                for ref_row in cursor.fetchall():
                    if ref_row["face_encoding"]:
                        encs.append(ref_row["face_encoding"])

                results.append({
                    "id": sid,
                    "roll_number": s_dict["roll_number"],
                    "name": s_dict["name"],
                    "course": s_dict.get("course", ""),
                    "encodings": encs
                })
        return results

    def get_all_students(self, search_query: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all students, optionally filtered by name or roll number."""
        query = "SELECT id, roll_number, name, course, registration_type, created_at FROM students"
        params = []
        if search_query and search_query.strip():
            term = f"%{search_query.strip()}%"
            query += " WHERE roll_number LIKE ? OR name LIKE ? OR course LIKE ?"
            params.extend([term, term, term])
        query += " ORDER BY roll_number ASC;"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def delete_student(self, student_id: int) -> bool:
        """Delete student and all associated cascading records."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM students WHERE id = ?;", (student_id,))
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted student id={student_id} and associated references.")
            return deleted

    # ---------------------------------------------------------
    # Attendance Operations
    # ---------------------------------------------------------
    def mark_attendance(
        self,
        student_id: int,
        roll_number: str,
        date_str: Optional[str] = None,
        time_str: Optional[str] = None,
        status: str = "PRESENT"
    ) -> Tuple[bool, str]:
        """
        Mark attendance for a student on date_str (default: today YYYY-MM-DD).
        Enforces duplicate attendance prevention for same student on same date.
        Returns (success: bool, message: str).
        """
        now = datetime.datetime.now()
        date_str = date_str or now.strftime("%Y-%m-%d")
        time_str = time_str or now.strftime("%H:%M:%S")
        roll_number = roll_number.strip().upper()

        # Check existing attendance
        check_query = "SELECT id FROM attendance WHERE student_id = ? AND date = ?;"
        insert_query = """
            INSERT INTO attendance (student_id, roll_number, date, time, status)
            VALUES (?, ?, ?, ?, ?);
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(check_query, (student_id, date_str))
            existing = cursor.fetchone()
            if existing:
                msg = f"Attendance already marked for {roll_number} on {date_str}."
                logger.warning(msg)
                return False, msg

            try:
                cursor.execute(insert_query, (student_id, roll_number, date_str, time_str, status))
                msg = "Attendance Marked Successfully"
                logger.info(f"Marked attendance: {roll_number} on {date_str} at {time_str}")
                return True, msg
            except sqlite3.IntegrityError:
                return False, f"Duplicate attendance constraint for {roll_number} on {date_str}."

    def get_attendance_records(
        self,
        date_str: Optional[str] = None,
        search_query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve attendance records joined with student details.
        Optionally filter by date and student roll/name.
        """
        query = """
            SELECT a.id, a.student_id, a.roll_number, s.name, s.course, a.date, a.time, a.status
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE 1=1
        """
        params = []
        if date_str and date_str.strip():
            query += " AND a.date = ?"
            params.append(date_str.strip())
        if search_query and search_query.strip():
            term = f"%{search_query.strip()}%"
            query += " AND (a.roll_number LIKE ? OR s.name LIKE ?)"
            params.extend([term, term])
        query += " ORDER BY a.date DESC, a.time DESC;"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def is_attendance_marked_today(self, student_id: int) -> bool:
        """Check if student has already marked attendance today."""
        today = datetime.date.today().strftime("%Y-%m-%d")
        query = "SELECT id FROM attendance WHERE student_id = ? AND date = ?;"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (student_id, today))
            return cursor.fetchone() is not None

    # ---------------------------------------------------------
    # Authentication Logs & Statistics
    # ---------------------------------------------------------
    def log_auth_attempt(
        self,
        student_id: Optional[int],
        roll_number: str,
        result: str,
        similarity_score: Optional[float] = None,
        details: str = ""
    ) -> int:
        """Log an authentication attempt for audit and security tracking."""
        query = """
            INSERT INTO authentication_logs (student_id, roll_number, result, similarity_score, details)
            VALUES (?, ?, ?, ?, ?);
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (student_id, roll_number.strip().upper(), result, similarity_score, details))
            return cursor.lastrowid

    def get_dashboard_stats(self) -> Dict[str, int]:
        """Fetch summary metrics for the main dashboard display."""
        today = datetime.date.today().strftime("%Y-%m-%d")
        stats = {
            "total_students": 0,
            "attendance_today": 0,
            "failed_auth_today": 0,
            "total_auth_today": 0,
        }
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Registered students
            cursor.execute("SELECT COUNT(*) AS c FROM students;")
            stats["total_students"] = cursor.fetchone()["c"]

            # Today's attendance
            cursor.execute("SELECT COUNT(*) AS c FROM attendance WHERE date = ?;", (today,))
            stats["attendance_today"] = cursor.fetchone()["c"]

            # Today's failed auth attempts
            cursor.execute(
                """
                SELECT COUNT(*) AS c FROM authentication_logs
                WHERE DATE(timestamp) = ? AND result NOT IN ('SUCCESS');
                """,
                (today,)
            )
            stats["failed_auth_today"] = cursor.fetchone()["c"]

            # Today's total auth attempts
            cursor.execute(
                "SELECT COUNT(*) AS c FROM authentication_logs WHERE DATE(timestamp) = ?;",
                (today,)
            )
            stats["total_auth_today"] = cursor.fetchone()["c"]

        return stats
