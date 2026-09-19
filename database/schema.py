"""
database/schema.py - SQLite Schema definitions and database initialization.
"""

import sqlite3
import os
import logging
import config

logger = logging.getLogger("ExamAuth.Database")

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

-- Students table stores core demographic details and primary 128-d face embedding
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_number TEXT UNIQUE NOT NULL COLLATE NOCASE,
    name TEXT NOT NULL,
    course TEXT DEFAULT '',
    face_encoding BLOB NOT NULL,
    registration_type TEXT DEFAULT 'WEBCAM',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast lookup by roll number
CREATE INDEX IF NOT EXISTS idx_students_roll ON students(roll_number);

-- Optional multiple reference face encodings per student
CREATE TABLE IF NOT EXISTS student_reference_encodings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    face_encoding BLOB NOT NULL,
    source_label TEXT DEFAULT 'reference',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ref_encodings_student ON student_reference_encodings(student_id);

-- Exam Sessions table for scheduling exam courses, titles, halls and time slots
CREATE TABLE IF NOT EXISTS exam_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_code TEXT UNIQUE NOT NULL COLLATE NOCASE,
    course_name TEXT NOT NULL,
    exam_title TEXT NOT NULL,
    hall_number TEXT NOT NULL,
    exam_date TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sessions_date ON exam_sessions(exam_date);
CREATE INDEX IF NOT EXISTS idx_sessions_code ON exam_sessions(session_code);

-- Attendance table records verified attendances
-- UNIQUE constraint on (student_id, date) prevents duplicate marking on the same day/session
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    roll_number TEXT NOT NULL COLLATE NOCASE,
    session_id INTEGER,
    date TEXT NOT NULL,
    time TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PRESENT',
    pass_code TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES exam_sessions(id) ON DELETE SET NULL,
    UNIQUE(student_id, date)
);

CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);
CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance(student_id);

-- Authentication logs for auditing all verification attempts
CREATE TABLE IF NOT EXISTS authentication_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    roll_number TEXT NOT NULL COLLATE NOCASE,
    session_id INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    result TEXT NOT NULL,
    similarity_score REAL,
    liveness_score REAL,
    details TEXT,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE SET NULL,
    FOREIGN KEY (session_id) REFERENCES exam_sessions(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON authentication_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_result ON authentication_logs(result);
"""


def _migrate_columns_if_missing(conn: sqlite3.Connection) -> None:
    """Helper to ensure new columns exist in existing database files."""
    cursor = conn.cursor()
    # Check attendance columns
    cursor.execute("PRAGMA table_info(attendance);")
    att_cols = [row[1] for row in cursor.fetchall()]
    if "session_id" not in att_cols:
        cursor.execute("ALTER TABLE attendance ADD COLUMN session_id INTEGER;")
    if "pass_code" not in att_cols:
        cursor.execute("ALTER TABLE attendance ADD COLUMN pass_code TEXT;")

    # Check authentication_logs columns
    cursor.execute("PRAGMA table_info(authentication_logs);")
    log_cols = [row[1] for row in cursor.fetchall()]
    if "session_id" not in log_cols:
        cursor.execute("ALTER TABLE authentication_logs ADD COLUMN session_id INTEGER;")
    if "liveness_score" not in log_cols:
        cursor.execute("ALTER TABLE authentication_logs ADD COLUMN liveness_score REAL;")


def init_db(db_path: str = None) -> None:
    """
    Initialize SQLite database schema and indices.
    Safe to call repeatedly; uses IF NOT EXISTS and migrates existing tables.
    """
    if db_path is None:
        db_path = config.DB_PATH

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    logger.info(f"Initializing database at: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            conn.executescript(SCHEMA_SQL)
            _migrate_columns_if_missing(conn)
        logger.info("Database schema initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Failed to initialize database schema: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at:", config.DB_PATH)

