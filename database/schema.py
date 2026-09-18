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

-- Attendance table records verified attendances
-- UNIQUE constraint on (student_id, date) prevents duplicate marking on the same day/session
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    roll_number TEXT NOT NULL COLLATE NOCASE,
    date TEXT NOT NULL,
    time TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PRESENT',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
    UNIQUE(student_id, date)
);

CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);
CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance(student_id);

-- Authentication logs for auditing all verification attempts
CREATE TABLE IF NOT EXISTS authentication_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    roll_number TEXT NOT NULL COLLATE NOCASE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    result TEXT NOT NULL,
    similarity_score REAL,
    details TEXT,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON authentication_logs(timestamp);
"""


def init_db(db_path: str = None) -> None:
    """
    Initialize SQLite database schema and indices.
    Safe to call repeatedly; uses IF NOT EXISTS.
    """
    if db_path is None:
        db_path = config.DB_PATH

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    logger.info(f"Initializing database at: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            conn.executescript(SCHEMA_SQL)
        logger.info("Database schema initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Failed to initialize database schema: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at:", config.DB_PATH)
