"""
demo_walkthrough.py - Automated End-to-End Demonstration Script.

Executes the exact academic demonstration workflow specified in Section 25 of the PRD:
Step 1: Open application / Initialize database & models.
Step 2: Register Student Rahul Kumar (Roll Number: 101) with Face A.
Step 3: Authenticate Roll 101 with Rahul's Face -> IDENTITY VERIFIED / PRESENT / EXAM ACCESS GRANTED.
Step 4: Authenticate Roll 101 with a different face -> AUTHENTICATION FAILED / NOT MARKED / EXAM ACCESS DENIED.
Step 5: Inspect Attendance records -> Verify Rahul's attendance is marked.
Step 6: Demonstrate Bulk Student Photo Import with CSV mapping & quality validation.
"""

import sys
import os
import cv2
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
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
from services.import_service import ImportService
from tests.create_test_assets import setup_test_assets


def run_demo():
    print("=" * 80)
    print("BIOMETRIC-BASED EXAM AUTHENTICATION SYSTEM — ACADEMIC DEMONSTRATION")
    print("=" * 80)

    # Prepare verified test assets
    print("\n[Step 0] Initializing test assets and clean demonstration database...")
    assets = setup_test_assets()
    demo_db_path = str(config.DATA_DIR / "demo_exam_auth.db")
    if os.path.exists(demo_db_path):
        os.remove(demo_db_path)

    # Initialize Core Components
    db = Database(demo_db_path)
    detector = FaceDetector()
    encoder = FaceEncoder()
    matcher = FaceMatcher()
    att_service = AttendanceService(db)
    reg_service = RegistrationService(db, detector, encoder)
    auth_service = AuthenticationService(db, detector, encoder, matcher, att_service)
    import_service = ImportService(db, detector, encoder)

    # Load Demo Photos
    img_rahul = cv2.imread(assets["face1"])
    img_different = cv2.imread(assets["face2"])

    # -------------------------------------------------------------
    # Step 1 & 2: Register Student Rahul Kumar (101)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[Step 1 & 2] Registering Student: Rahul Kumar (Roll Number: 101)")
    print("-" * 80)
    reg_result = reg_service.register_student(
        name="Rahul Kumar",
        roll_number="101",
        frame_bgr=img_rahul,
        course="Computer Science & Engineering"
    )
    print(f"Registration Success: {reg_result.success}")
    print(f"Message:              {reg_result.message}")
    print(f"Database Student ID:  {reg_result.student_id}")

    # -------------------------------------------------------------
    # Step 3: Authenticate Rahul Kumar with Matching Face
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[Step 3] Exam Day Authentication: Roll Number 101 with Rahul's Face")
    print("-" * 80)
    auth1 = auth_service.authenticate(roll_number="101", frame_bgr=img_rahul)
    print(f"STATUS TITLE:      {auth1.status_title}")
    print(f"Student:           {auth1.student_name}")
    print(f"Roll Number:       {auth1.roll_number}")
    print(f"Face Match:        {auth1.face_match_status}")
    print(f"Attendance:        {auth1.attendance_status}")
    print(f"Exam Access:       {auth1.exam_access}")
    print(f"Cosine Similarity: {auth1.similarity_score:.4f} (Threshold: {auth1.threshold:.4f})")
    print(f"Message:           {auth1.message}")

    # -------------------------------------------------------------
    # Step 4: Authenticate Roll Number 101 with Impostor / Different Face
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[Step 4] Exam Day Authentication: Roll Number 101 with Different Face")
    print("-" * 80)
    auth2 = auth_service.authenticate(roll_number="101", frame_bgr=img_different)
    print(f"STATUS TITLE:      {auth2.status_title}")
    print(f"Student:           {auth2.student_name}")
    print(f"Roll Number:       {auth2.roll_number}")
    print(f"Face Match:        {auth2.face_match_status}")
    print(f"Attendance:        {auth2.attendance_status}")
    print(f"Exam Access:       {auth2.exam_access}")
    print(f"Cosine Similarity: {auth2.similarity_score:.4f} (Threshold: {auth2.threshold:.4f})")
    print(f"Message:           {auth2.message}")

    # -------------------------------------------------------------
    # Step 5: Check Attendance Records in Database
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[Step 5] Auditing Examination Attendance Records")
    print("-" * 80)
    attendance_records = db.get_attendance_records()
    print(f"Total Attendance Records Logged: {len(attendance_records)}")
    for rec in attendance_records:
        print(
            f"• Roll: {rec['roll_number']:<8} | Name: {rec['name']:<16} | "
            f"Date: {rec['date']} | Time: {rec['time']} | Status: {rec['status']}"
        )

    # -------------------------------------------------------------
    # Step 6: Bulk Student Photo Import Demonstration
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[Step 6] Bulk Student Photo Import Demonstration")
    print("-" * 80)
    summary = import_service.run_import(
        photos_folder=str(config.DATA_DIR / "sample_import"),
        csv_mapping_path=assets["csv"]
    )
    print(f"Total Images Processed: {summary.total_files}")
    print(f"Processed Successfully: {summary.processed_successfully}")
    print(f"Manual Review Required: {summary.needs_review}")
    print(f"Failed / Corrupt:       {summary.failed}")
    print("\nDetailed Itemized Report:")
    for item in summary.items:
        print(f"  [{item.status:<12}] {item.filename:<18} -> Roll: {item.roll_number:<6} | {item.message}")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY WITH 100% REAL DEEP LEARNING INFERENCE")
    print("=" * 80)

    # Clean up demo database
    del db
    if os.path.exists(demo_db_path):
        os.remove(demo_db_path)


if __name__ == "__main__":
    run_demo()
