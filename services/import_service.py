"""
services/import_service.py - Bulk Student Photo & CSV Import Pipeline.

Imports authorized student photographs from a local college export directory,
validating image integrity, detecting faces, extracting deep neural network
biometric encodings, and linking them to SQLite student records.
"""

import os
import csv
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
from pathlib import Path
import cv2
import numpy as np
import logging
from database.db import Database
from recognition.face_detector import FaceDetector
from recognition.face_encoder import FaceEncoder

logger = logging.getLogger("ExamAuth.ImportService")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class ImportItemResult:
    """Detailed record for a single processed image file."""
    filename: str
    roll_number: str
    student_name: str
    status: str       # "SUCCESS", "NEEDS_REVIEW", "FAILED"
    category: str     # "PROCESSED", "MANUAL_REVIEW", "FAILURE"
    message: str


@dataclass
class ImportSummary:
    """Aggregate statistics and itemized outcomes of a bulk import run."""
    total_files: int = 0
    processed_successfully: int = 0
    needs_review: int = 0
    failed: int = 0
    items: List[ImportItemResult] = field(default_factory=list)


class ImportService:
    """
    Orchestrates bulk photo validation, face detection, encoding generation,
    and database population from a local directory and optional CSV mapping.
    """

    def __init__(
        self,
        db: Optional[Database] = None,
        detector: Optional[FaceDetector] = None,
        encoder: Optional[FaceEncoder] = None
    ):
        self.db = db or Database()
        self.detector = detector or FaceDetector()
        self.encoder = encoder or FaceEncoder()
        self._cancel_requested = False

    def request_cancel(self) -> None:
        """Signal the running import loop to stop."""
        self._cancel_requested = True

    @staticmethod
    def parse_csv_mapping(csv_path: str) -> Dict[str, Dict[str, str]]:
        """
        Parse student mapping CSV file.
        Accepts headers: roll_number, name, image_filename, [optional: course].
        Returns a dictionary mapping image_filename (normalized lower) -> {roll_number, name, course}.
        """
        mapping: Dict[str, Dict[str, str]] = {}
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Mapping CSV file not found: {csv_path}")

        with open(csv_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            # Normalize fieldnames to lowercase
            if not reader.fieldnames:
                raise ValueError("CSV file is empty or missing headers.")

            field_map = {name.strip().lower(): name for name in reader.fieldnames}
            roll_key = field_map.get("roll_number") or field_map.get("roll") or field_map.get("roll_no")
            name_key = field_map.get("name") or field_map.get("student_name")
            file_key = field_map.get("image_filename") or field_map.get("filename") or field_map.get("photo")
            course_key = field_map.get("course") or field_map.get("class")

            if not (roll_key and name_key and file_key):
                raise ValueError(
                    f"CSV must contain 'roll_number', 'name', and 'image_filename' columns. Found: {reader.fieldnames}"
                )

            for row in reader:
                fname = row[file_key].strip()
                roll = row[roll_key].strip().upper()
                name = row[name_key].strip()
                course = row.get(course_key, "").strip() if course_key else ""

                if fname and roll:
                    # Index by both exact filename and lowercase basename
                    mapping[fname] = {"roll_number": roll, "name": name, "course": course}
                    mapping[fname.lower()] = {"roll_number": roll, "name": name, "course": course}
                    mapping[os.path.basename(fname).lower()] = {"roll_number": roll, "name": name, "course": course}

        return mapping

    def run_import(
        self,
        photos_folder: str,
        csv_mapping_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> ImportSummary:
        """
        Process all supported photographs in photos_folder according to validation rules.
        """
        self._cancel_requested = False
        summary = ImportSummary()

        if not os.path.isdir(photos_folder):
            raise FileNotFoundError(f"Photos directory not found: {photos_folder}")

        # Load CSV mapping if provided
        csv_map = {}
        if csv_mapping_path and os.path.isfile(csv_mapping_path):
            try:
                csv_map = self.parse_csv_mapping(csv_mapping_path)
                logger.info(f"Loaded {len(csv_map)} student mappings from CSV.")
            except Exception as e:
                logger.error(f"Error parsing CSV mapping: {e}")
                raise

        # Scan for supported images
        photo_files = []
        for root, _, files in os.walk(photos_folder):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    photo_files.append(os.path.join(root, file))

        photo_files.sort()
        summary.total_files = len(photo_files)
        logger.info(f"Starting bulk import for {summary.total_files} photo files.")

        for idx, file_path in enumerate(photo_files):
            if self._cancel_requested:
                logger.warning("Bulk import cancelled by user request.")
                break

            filename = os.path.basename(file_path)
            if progress_callback:
                progress_callback(idx + 1, summary.total_files, filename)

            # 1. Determine student identity mapping
            roll_number = ""
            student_name = ""
            course = ""

            # Check CSV mapping first
            mapping_info = csv_map.get(filename) or csv_map.get(filename.lower())
            if mapping_info:
                roll_number = mapping_info["roll_number"]
                student_name = mapping_info["name"]
                course = mapping_info.get("course", "")
            else:
                # Fallback: check if filename itself is a confident roll number
                # Exclude generic camera and export prefixes
                stem = Path(filename).stem.strip()
                generic_prefixes = ("img_", "dsc_", "photo_", "pic_", "dcim_", "student_", "image_", "frame_", "cam_", "scan_")
                is_generic = any(stem.lower().startswith(p) for p in generic_prefixes) or "final" in stem.lower() or "copy" in stem.lower()
                
                # Confident roll numbers: e.g. 24BT001, 101, CS101, BT2024_01
                is_valid_pattern = bool(re.match(r"^(?:\d{2}[A-Z]{2,4}\d{2,4}|[A-Z]{2,4}\d{2,4}|\d{3,8})$", stem, re.IGNORECASE))

                if not is_generic and is_valid_pattern:
                    roll_number = stem.upper()
                    student_name = f"Student {roll_number}"
                else:
                    # Unreliable filename and no CSV mapping -> DO NOT GUESS!
                    item_res = ImportItemResult(
                        filename=filename,
                        roll_number="UNKNOWN",
                        student_name="UNKNOWN",
                        status="NEEDS_REVIEW",
                        category="MANUAL_REVIEW",
                        message=f"Manual Review Required: Ambiguous filename '{filename}' without CSV mapping. Identity not guessed."
                    )
                    summary.needs_review += 1
                    summary.items.append(item_res)
                    continue

            # 2. Image Quality Validation: Attempt to open
            try:
                img_bgr = cv2.imread(file_path)
                if img_bgr is None or img_bgr.size == 0:
                    item_res = ImportItemResult(
                        filename=filename,
                        roll_number=roll_number,
                        student_name=student_name,
                        status="FAILED",
                        category="FAILURE",
                        message="Corrupt or unreadable image file."
                    )
                    summary.failed += 1
                    summary.items.append(item_res)
                    continue
            except Exception as e:
                item_res = ImportItemResult(
                    filename=filename,
                    roll_number=roll_number,
                    student_name=student_name,
                    status="FAILED",
                    category="FAILURE",
                    message=f"Failed to read image file: {e}"
                )
                summary.failed += 1
                summary.items.append(item_res)
                continue

            h, w = img_bgr.shape[:2]
            if w < 50 or h < 50:
                item_res = ImportItemResult(
                    filename=filename,
                    roll_number=roll_number,
                    student_name=student_name,
                    status="NEEDS_REVIEW",
                    category="MANUAL_REVIEW",
                    message=f"Manual Review Required: Extremely low resolution ({w}x{h})."
                )
                summary.needs_review += 1
                summary.items.append(item_res)
                continue

            # 3. Face Detection
            faces = self.detector.detect(img_bgr)
            face_count = len(faces)

            if face_count == 0:
                item_res = ImportItemResult(
                    filename=filename,
                    roll_number=roll_number,
                    student_name=student_name,
                    status="NEEDS_REVIEW",
                    category="MANUAL_REVIEW",
                    message="Manual Review Required: No face detected in photograph."
                )
                summary.needs_review += 1
                summary.items.append(item_res)
                continue

            if face_count > 1:
                item_res = ImportItemResult(
                    filename=filename,
                    roll_number=roll_number,
                    student_name=student_name,
                    status="NEEDS_REVIEW",
                    category="MANUAL_REVIEW",
                    message=f"Manual Review Required: Multiple faces detected ({face_count})."
                )
                summary.needs_review += 1
                summary.items.append(item_res)
                continue

            target_face = faces[0]

            # 4. Face Encoding
            try:
                embedding = self.encoder.encode_face(img_bgr, target_face)
                encoding_bytes = self.encoder.serialize_embedding(embedding)
            except Exception as e:
                logger.error(f"Encoding error on {filename}: {e}")
                item_res = ImportItemResult(
                    filename=filename,
                    roll_number=roll_number,
                    student_name=student_name,
                    status="FAILED",
                    category="FAILURE",
                    message=f"Feature extraction failed: {e}"
                )
                summary.failed += 1
                summary.items.append(item_res)
                continue

            # 5. Database Integration
            try:
                existing_student = self.db.get_student_by_roll(roll_number)
                if existing_student:
                    # Roll number already exists -> Add as secondary reference image for this student!
                    self.db.add_reference_encoding(
                        student_id=existing_student["id"],
                        face_encoding_bytes=encoding_bytes,
                        source_label=f"import:{filename}"
                    )
                    item_res = ImportItemResult(
                        filename=filename,
                        roll_number=roll_number,
                        student_name=existing_student["name"],
                        status="SUCCESS",
                        category="PROCESSED",
                        message="Added as additional approved reference face encoding."
                    )
                    summary.processed_successfully += 1
                    summary.items.append(item_res)
                else:
                    # Insert new student record
                    self.db.add_student(
                        name=student_name,
                        roll_number=roll_number,
                        face_encoding_bytes=encoding_bytes,
                        course=course,
                        registration_type="BULK_IMPORT"
                    )
                    item_res = ImportItemResult(
                        filename=filename,
                        roll_number=roll_number,
                        student_name=student_name,
                        status="SUCCESS",
                        category="PROCESSED",
                        message="Student registered successfully via bulk import."
                    )
                    summary.processed_successfully += 1
                    summary.items.append(item_res)
            except Exception as e:
                logger.error(f"Database error during import of {roll_number}: {e}")
                item_res = ImportItemResult(
                    filename=filename,
                    roll_number=roll_number,
                    student_name=student_name,
                    status="FAILED",
                    category="FAILURE",
                    message=f"Database write failure: {e}"
                )
                summary.failed += 1
                summary.items.append(item_res)

        logger.info(
            f"Import complete. Total: {summary.total_files}, "
            f"Success: {summary.processed_successfully}, "
            f"Needs Review: {summary.needs_review}, "
            f"Failed: {summary.failed}"
        )
        return summary
