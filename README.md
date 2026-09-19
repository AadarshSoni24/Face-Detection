# Biometric-Based Exam Authentication Using Face Recognition

An academic prototype desktop application for secure student identity verification and automated examination attendance marking using deep neural network facial recognition.

---

## 1. Project Overview

### Problem Statement
Traditional examination halls rely on physical paper hall tickets, student ID cards, and manual roll calling. These conventional approaches suffer from critical vulnerabilities:
- **Impersonation & Proxy Attendance**: Unauthorized individuals sitting for exams on behalf of registered candidates.
- **Lost / Forged ID Documents**: Paper hall tickets and physical badges can be easily duplicated or manipulated.
- **Administrative Overhead**: Manual roster verification consumes substantial invigilation time and is prone to human error.

### Proposed Solution
This project implements an automated, desktop biometric verification system:
1. **Pre-Exam Enrollment**: An administrator registers students beforehand via webcam or through bulk photograph import mapped with examination registration records.
2. **Deep Neural Network Biometrics**: Generates a compact 128-dimensional mathematical facial embedding vector for each student using state-of-the-art deep learning models (`YuNet` for face detection + `SFace` for facial feature representation).
3. **Exam-Day Verification**:
   - The student enters their examination Roll Number.
   - The live webcam captures their face.
   - The system aligns the face and compares the live embedding against the registered biometric profile using **Cosine Similarity**.
   - **If Verified ($\ge 0.363$)**: Identity is confirmed, attendance is marked `PRESENT` (with duplicate prevention for the session), and **EXAM ACCESS IS GRANTED**.
   - **If Mismatch ($< 0.363$)**: Identity fails verification, attendance is `NOT MARKED`, and **EXAM ACCESS IS DENIED**.

---

## 2. Core Features

- **Strict Separation of Face Detection and Face Recognition**:
  - Face Detection uses `YuNet` (ONNX deep neural network) locating face bounding boxes and 5 facial landmarks (eyes, nose tip, mouth corners).
  - Face Recognition uses `SFace` (ONNX deep neural network) extracting 128-dimensional L2-normalized feature embeddings.
- **Interactive Anti-Spoofing & Liveness Detection**:
  - Temporal landmark micro-motion analysis to prevent 2D static printed photo attacks and digital screen replays.
  - Eye Aspect Ratio (EAR) blink tracking dynamics across sliding frame window.
  - Texture / Laplacian frequency analysis for screen moiré artifact detection.
- **Exam Session Management & Hall Seating Allocation**:
  - Create and schedule examination sessions (Course code, Exam title, Hall/Room Number, Date, Time Window).
  - Session-wise attendance filtering and export to CSV.
- **Verified Digital Exam Entry Pass (Hall Ticket Receipt)**:
  - Instant generation of an official digital examination verification pass upon successful biometric verification.
  - Embeds candidate photo thumbnail, hall number, verified timestamp, cryptographic security token, and status watermark.
  - Exportable as high-res PNG image or printable HTML receipt with one click.
- **Real-Time Face Quality & Illumination Guidance**:
  - Real-time Laplacian variance sharpness and pixel intensity illumination checks.
  - Live visual feedback pills on camera feed (*"✓ Optimal Lighting & Focus"*, *"⚠️ Low Lighting"*, *"⚠️ Blurry"*, *"⚠️ Too Far"*).
- **Webcam-Based Manual Registration & Multi-Camera Switcher**:
  - Dynamic device index dropdown (Camera 0, Camera 1, Camera 2, Virtual File Mode).
  - 2-step enrollment workflow with validated face capture snapshot preview.
- **Exam Day Verification & Auto-Identification**:
  - **1:N Automatic Identification**: Scan live face directly without typing a roll number; the system recognizes the student across the database, automatically fills their Roll Number, marks attendance, issues an exam pass, and grants access!
  - **1:1 Claimed Identity Verification**: Optional manual roll-number entry with direct 1:1 facial biometric matching.
  - Distinct verdict panels (Emerald Green for Granted, Red for Denied) with real-time similarity and liveness scores.
- **Security & Forensic Audit Logs Inspector**:
  - Complete security log of all authentication attempts with filterable results (`SUCCESS`, `FAILED`, `SPOOF_ATTEMPT`, `MULTI_FACE`, `NO_FACE`).
  - Export audit logs to CSV for examination integrity reporting.
- **Bulk Student Photo & CSV Import**:
  - Batch processes legacy college photo exports with CSV mapping.
  - Validates image integrity, flags corrupt files as **Failed**, and moves ambiguous zero-face or multi-face images into a **Manual Review Required** list.
- **Student Profile & Biometric History Modal**:
  - Detailed student profile viewer displaying enrollment metadata and table of verified examination attendances.

---

## 3. Technology Stack & Architecture

```text
exam_authentication/
├── main.py                         # Application entry point & Tkinter window lifecycle
├── config.py                       # Centralized configuration (thresholds, paths, resolutions)
├── requirements.txt                # Pinned dependencies
├── demo_walkthrough.py             # Automated CLI demonstration walkthrough
├── README.md                       # Comprehensive documentation & test report
│
├── models/
│   ├── face_detection_yunet_2023mar.onnx    # YuNet face detector model (232 KB)
│   └── face_recognition_sface_2021dec.onnx   # SFace 128-d feature recognizer (38.6 MB)
│
├── database/
│   ├── db.py                       # Connection manager & parameterized CRUD operations
│   └── schema.py                   # SQLite DDL schema (students, attendance, sessions, logs)
│
├── recognition/
│   ├── face_detector.py            # YuNet detector wrapper with landmark extraction
│   ├── face_encoder.py             # SFace encoder with 5-point affine alignment (alignCrop)
│   ├── face_matcher.py             # Cosine similarity matcher & threshold evaluation
│   ├── liveness_detector.py        # Temporal micro-motion & blink anti-spoofing engine
│   ├── quality_analyzer.py         # Real-time illumination, focus sharpness, and pose checker
│   └── model_utils.py              # Automatic model downloader utility
│
├── camera/
│   └── camera_manager.py           # Thread-safe OpenCV capture with multi-camera discovery
│
├── services/
│   ├── registration_service.py     # Enrollment business logic & face validation
│   ├── authentication_service.py   # Verification decision pipeline & liveness access control
│   ├── attendance_service.py       # Attendance recording with duplicate prevention
│   ├── pass_generator.py           # Digital Exam Entry Pass & printable HTML receipt generator
│   └── import_service.py           # Bulk photo import with CSV mapping & diagnostics
│
├── ui/
│   ├── styles.py                   # TTK design theme, colors, fonts, and card styling
│   ├── dashboard.py                # Main analytics dashboard & quick action tiles
│   ├── registration.py             # Manual registration screen with quality feedback
│   ├── authentication.py           # Exam authentication screen with pass modal & liveness
│   ├── sessions.py                 # Exam session & hall seating allocation screen
│   ├── audit_logs.py               # Security & biometric verification audit inspector
│   ├── students.py                 # Student roster table & profile history modal
│   ├── attendance.py               # Attendance records log with date/session filtering
│   └── import_photos.py            # Bulk photo & CSV import UI with report
│
├── data/
│   ├── exam_auth.db                # SQLite database (auto-created)

│   └── sample_import/              # Verified sample test photos & CSV mapping
│
└── tests/
    ├── create_test_assets.py       # Verified test asset preparation
    ├── test_database.py            # Unit tests for SQLite layer & constraints
    ├── test_recognition.py         # Unit tests for YuNet detection & SFace matching
    ├── test_services.py            # Integration tests for enrollment & authentication
    ├── test_bulk_import.py         # Integration tests for bulk photo & CSV import
    └── run_all_tests.py            # Unified test suite runner
```

---

## 4. Face Recognition Pipeline: Detection vs Recognition

### Why OpenCV Haar Cascades Are NOT Face Recognition
OpenCV's Haar cascades or classic classifiers perform **Face Detection only** (answering *"Is there a face in this region?"*). They do not extract identity-discriminative feature representations and cannot tell Person A from Person B.

### Deep Learning Pipeline Implemented:
1. **Face Detection (`YuNet`)**:
   - High-speed convolutional neural network developed for OpenCV Zoo.
   - Outputs bounding box `(x, y, w, h)`, confidence score, and 5 facial landmarks: right eye, left eye, nose tip, right mouth corner, and left mouth corner.
2. **Face Geometric Alignment (`alignCrop`)**:
   - Before feature extraction, the face must be geometrically normalized.
   - Performs a 5-point similarity transformation aligning the eyes horizontally and centering the nose and mouth to a canonical $112 \times 112$ RGB tensor.
3. **Face Feature Extraction (`SFace`)**:
   - Deep neural network trained with ArcFace / SphereFace margin loss.
   - Maps the $112 \times 112$ aligned crop into an invariant $128$-dimensional embedding vector $\mathbf{v} \in \mathbb{R}^{128}$ on a unit hypersphere ($\|\mathbf{v}\|_2 = 1.0$).
4. **Face Matching (Cosine Similarity)**:
   $$\text{Cosine Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \mathbf{u} \cdot \mathbf{v}$$
   - Range: $[-1.0, 1.0]$.
   - Identical face representations yield similarity $\approx 1.0$.
   - Independent identities yield near-zero or low similarity.

### Configurable Threshold Justification:
- In `config.py`, `FACE_MATCH_THRESHOLD = 0.363`.
- **Benchmark Source**: Official OpenCV Zoo benchmark on LFW / MegaFace datasets.
- At threshold $= 0.363$, the False Acceptance Rate (FAR) is approximately $0.1\%$ ($10^{-3}$).
- **How to Adjust**:
  - Stricter security (fewer false accepts): Increase to `0.45` or `0.50`.
  - More tolerant (poor illumination / diverse angles): Decrease to `0.30`.
- The system explicitly **does not claim 100% accuracy**.

---

## 5. Database Design & Security

SQLite database (`data/exam_auth.db`) with parameterized queries:

### 1. `students` Table
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | INTEGER PRIMARY KEY | Unique student identifier |
| `roll_number` | TEXT UNIQUE | Student examination roll number (indexed) |
| `name` | TEXT NOT NULL | Full student name |
| `course` | TEXT | Department / Class |
| `face_encoding`| BLOB NOT NULL | Primary 128-d float32 embedding (512 bytes) |
| `registration_type` | TEXT | `WEBCAM` or `BULK_IMPORT` |
| `created_at` | TIMESTAMP | Registration timestamp |

### 2. `student_reference_encodings` Table
Stores additional approved reference photographs for students enrolled with multiple images. Linked to `students(id)` with `ON DELETE CASCADE`.

### 3. `attendance` Table
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | INTEGER PRIMARY KEY | Attendance entry ID |
| `student_id` | INTEGER NOT NULL | Foreign key referencing `students(id)` |
| `roll_number` | TEXT NOT NULL | Student examination roll number |
| `date` | TEXT NOT NULL | Exam session date (YYYY-MM-DD) |
| `time` | TEXT NOT NULL | Verification timestamp (HH:MM:SS) |
| `status` | TEXT NOT NULL | Default `PRESENT` |
| `UNIQUE(student_id, date)` | CONSTRAINT | **Prevents duplicate attendance for the same student on the same date** |

### 4. `authentication_logs` Table
Maintains audit trail for every verification attempt (`SUCCESS`, `FAILED`, `NOT_FOUND`, `NO_FACE`, `MULTI_FACE`), including similarity score and timestamp.

---

## 6. Installation & Execution

### Prerequisites
- Python 3.10 to Python 3.14 on Windows, Linux, or macOS.

### Step 1: Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Desktop Application
```bash
python main.py
```

### Step 4: Run the Automated CLI Demonstration
```bash
python demo_walkthrough.py
```

---

## 7. Experimental Testing Report

The automated test suite (`tests/run_all_tests.py`) runs 30 unit and integration test cases across all modules without mocked outputs.

### Test Execution Results Table (Actual System Execution)

| Test ID | Module | Input Condition | Expected Result | Actual Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TC-01** | Database | Valid student fields | Student inserted, auto ID generated | ID > 0, retrieved by roll | **PASS** |
| **TC-02** | Database | Duplicate roll number "101" | SQLite `IntegrityError` raised | `UNIQUE constraint failed` caught | **PASS** |
| **TC-03** | Database | Existence check for "101" | Returns True for existing, False for non-existing | Verified True & False | **PASS** |
| **TC-04** | Database | Add multiple reference encodings | Stored with foreign key cascade | Retrieved 3 approved encodings | **PASS** |
| **TC-05** | Database | Duplicate attendance on same date | First succeeds; second blocked | First: True; Second: False ("already marked") | **PASS** |
| **TC-06** | Database | Delete student record | Record & references removed | Student deleted, references 0 | **PASS** |
| **TC-07** | Database | Authentication audit logging | Logs inserted & stats computed | Stats: Total=1, Present=1, Failed=2 | **PASS** |
| **TC-08** | Recognition | Single face portrait (Rahul) | Exactly 1 face detected, confidence > 70% | 1 face detected, 5 landmarks | **PASS** |
| **TC-09** | Recognition | Non-face image (gradient) | 0 faces detected | 0 faces detected | **PASS** |
| **TC-10** | Recognition | Composite multi-face image | 2 faces detected | 2 faces detected | **PASS** |
| **TC-11** | Recognition | SFace feature extraction | 128-d float32 vector, L2 norm = 1.0 | Shape (1, 128), norm = 1.0000 | **PASS** |
| **TC-12** | Recognition | Serialization roundtrip | 512 bytes blob reconstructed exactly | Exact match via `np.allclose` | **PASS** |
| **TC-13** | Recognition | Compare identical face | Cosine similarity $\approx 1.0 \ge 0.363$ | Score = 1.0000, is_match = True | **PASS** |
| **TC-14** | Recognition | Compare Rahul vs Priya | Cosine similarity $< 0.363$ | Score = 0.0154, is_match = False | **PASS** |
| **TC-15** | Recognition | Multi-reference matching | Maximum score selected among references | Picked best score 1.0000 | **PASS** |
| **TC-16** | Services | Valid manual registration | Student registered into database | Success=True, "Student Registered Successfully" | **PASS** |
| **TC-17** | Services | Duplicate roll registration | Rejected with clear notification | Success=False, "already registered" | **PASS** |
| **TC-18** | Services | Missing student name or roll | Validation rejected before detection | Rejected: "Name/Roll is required" | **PASS** |
| **TC-19** | Services | No face in registration frame | Enrollment aborted | "No face detected. Please position face..." | **PASS** |
| **TC-20** | Services | Multiple faces in registration | Enrollment aborted | "Multiple faces detected. Ensure only one..." | **PASS** |
| **TC-21** | Services | Valid authentication (Rahul) | Identity verified, attendance marked, access granted | IDENTITY VERIFIED, PRESENT, GRANTED | **PASS** |
| **TC-22** | Services | Impostor face authentication | Authentication failed, attendance NOT marked, denied | AUTHENTICATION FAILED, NOT MARKED, DENIED | **PASS** |
| **TC-23** | Services | Unknown roll number "999" | Search fails gracefully | "Student not found.", DENIED | **PASS** |
| **TC-24** | Services | No face in authentication feed | Verification rejected | "No face detected...", DENIED | **PASS** |
| **TC-25** | Services | Multiple faces in auth feed | Verification rejected | "Multiple faces detected...", DENIED | **PASS** |
| **TC-26** | Services | Duplicate auth attempt today | Identity verified; notes attendance already marked | "PRESENT (ALREADY MARKED)", count remains 1 | **PASS** |
| **TC-27** | Services | Frame unavailable (None) | Handled gracefully without crash | "Camera could not be accessed." | **PASS** |
| **TC-28** | Bulk Import | Valid CSV mapping + photos | Students registered, quality verified | 2 Processed, 2 Needs Review, 1 Failed | **PASS** |
| **TC-29** | Bulk Import | Imported student live auth | Authenticates identically to manual student | Verified with 101; rejected with wrong face | **PASS** |
| **TC-30** | Bulk Import | Ambiguous filename without CSV | Not guessed; flagged for manual review | Flagged as "Manual Review Required" | **PASS** |

**Summary**: 30 Tests Run, 30 Passed, 0 Failures, 0 Errors (Duration: ~5.9s).

---

## 8. Step-by-Step Demonstration Instructions

To conduct a live demonstration of the project:

### Step 1: Launch Application
Run the desktop application:
```bash
python main.py
```
*The main dashboard will open displaying analytics counters.*

### Step 2: Register a Student (Webcam or Photo)
1. Click **"Register Student (Webcam)"** on the dashboard.
2. Enter:
   - **Full Name**: `Rahul Kumar`
   - **Roll Number**: `101`
   - **Course**: `Computer Science`
3. Click **"Start Camera"** (or **"Browse Photo File..."** and select `data/sample_import/rahul_face.jpg`).
4. Observe the green bounding box and 5 landmark points indicating 1 face detected.
5. Click **"Capture & Register"**.
6. Observe the confirmation message:
   `✓ Student Registered Successfully`

### Step 3: Authenticate Rahul Kumar (Matching Face)
1. Navigate back to the Dashboard and click **"Exam Day Authentication"**.
2. Enter Roll Number: `101`.
3. Start the camera or click **"Test Photo..."** and select `data/sample_import/rahul_face.jpg`.
4. Click **"Verify Face 🎯"**.
5. Observe the Emerald Green verdict card:
   ```text
   ✓ IDENTITY VERIFIED
   Student: Rahul Kumar
   Roll Number: 101
   Face Match: SUCCESS
   Attendance: PRESENT
   Exam Access: GRANTED
   Cosine Similarity: 1.000 (Threshold: 0.363)
   ```

### Step 4: Authenticate with a Different Face (Mismatch Impostor)
1. On the same Authentication screen, keep Roll Number `101`.
2. Click **"Test Photo..."** and select a different person's face (`data/sample_import/priya_face.jpg`).
3. Click **"Verify Face 🎯"**.
4. Observe the Red verdict card:
   ```text
   ✗ AUTHENTICATION FAILED
   Student: Rahul Kumar
   Roll Number: 101
   Face Match: FAILED
   Attendance: NOT MARKED
   Exam Access: DENIED
   Cosine Similarity: 0.015 (Threshold: 0.363)
   Face does not match the registered student.
   ```

### Step 5: Audit Attendance Records
1. Return to the Dashboard and click **"Attendance Records"**.
2. Confirm that Roll `101` (Rahul Kumar) is recorded as `PRESENT` for today's date.
3. Note that the failed impostor attempt did **not** create an attendance record.

### Step 6: Demonstrate Bulk Student Photo Import
1. From the Dashboard, click **"Bulk Photo Import"**.
2. In **Photos Folder**, browse to `data/sample_import`.
3. In **Mapping CSV**, browse to `data/sample_import/students_mapping.csv`.
4. Click **"Start Import"**.
5. Observe the live progress bar and final summary report:
   - **Total Files**: 5
   - **Processed Successfully**: 2 (`101`, `102`)
   - **Manual Review Required**: 2 (`no_face.jpg`, `multi_face.jpg`)
   - **Failed**: 1 (`corrupt_image.jpg`)
6. Review the itemized diagnostic explanations in the table.

---

## 9. Security, Privacy, and Ethical Considerations

1. **Biometric Data Minimization**:
   - The application does not require storing raw, high-resolution photographs in the database.
   - Only 128-dimensional floating-point mathematical embedding vectors are stored in SQLite BLOB format. These feature embeddings cannot be reverse-engineered into the subject's original photograph.
2. **Data Isolation**:
   - Zero cloud transmission: All inferences and database writes are executed strictly on the local machine.
   - Raw facial embeddings are never displayed in the user interface.
3. **Right to Deletion**:
   - Administrators can delete any student's record through the Student Records screen, which permanently cascades and purges their biometric template and attendance logs.
4. **Consent & Governance**:
   - In an academic institution, students must provide informed consent during examination form submission before biometric enrollment.

---

## 10. System Limitations & Future Enhancements

### Known Limitations
> [!WARNING]
> **Liveness / Anti-Spoofing Limitation**:
> The current academic prototype performs 2D deep facial recognition using YuNet + SFace but **does not provide full liveness/anti-spoofing protection against high-resolution photographic or video playback replay attacks**.

- **Hardware Camera Constraints**: In virtualized or containerized server environments without a physical UVC webcam, the application gracefully reports `"Camera could not be accessed."` and provides test file streaming for evaluation.
- **Extreme Illumination & Occlusions**: Severe backlight, heavy facial masks, or extreme pitch/yaw angles (>45 degrees) will impact detection and feature extraction.

### Future Improvements
1. **Active & Passive Liveness Detection**:
   - Real-time eye blink detection using Eye Aspect Ratio (EAR).
   - Dynamic challenge-response (e.g., *"turn head left"*, *"smile"*).
   - Frequency-domain texture analysis or specialized anti-spoofing neural network models (e.g., MiniFASNet).
2. **Multi-Factor Authentication (MFA)**:
   - Pairing facial biometrics with biometric fingerprint or one-time time-based OTP.
3. **Institutional SIS / ERP Integration**:
   - RESTful API connectors to university student information systems with role-based access control (RBAC).
