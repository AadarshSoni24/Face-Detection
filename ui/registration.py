"""
ui/registration.py - Student Registration Screen (Webcam & Biometric Enrollment).

Enables administrators to register students beforehand, capturing live face frames,
validating image illumination/quality, generating 128-dimensional deep feature embeddings,
and persisting student records into SQLite.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Callable, Optional
import cv2
import numpy as np
from PIL import Image, ImageTk
import logging
import config
from database.db import Database
from camera.camera_manager import CameraManager, CameraUnavailableError
from recognition.face_detector import FaceDetector, DetectedFace
from recognition.face_encoder import FaceEncoder
from recognition.quality_analyzer import QualityAnalyzer, QualityAssessment
from services.registration_service import RegistrationService, RegistrationResult
from ui.styles import (
    FONT_TITLE, FONT_SUBHEADER, FONT_BODY, FONT_SMALL,
    FONT_HEADER, FONT_BODY_BOLD, FONT_VERDICT
)

logger = logging.getLogger("ExamAuth.RegistrationUI")


class RegistrationView(ttk.Frame):
    """Student Registration Screen with interactive camera feed and real-time detection."""

    def __init__(
        self,
        parent: tk.Widget,
        db: Database,
        camera_manager: CameraManager,
        detector: FaceDetector,
        encoder: FaceEncoder,
        navigate_callback: Callable[[str], None]
    ):
        super().__init__(parent, style="App.TFrame")
        self.db = db
        self.camera_manager = camera_manager
        self.detector = detector
        self.encoder = encoder
        self.navigate = navigate_callback

        self.quality_analyzer = QualityAnalyzer()
        self.registration_service = RegistrationService(
            db=self.db,
            detector=self.detector,
            encoder=self.encoder
        )

        self._feed_active = False
        self._poll_job: Optional[str] = None
        self._current_raw_frame: Optional[np.ndarray] = None
        self._photo_image: Optional[ImageTk.PhotoImage] = None
        self._captured_snapshot_frame: Optional[np.ndarray] = None
        self._captured_snapshot_thumb: Optional[ImageTk.PhotoImage] = None

        self._build_ui()

    def _build_ui(self) -> None:
        """Construct registration interface."""
        # Top Header Bar
        header_frame = tk.Frame(self, bg=config.COLOR_HEADER_BG, height=60)
        header_frame.pack(fill="x", side="top")
        header_frame.pack_propagate(False)

        header_inner = tk.Frame(header_frame, bg=config.COLOR_HEADER_BG)
        header_inner.pack(fill="both", expand=True, padx=20)

        ttk.Button(
            header_inner,
            text="← Back to Dashboard",
            style="Secondary.TButton",
            command=self._on_back_clicked
        ).pack(side="left", pady=12)

        tk.Label(
            header_inner,
            text="Student Biometric Registration",
            font=FONT_HEADER,
            fg="#FFFFFF",
            bg=config.COLOR_HEADER_BG
        ).pack(side="left", padx=20, pady=12)

        # Main Layout: 2 Columns (Left: Form Inputs & Controls, Right: Live Camera)
        main_box = tk.Frame(self, bg=config.COLOR_BG)
        main_box.pack(fill="both", expand=True, padx=24, pady=20)

        # Left Column: Form Panel (Width ~380)
        left_panel = tk.Frame(main_box, bg=config.COLOR_CARD_BG, bd=1, relief="solid", highlightthickness=0)
        left_panel.config(highlightbackground=config.COLOR_BORDER)
        left_panel.pack(side="left", fill="y", padx=(0, 16), ipadx=16, ipady=16)

        tk.Label(
            left_panel,
            text="Student Information",
            font=FONT_SUBHEADER,
            fg=config.COLOR_TEXT_PRIMARY,
            bg=config.COLOR_CARD_BG
        ).pack(anchor="w", padx=16, pady=(16, 12))

        # Field: Full Name
        tk.Label(
            left_panel,
            text="Full Name *",
            font=FONT_BODY_BOLD,
            fg=config.COLOR_TEXT_PRIMARY,
            bg=config.COLOR_CARD_BG
        ).pack(anchor="w", padx=16, pady=(6, 2))

        self.entry_name = ttk.Entry(left_panel, style="App.TEntry", width=32)
        self.entry_name.pack(fill="x", padx=16, pady=(0, 10))

        # Field: Roll Number
        tk.Label(
            left_panel,
            text="Roll Number / Student ID *",
            font=FONT_BODY_BOLD,
            fg=config.COLOR_TEXT_PRIMARY,
            bg=config.COLOR_CARD_BG
        ).pack(anchor="w", padx=16, pady=(6, 2))

        self.entry_roll = ttk.Entry(left_panel, style="App.TEntry", width=32)
        self.entry_roll.pack(fill="x", padx=16, pady=(0, 10))

        # Field: Course / Department
        tk.Label(
            left_panel,
            text="Course / Department (Optional)",
            font=FONT_BODY,
            fg=config.COLOR_TEXT_PRIMARY,
            bg=config.COLOR_CARD_BG
        ).pack(anchor="w", padx=16, pady=(6, 2))

        self.entry_course = ttk.Entry(left_panel, style="App.TEntry", width=32)
        self.entry_course.pack(fill="x", padx=16, pady=(0, 14))

        # Action Buttons: Explicit 2-Step Workflow
        tk.Label(
            left_panel,
            text="Biometric Face Enrollment",
            font=FONT_SUBHEADER,
            fg=config.COLOR_TEXT_PRIMARY,
            bg=config.COLOR_CARD_BG
        ).pack(anchor="w", padx=16, pady=(6, 6))

        # Face Capture Snapshot Preview Box
        self.box_snapshot_preview = tk.Frame(
            left_panel,
            bg="#F1F5F9",
            bd=1,
            relief="solid",
            highlightthickness=0,
            height=90
        )
        self.box_snapshot_preview.config(highlightbackground=config.COLOR_BORDER)
        self.box_snapshot_preview.pack(fill="x", padx=16, pady=(0, 10))
        self.box_snapshot_preview.pack_propagate(False)

        self.lbl_snapshot_img = tk.Label(self.box_snapshot_preview, bg="#F1F5F9")
        self.lbl_snapshot_img.pack(side="left", padx=10, pady=6)

        self.lbl_snapshot_status = tk.Label(
            self.box_snapshot_preview,
            text="No face captured yet.\nPosition face & click 'Capture Face'",
            font=FONT_SMALL,
            fg=config.COLOR_TEXT_MUTED,
            bg="#F1F5F9",
            justify="left"
        )
        self.lbl_snapshot_status.pack(side="left", padx=6, fill="both", expand=True)

        # Step 1: Capture Face Button
        capture_btn_row = tk.Frame(left_panel, bg=config.COLOR_CARD_BG)
        capture_btn_row.pack(fill="x", padx=16, pady=(0, 8))

        self.btn_capture_face = ttk.Button(
            capture_btn_row,
            text="📸 Step 1: Capture Face",
            style="Primary.TButton",
            command=self._on_capture_face_clicked
        )
        self.btn_capture_face.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_retake = ttk.Button(
            capture_btn_row,
            text="🔄 Retake",
            style="Secondary.TButton",
            command=self._on_retake_clicked,
            state="disabled"
        )
        self.btn_retake.pack(side="right", padx=(4, 0))

        # Step 2: Register Student Button (Enabled ONLY when face is captured)
        self.btn_register = ttk.Button(
            left_panel,
            text="💾 Step 2: Register Student",
            style="Success.TButton",
            command=self._on_register_clicked,
            state="disabled"
        )
        self.btn_register.pack(fill="x", padx=16, pady=(0, 6))

        ttk.Button(
            left_panel,
            text="🧹 Clear Fields",
            style="Secondary.TButton",
            command=self._clear_fields
        ).pack(fill="x", padx=16, pady=(0, 10))

        ttk.Separator(left_panel, orient="horizontal").pack(fill="x", padx=16, pady=4)

        # Camera Device Selector & Source Control
        cam_sel_row = tk.Frame(left_panel, bg=config.COLOR_CARD_BG)
        cam_sel_row.pack(fill="x", padx=16, pady=(2, 4))

        tk.Label(cam_sel_row, text="Camera:", font=FONT_SMALL, fg=config.COLOR_TEXT_MUTED, bg=config.COLOR_CARD_BG).pack(side="left")
        self.combo_cam = ttk.Combobox(cam_sel_row, values=["Camera 0", "Camera 1", "Camera 2"], state="readonly", width=12)
        self.combo_cam.set(f"Camera {config.DEFAULT_CAMERA_INDEX}")
        self.combo_cam.pack(side="right")
        self.combo_cam.bind("<<ComboboxSelected>>", self._on_camera_selected)

        cam_btn_row = tk.Frame(left_panel, bg=config.COLOR_CARD_BG)
        cam_btn_row.pack(fill="x", padx=16, pady=4)

        self.btn_toggle_cam = ttk.Button(
            cam_btn_row,
            text="▶ Start Camera",
            style="Primary.TButton",
            command=self._toggle_camera
        )
        self.btn_toggle_cam.pack(side="left", fill="x", expand=True, padx=(0, 4))

        ttk.Button(
            cam_btn_row,
            text="📁 Browse Photo...",
            style="Secondary.TButton",
            command=self._browse_photo_file
        ).pack(side="right", fill="x", expand=True, padx=(4, 0))

        # Status Message Box in Left Panel
        self.status_box = tk.Label(
            left_panel,
            text="Ready. Start camera or load photo.",
            font=FONT_SMALL,
            fg=config.COLOR_TEXT_MUTED,
            bg="#F1F5F9",
            wraplength=280,
            justify="center",
            padx=8,
            pady=8
        )
        self.status_box.pack(fill="x", padx=16, pady=(10, 0), side="bottom")

        # Right Column: Camera Preview Box
        right_panel = tk.Frame(main_box, bg=config.COLOR_CARD_BG, bd=1, relief="solid")
        right_panel.config(highlightbackground=config.COLOR_BORDER)
        right_panel.pack(side="right", fill="both", expand=True)

        preview_header = tk.Frame(right_panel, bg=config.COLOR_CARD_BG)
        preview_header.pack(fill="x", padx=16, pady=12)

        tk.Label(
            preview_header,
            text="Live Camera Preview & Real-time Quality Guidance",
            font=FONT_SUBHEADER,
            fg=config.COLOR_TEXT_PRIMARY,
            bg=config.COLOR_CARD_BG
        ).pack(side="left")

        # Badges (Face Detection + Image Quality)
        badges_frame = tk.Frame(preview_header, bg=config.COLOR_CARD_BG)
        badges_frame.pack(side="right")

        self.lbl_quality_badge = tk.Label(
            badges_frame,
            text="Quality Check",
            font=FONT_SMALL,
            fg="#64748B",
            bg="#E2E8F0",
            padx=8,
            pady=2
        )
        self.lbl_quality_badge.pack(side="left", padx=(0, 8))

        self.lbl_detection_badge = tk.Label(
            badges_frame,
            text="Camera Inactive",
            font=FONT_SMALL,
            fg="#64748B",
            bg="#E2E8F0",
            padx=8,
            pady=2
        )
        self.lbl_detection_badge.pack(side="left")

        # Canvas for video display (640x480 native aspect)
        self.canvas_cam = tk.Canvas(
            right_panel,
            width=config.CAMERA_FRAME_WIDTH,
            height=config.CAMERA_FRAME_HEIGHT,
            bg="#0F172A",
            highlightthickness=0
        )
        self.canvas_cam.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        # Initial canvas message
        self.canvas_cam.create_text(
            320, 240,
            text="Camera feed is stopped.\nClick 'Start Camera' or 'Browse Photo...'",
            fill="#94A3B8",
            font=FONT_BODY,
            justify="center",
            tags="placeholder"
        )

    def on_show(self) -> None:
        """Called when this screen becomes visible."""
        self._clear_fields()
        self._start_camera_stream()

    def on_hide(self) -> None:
        """Called when navigating away from this screen."""
        self._stop_camera_stream()

    def _on_back_clicked(self) -> None:
        """Handle back button navigation."""
        self._stop_camera_stream()
        self.navigate("dashboard")

    def _on_camera_selected(self, event=None) -> None:
        """Handle camera dropdown index change."""
        sel = self.combo_cam.get()
        idx = int(sel.split()[-1])
        if self._feed_active:
            self._stop_camera_stream()
            self.camera_manager.switch_camera(idx)
            self._start_camera_stream()
        else:
            self.camera_manager.camera_index = idx

    def _start_camera_stream(self) -> None:
        """Start hardware webcam capture and UI polling loop."""
        if self._feed_active:
            return

        try:
            self.camera_manager.start()
            self._feed_active = True
            self.btn_toggle_cam.config(text="⏹ Stop Camera", style="Secondary.TButton")
            self.status_box.config(
                text="Camera active. Position face in front of the lens.",
                fg=config.COLOR_PRIMARY,
                bg="#EFF6FF"
            )
            self._poll_frame()
        except CameraUnavailableError:
            self._feed_active = False
            self.btn_toggle_cam.config(text="▶ Start Camera", style="Primary.TButton")
            self.status_box.config(
                text="Camera could not be accessed.\nYou can use 'Browse Photo...' to test.",
                fg=config.COLOR_DANGER,
                bg=config.COLOR_DANGER_BG
            )
            self.lbl_detection_badge.config(text="Camera Unavailable", fg="#DC2626", bg="#FEF2F2")

    def _stop_camera_stream(self) -> None:
        """Stop webcam capture and clear polling."""
        if self._poll_job:
            self.after_cancel(self._poll_job)
            self._poll_job = None

        self._feed_active = False
        self.camera_manager.stop()
        self.btn_toggle_cam.config(text="▶ Start Camera", style="Primary.TButton")
        self.lbl_detection_badge.config(text="Camera Inactive", fg="#64748B", bg="#E2E8F0")

    def _toggle_camera(self) -> None:
        """Toggle camera on/off."""
        if self._feed_active:
            self._stop_camera_stream()
            self.canvas_cam.delete("all")
            self.canvas_cam.create_text(
                320, 240,
                text="Camera feed stopped.\nClick 'Start Camera' or 'Browse Photo...'",
                fill="#94A3B8",
                font=FONT_BODY,
                justify="center",
                tags="placeholder"
            )
        else:
            self._start_camera_stream()

    def _browse_photo_file(self) -> None:
        """Load a local image file as virtual camera input."""
        file_path = filedialog.askopenfilename(
            parent=self,
            title="Select Student Photograph",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        self._stop_camera_stream()
        try:
            self.camera_manager.start_virtual_mode(file_path)
            self._feed_active = True
            self.btn_toggle_cam.config(text="⏹ Stop Camera", style="Secondary.TButton")
            self.status_box.config(
                text=f"Loaded image: {file_path.split('/')[-1].split('\\')[-1]}",
                fg=config.COLOR_SUCCESS,
                bg=config.COLOR_SUCCESS_BG
            )
            self._poll_frame()
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not load image: {e}", parent=self)

    def _poll_frame(self) -> None:
        """Poll latest frame, evaluate quality, and render overlays."""
        if not self._feed_active:
            return

        frame = self.camera_manager.read_frame()
        if frame is not None:
            self._current_raw_frame = frame.copy()

            # Detect faces on frame
            faces = self.detector.detect(frame)
            face_count = len(faces)

            # Update detection badge
            if face_count == 0:
                self.lbl_detection_badge.config(text="No Face Detected", fg="#D97706", bg="#FEF3C7")
                self.lbl_quality_badge.config(text="Awaiting Face", fg="#64748B", bg="#E2E8F0")
            elif face_count == 1:
                self.lbl_detection_badge.config(
                    text=f"1 Face ({faces[0].confidence*100:.1f}%)",
                    fg="#059669",
                    bg="#D1FAE5"
                )
                # Quality check on detected face
                q: QualityAssessment = self.quality_analyzer.evaluate_frame(
                    frame,
                    face_bbox=(faces[0].x, faces[0].y, faces[0].w, faces[0].h)
                )
                self.lbl_quality_badge.config(
                    text=q.message,
                    fg=q.badge_color,
                    bg="#ECFDF5" if q.is_acceptable else "#FEF3C7"
                )
            else:
                self.lbl_detection_badge.config(
                    text=f"⚠️ {face_count} Faces Detected",
                    fg="#DC2626",
                    bg="#FEE2E2"
                )
                self.lbl_quality_badge.config(text="Multiple Faces", fg="#DC2626", bg="#FEE2E2")

            # Draw visual landmarks on frame
            annotated_frame = self.detector.draw_faces(frame, faces)

            # Render to Tkinter canvas
            self._render_frame_to_canvas(annotated_frame)

        # Schedule next frame update
        self._poll_job = self.after(33, self._poll_frame)

    def _render_frame_to_canvas(self, frame_bgr: np.ndarray) -> None:
        """Convert BGR frame to PhotoImage and update canvas."""
        canvas_w = self.canvas_cam.winfo_width() or config.CAMERA_FRAME_WIDTH
        canvas_h = self.canvas_cam.winfo_height() or config.CAMERA_FRAME_HEIGHT

        rgb_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        pil_img.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)
        self._photo_image = ImageTk.PhotoImage(pil_img)

        self.canvas_cam.delete("all")
        x = max(0, (canvas_w - self._photo_image.width()) // 2)
        y = max(0, (canvas_h - self._photo_image.height()) // 2)
        self.canvas_cam.create_image(x, y, anchor="nw", image=self._photo_image)

    def _on_capture_face_clicked(self) -> None:
        """Step 1: Capture snapshot and validate detection & quality."""
        if self._current_raw_frame is None:
            messagebox.showerror(
                "Camera Inactive",
                "No live camera feed detected.\nPlease click '▶ Start Camera' or '📁 Browse Photo...' first.",
                parent=self
            )
            return

        frame = self._current_raw_frame.copy()
        faces = self.detector.detect(frame)
        face_count = len(faces)

        if face_count == 0:
            self.lbl_snapshot_status.config(
                text="❌ No face detected!\nPlease face the camera directly.",
                fg=config.COLOR_DANGER
            )
            messagebox.showwarning("No Face", "No face detected. Please look directly into the camera.", parent=self)
            return

        if face_count > 1:
            self.lbl_snapshot_status.config(
                text=f"❌ Multiple faces detected ({face_count})!\nOnly 1 person allowed.",
                fg=config.COLOR_DANGER
            )
            messagebox.showwarning("Multiple Faces", f"Multiple faces ({face_count}) detected.", parent=self)
            return

        target_face = faces[0]
        q = self.quality_analyzer.evaluate_frame(frame, face_bbox=(target_face.x, target_face.y, target_face.w, target_face.h))

        self._captured_snapshot_frame = frame

        # Crop face thumbnail
        h_img, w_img = frame.shape[:2]
        pad = int(max(target_face.w, target_face.h) * 0.25)
        x1 = max(0, target_face.x - pad)
        y1 = max(0, target_face.y - pad)
        x2 = min(w_img, target_face.x + target_face.w + pad)
        y2 = min(h_img, target_face.y + target_face.h + pad)
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size > 0:
            rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(rgb_crop).resize((70, 70), Image.Resampling.LANCZOS)
            self._captured_snapshot_thumb = ImageTk.PhotoImage(pil_crop)
            self.lbl_snapshot_img.config(image=self._captured_snapshot_thumb)

        self.lbl_snapshot_status.config(
            text=f"✓ Face Captured & Validated\n{q.message}\nReady to Register!",
            fg=config.COLOR_SUCCESS,
            font=FONT_BODY_BOLD
        )

        self.btn_register.config(state="normal")
        self.btn_retake.config(state="normal")
        self.status_box.config(
            text="✓ Face captured! Now enter Name & Roll Number, then click 'Step 2: Register Student'.",
            fg=config.COLOR_SUCCESS,
            bg=config.COLOR_SUCCESS_BG
        )

    def _on_retake_clicked(self) -> None:
        """Reset captured snapshot."""
        self._captured_snapshot_frame = None
        self._captured_snapshot_thumb = None
        self.lbl_snapshot_img.config(image="")
        self.lbl_snapshot_status.config(
            text="No face captured yet.\nPosition face & click 'Capture Face'",
            fg=config.COLOR_TEXT_MUTED,
            font=FONT_SMALL
        )
        self.btn_register.config(state="disabled")
        self.btn_retake.config(state="disabled")
        self.status_box.config(
            text="Snapshot cleared. Click 'Step 1: Capture Face' when ready.",
            fg=config.COLOR_TEXT_MUTED,
            bg="#F1F5F9"
        )

    def _on_register_clicked(self) -> None:
        """Step 2: Commit registered student."""
        name = self.entry_name.get().strip()
        roll = self.entry_roll.get().strip()
        course = self.entry_course.get().strip()

        if not name:
            messagebox.showwarning("Validation Error", "Student Name is required.", parent=self)
            self.entry_name.focus()
            return
        if not roll:
            messagebox.showwarning("Validation Error", "Roll Number is required.", parent=self)
            self.entry_roll.focus()
            return

        if self._captured_snapshot_frame is None:
            messagebox.showwarning("Face Required", "Please click 'Step 1: Capture Face' first!", parent=self)
            return

        result: RegistrationResult = self.registration_service.register_student(
            name=name,
            roll_number=roll,
            frame_bgr=self._captured_snapshot_frame,
            course=course
        )

        if result.success:
            self.status_box.config(
                text=f"✓ {result.message}\nStudent: {name} ({roll.upper()})",
                fg=config.COLOR_SUCCESS,
                bg=config.COLOR_SUCCESS_BG
            )
            messagebox.showinfo(
                "Registration Successful",
                f"{result.message}\n\nName: {name}\nRoll Number: {roll.upper()}\nBiometric template stored in SQLite.",
                parent=self
            )
            self._clear_fields()
        else:
            self.status_box.config(
                text=f"✗ {result.message}",
                fg=config.COLOR_DANGER,
                bg=config.COLOR_DANGER_BG
            )
            messagebox.showerror("Registration Failed", result.message, parent=self)

    def _clear_fields(self) -> None:
        """Reset form fields."""
        self.entry_name.delete(0, "end")
        self.entry_roll.delete(0, "end")
        self.entry_course.delete(0, "end")
        self._on_retake_clicked()
        self.entry_name.focus()
