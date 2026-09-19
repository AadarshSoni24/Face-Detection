"""
ui/authentication.py - Exam Day Student Authentication Screen.

Verifies student identity by matching live webcam face against registered 128-d
biometric embeddings, evaluates anti-spoofing liveness, logs audit records,
marks attendance, and issues official verified digital examination passes.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Callable, Optional, Dict, List
import webbrowser
import cv2
import numpy as np
from PIL import Image, ImageTk
import logging
import os
import config
from database.db import Database
from camera.camera_manager import CameraManager, CameraUnavailableError
from recognition.face_detector import FaceDetector
from recognition.face_encoder import FaceEncoder
from recognition.face_matcher import FaceMatcher
from recognition.liveness_detector import LivenessDetector
from recognition.quality_analyzer import QualityAnalyzer
from services.authentication_service import AuthenticationService, AuthenticationResult
from services.attendance_service import AttendanceService
from services.pass_generator import PassGenerator, ExamPass
from ui.styles import (
    FONT_TITLE, FONT_SUBHEADER, FONT_BODY, FONT_SMALL,
    FONT_HEADER, FONT_BODY_BOLD, FONT_VERDICT
)

logger = logging.getLogger("ExamAuth.AuthenticationUI")


class AuthenticationView(ttk.Frame):
    """Exam Day Authentication Screen with prominent result card and liveness validation."""

    def __init__(
        self,
        parent: tk.Widget,
        db: Database,
        camera_manager: CameraManager,
        detector: FaceDetector,
        encoder: FaceEncoder,
        matcher: FaceMatcher,
        navigate_callback: Callable[[str], None]
    ):
        super().__init__(parent, style="App.TFrame")
        self.db = db
        self.camera_manager = camera_manager
        self.detector = detector
        self.encoder = encoder
        self.matcher = matcher
        self.navigate = navigate_callback

        self.liveness_detector = LivenessDetector()
        self.quality_analyzer = QualityAnalyzer()
        self.pass_generator = PassGenerator()
        self.attendance_service = AttendanceService(db=self.db)
        self.auth_service = AuthenticationService(
            db=self.db,
            detector=self.detector,
            encoder=self.encoder,
            matcher=self.matcher,
            attendance_service=self.attendance_service,
            liveness_detector=self.liveness_detector,
            quality_analyzer=self.quality_analyzer,
            pass_generator=self.pass_generator
        )

        self._feed_active = False
        self._poll_job: Optional[str] = None
        self._current_raw_frame: Optional[np.ndarray] = None
        self._photo_image: Optional[ImageTk.PhotoImage] = None
        self._last_result: Optional[AuthenticationResult] = None
        self._pass_preview_img: Optional[ImageTk.PhotoImage] = None

        self._build_ui()

    def _build_ui(self) -> None:
        """Construct authentication layout."""
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
            text="Exam Day Biometric Authentication",
            font=FONT_HEADER,
            fg="#FFFFFF",
            bg=config.COLOR_HEADER_BG
        ).pack(side="left", padx=20, pady=12)

        ttk.Button(
            header_inner,
            text="📋 Attendance Records",
            style="Secondary.TButton",
            command=self._on_view_attendance_clicked
        ).pack(side="right", pady=12)

        # Main Layout: 2 Columns (Left: Input & Result Card, Right: Live Camera)
        main_box = tk.Frame(self, bg=config.COLOR_BG)
        main_box.pack(fill="both", expand=True, padx=24, pady=16)

        # Left Column: Roll Number Entry & Verdict Card (Width ~430)
        left_panel = tk.Frame(main_box, bg=config.COLOR_CARD_BG, bd=1, relief="solid", highlightthickness=0)
        left_panel.config(highlightbackground=config.COLOR_BORDER)
        left_panel.pack(side="left", fill="both", padx=(0, 16), ipadx=14, ipady=12)

        tk.Label(
            left_panel,
            text="Student Verification",
            font=FONT_SUBHEADER,
            fg=config.COLOR_TEXT_PRIMARY,
            bg=config.COLOR_CARD_BG
        ).pack(anchor="w", padx=14, pady=(10, 6))

        # Active Exam Session Selector
        session_box = tk.Frame(left_panel, bg=config.COLOR_CARD_BG)
        session_box.pack(fill="x", padx=14, pady=(0, 6))

        tk.Label(session_box, text="Exam Session:", font=FONT_SMALL, fg=config.COLOR_TEXT_MUTED, bg=config.COLOR_CARD_BG).pack(side="left")
        self.combo_session = ttk.Combobox(session_box, state="readonly", width=30)
        self.combo_session.pack(side="right", fill="x", expand=True, padx=(8, 0))
        self._populate_sessions()

        # Roll Number Input Field
        tk.Label(
            left_panel,
            text="Roll Number (Optional — or blank for Auto-Identify):",
            font=FONT_BODY_BOLD,
            fg=config.COLOR_TEXT_PRIMARY,
            bg=config.COLOR_CARD_BG
        ).pack(anchor="w", padx=14, pady=(2, 2))

        input_row = tk.Frame(left_panel, bg=config.COLOR_CARD_BG)
        input_row.pack(fill="x", padx=14, pady=(0, 6))

        self.entry_roll = ttk.Entry(input_row, style="App.TEntry", font=(FONT_BODY[0], 11))
        self.entry_roll.pack(side="left", fill="x", expand=True)
        self.entry_roll.bind("<Return>", lambda e: self._on_verify_clicked())

        ttk.Button(
            input_row,
            text="Verify Face 🎯",
            style="Primary.TButton",
            command=self._on_verify_clicked
        ).pack(side="right", padx=(6, 0))

        # Auto-Identify Button Row
        auto_row = tk.Frame(left_panel, bg=config.COLOR_CARD_BG)
        auto_row.pack(fill="x", padx=14, pady=(0, 8))

        ttk.Button(
            auto_row,
            text="🔍 Auto-Identify Face (Auto-fills Roll No)",
            style="Success.TButton",
            command=self._on_auto_identify_clicked
        ).pack(fill="x")

        # Anti-Spoofing Liveness Toggle & Camera Controls
        ctrl_bar = tk.Frame(left_panel, bg=config.COLOR_CARD_BG)
        ctrl_bar.pack(fill="x", padx=14, pady=(0, 8))

        self.var_liveness = tk.BooleanVar(value=config.LIVENESS_CHECK_ENABLED_DEFAULT)
        self.chk_liveness = ttk.Checkbutton(
            ctrl_bar,
            text="🛡️ Anti-Spoofing Liveness Check",
            variable=self.var_liveness
        )
        self.chk_liveness.pack(side="left")

        # Camera selector
        self.combo_cam = ttk.Combobox(ctrl_bar, values=["Camera 0", "Camera 1", "Camera 2"], state="readonly", width=9)
        self.combo_cam.set(f"Camera {config.DEFAULT_CAMERA_INDEX}")
        self.combo_cam.pack(side="right")
        self.combo_cam.bind("<<ComboboxSelected>>", self._on_camera_selected)

        cam_btn_row = tk.Frame(left_panel, bg=config.COLOR_CARD_BG)
        cam_btn_row.pack(fill="x", padx=14, pady=(0, 10))

        self.btn_toggle_cam = ttk.Button(
            cam_btn_row,
            text="▶ Start Camera",
            style="Secondary.TButton",
            command=self._toggle_camera
        )
        self.btn_toggle_cam.pack(side="left", fill="x", expand=True, padx=(0, 4))

        ttk.Button(
            cam_btn_row,
            text="📁 Test Photo...",
            style="Secondary.TButton",
            command=self._browse_test_photo
        ).pack(side="right", fill="x", expand=True, padx=(4, 0))

        ttk.Separator(left_panel, orient="horizontal").pack(fill="x", padx=14, pady=(0, 8))

        # -----------------------------------------------------------------
        # Verdict Result Card
        # -----------------------------------------------------------------
        tk.Label(
            left_panel,
            text="Authentication Verdict",
            font=FONT_SUBHEADER,
            fg=config.COLOR_TEXT_PRIMARY,
            bg=config.COLOR_CARD_BG
        ).pack(anchor="w", padx=14, pady=(0, 6))

        self.card_result = tk.Frame(
            left_panel,
            bg="#F8FAFC",
            bd=2,
            relief="solid",
            highlightthickness=0
        )
        self.card_result.config(highlightbackground="#CBD5E1")
        self.card_result.pack(fill="both", expand=True, padx=14, pady=(0, 12))

        self.card_inner = tk.Frame(self.card_result, bg="#F8FAFC", padx=14, pady=12)
        self.card_inner.pack(fill="both", expand=True)

        self.lbl_verdict_title = tk.Label(
            self.card_inner,
            text="AWAITING VERIFICATION",
            font=FONT_VERDICT,
            fg=config.COLOR_TEXT_MUTED,
            bg="#F8FAFC"
        )
        self.lbl_verdict_title.pack(anchor="w", pady=(0, 8))

        # Detailed Key-Value Rows
        self.lbl_student_info = tk.Label(
            self.card_inner,
            text="Student: —\nRoll Number: —",
            font=FONT_BODY_BOLD,
            fg=config.COLOR_TEXT_PRIMARY,
            bg="#F8FAFC",
            justify="left"
        )
        self.lbl_student_info.pack(anchor="w", pady=2)

        ttk.Separator(self.card_inner, orient="horizontal").pack(fill="x", pady=6)

        self.lbl_match_status = tk.Label(
            self.card_inner,
            text="Face Match: —",
            font=FONT_BODY,
            fg=config.COLOR_TEXT_MUTED,
            bg="#F8FAFC"
        )
        self.lbl_match_status.pack(anchor="w", pady=1)

        self.lbl_liveness_status = tk.Label(
            self.card_inner,
            text="Liveness: —",
            font=FONT_BODY,
            fg=config.COLOR_TEXT_MUTED,
            bg="#F8FAFC"
        )
        self.lbl_liveness_status.pack(anchor="w", pady=1)

        self.lbl_attendance_status = tk.Label(
            self.card_inner,
            text="Attendance: —",
            font=FONT_BODY,
            fg=config.COLOR_TEXT_MUTED,
            bg="#F8FAFC"
        )
        self.lbl_attendance_status.pack(anchor="w", pady=1)

        self.lbl_access_status = tk.Label(
            self.card_inner,
            text="Exam Access: —",
            font=FONT_BODY_BOLD,
            fg=config.COLOR_TEXT_MUTED,
            bg="#F8FAFC"
        )
        self.lbl_access_status.pack(anchor="w", pady=1)

        self.lbl_score_detail = tk.Label(
            self.card_inner,
            text="Similarity Score: —",
            font=FONT_SMALL,
            fg=config.COLOR_TEXT_MUTED,
            bg="#F8FAFC",
            justify="left"
        )
        self.lbl_score_detail.pack(anchor="w", pady=(4, 0))

        # Buttons on successful verification: View Pass & View Attendance
        self.btn_box_verdict = tk.Frame(self.card_inner, bg="#F8FAFC")

        self.btn_view_pass = ttk.Button(
            self.btn_box_verdict,
            text="🎫 View / Print Exam Pass",
            style="Success.TButton",
            command=self._on_show_exam_pass_modal
        )
        self.btn_view_pass.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_view_attendance = ttk.Button(
            self.btn_box_verdict,
            text="📋 Attendance Log",
            style="Primary.TButton",
            command=self._on_view_attendance_clicked
        )
        self.btn_view_attendance.pack(side="right", fill="x", expand=True, padx=(4, 0))

        # Right Column: Live Camera Feed
        right_panel = tk.Frame(main_box, bg=config.COLOR_CARD_BG, bd=1, relief="solid")
        right_panel.config(highlightbackground=config.COLOR_BORDER)
        right_panel.pack(side="right", fill="both", expand=True)

        preview_header = tk.Frame(right_panel, bg=config.COLOR_CARD_BG)
        preview_header.pack(fill="x", padx=16, pady=12)

        tk.Label(
            preview_header,
            text="Live Face Capture, Quality & Liveness Tracking",
            font=FONT_SUBHEADER,
            fg=config.COLOR_TEXT_PRIMARY,
            bg=config.COLOR_CARD_BG
        ).pack(side="left")

        # Badges (Quality + Detection)
        badges_f = tk.Frame(preview_header, bg=config.COLOR_CARD_BG)
        badges_f.pack(side="right")

        self.lbl_quality_badge = tk.Label(
            badges_f,
            text="Quality: Awaiting",
            font=FONT_SMALL,
            fg="#64748B",
            bg="#E2E8F0",
            padx=8,
            pady=2
        )
        self.lbl_quality_badge.pack(side="left", padx=(0, 8))

        self.lbl_detection_badge = tk.Label(
            badges_f,
            text="Camera Inactive",
            font=FONT_SMALL,
            fg="#64748B",
            bg="#E2E8F0",
            padx=8,
            pady=2
        )
        self.lbl_detection_badge.pack(side="left")

        self.canvas_cam = tk.Canvas(
            right_panel,
            width=config.CAMERA_FRAME_WIDTH,
            height=config.CAMERA_FRAME_HEIGHT,
            bg="#0F172A",
            highlightthickness=0
        )
        self.canvas_cam.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self.canvas_cam.create_text(
            320, 240,
            text="Camera feed is stopped.\nClick 'Start Camera' or 'Test Photo...'",
            fill="#94A3B8",
            font=FONT_BODY,
            justify="center",
            tags="placeholder"
        )

    def _populate_sessions(self) -> None:
        """Fetch active sessions into combobox."""
        sessions = self.db.get_active_sessions()
        self._session_map: Dict[str, int] = {}
        values = ["None (Default Session)"]

        for s in sessions:
            label = f"{s['session_code']} - {s['exam_title']} ({s['hall_number']})"
            values.append(label)
            self._session_map[label] = s["id"]

        self.combo_session["values"] = values
        self.combo_session.set(values[0])

    def _get_selected_session_id(self) -> Optional[int]:
        """Resolve selected session ID."""
        val = self.combo_session.get()
        return self._session_map.get(val)

    def on_show(self) -> None:
        """Invoked when navigating to this screen."""
        self._reset_verdict()
        self._populate_sessions()
        self.liveness_detector.reset()
        self.entry_roll.focus()
        self._start_camera_stream()

    def on_hide(self) -> None:
        """Invoked when leaving this screen."""
        self._stop_camera_stream()

    def _on_back_clicked(self) -> None:
        """Navigate back to dashboard."""
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

    def _start_camera_stream(self, show_error_dialog: bool = False) -> None:
        """Start hardware webcam capture."""
        if self._feed_active:
            return

        try:
            self.camera_manager.start()
            self._feed_active = True
            self.btn_toggle_cam.config(text="⏹ Stop Camera", style="Secondary.TButton")
            self._poll_frame()
        except CameraUnavailableError:
            self._feed_active = False
            self.btn_toggle_cam.config(text="▶ Start Camera", style="Primary.TButton")
            self.lbl_detection_badge.config(text="Camera Unavailable", fg="#DC2626", bg="#FEF2F2")
            self.canvas_cam.delete("all")
            self.canvas_cam.create_text(
                320, 240,
                text="⚠️ Camera could not be accessed.\n\n"
                     "Check your laptop's camera switch or privacy shutter.\n"
                     "OR click '📁 Test Photo...' to test with your photo!",
                fill="#F87171",
                font=FONT_BODY,
                justify="center",
                tags="placeholder"
            )
            if show_error_dialog:
                messagebox.showinfo(
                    "Camera Unavailable",
                    "Camera could not be accessed.\n\n"
                    "Tip: You can click '📁 Test Photo...' right now to test verification using your face photograph!",
                    parent=self
                )

    def _stop_camera_stream(self) -> None:
        """Stop camera polling and hardware capture."""
        if self._poll_job:
            self.after_cancel(self._poll_job)
            self._poll_job = None

        self._feed_active = False
        self.camera_manager.stop()
        self.btn_toggle_cam.config(text="▶ Start Camera", style="Primary.TButton")
        self.lbl_detection_badge.config(text="Camera Inactive", fg="#64748B", bg="#E2E8F0")

    def _toggle_camera(self) -> None:
        """Toggle camera on or off."""
        if self._feed_active:
            self._stop_camera_stream()
            self.canvas_cam.delete("all")
            self.canvas_cam.create_text(
                320, 240,
                text="Camera feed stopped.\nClick 'Start Camera' or 'Test Photo...'",
                fill="#94A3B8",
                font=FONT_BODY,
                justify="center",
                tags="placeholder"
            )
        else:
            self._start_camera_stream(show_error_dialog=True)

    def _browse_test_photo(self) -> None:
        """Load a test photo as virtual camera input for headless or camera-less testing."""
        file_path = filedialog.askopenfilename(
            parent=self,
            title="Select Test Photograph for Authentication",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        self._stop_camera_stream()
        try:
            self.camera_manager.start_virtual_mode(file_path)
            self._feed_active = True
            self.btn_toggle_cam.config(text="⏹ Stop Camera", style="Secondary.TButton")
            self._poll_frame()
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not load image: {e}", parent=self)

    def _poll_frame(self) -> None:
        """Poll frame, evaluate live quality and liveness buffer, render overlay."""
        if not self._feed_active:
            return

        frame = self.camera_manager.read_frame()
        if frame is not None:
            self._current_raw_frame = frame.copy()

            # Detect faces on frame
            faces = self.detector.detect(frame)
            face_count = len(faces)

            if face_count == 0:
                self.lbl_detection_badge.config(text="No Face Detected", fg="#D97706", bg="#FEF3C7")
                self.lbl_quality_badge.config(text="Awaiting Face", fg="#64748B", bg="#E2E8F0")
            elif face_count == 1:
                target_f = faces[0]
                self.lbl_detection_badge.config(
                    text=f"1 Face ({target_f.confidence*100:.1f}%)",
                    fg="#059669",
                    bg="#D1FAE5"
                )
                # Live quality check
                q = self.quality_analyzer.evaluate_frame(frame, face_bbox=(target_f.x, target_f.y, target_f.w, target_f.h))
                self.lbl_quality_badge.config(
                    text=q.message,
                    fg=q.badge_color,
                    bg="#ECFDF5" if q.is_acceptable else "#FEF3C7"
                )

                # Feed liveness buffer continuously
                if self.var_liveness.get():
                    self.liveness_detector.update(frame, target_f)
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

    def _on_auto_identify_clicked(self) -> None:
        """Scan face and automatically identify student without typing roll number."""
        if self._current_raw_frame is None:
            messagebox.showerror(
                "Camera Inactive",
                "Camera feed is not active. Please click '▶ Start Camera' or select a test photo.",
                parent=self
            )
            return

        session_id = self._get_selected_session_id()
        check_live = self.var_liveness.get()

        result: AuthenticationResult = self.auth_service.identify_face(
            frame_bgr=self._current_raw_frame,
            session_id=session_id,
            check_liveness=check_live
        )

        if result.success and result.roll_number:
            self.entry_roll.delete(0, "end")
            self.entry_roll.insert(0, result.roll_number)

        self._last_result = result
        self._display_verdict(result)

    def _on_verify_clicked(self) -> None:
        """
        Execute authentication verification.
        If roll number is provided: 1:1 match. If blank: auto-identify.
        """
        roll = self.entry_roll.get().strip()
        if not roll:
            self._on_auto_identify_clicked()
            return

        if self._current_raw_frame is None:
            messagebox.showerror(
                "Camera Error",
                "Camera could not be accessed. Please start the camera or select a test photo.",
                parent=self
            )
            return

        session_id = self._get_selected_session_id()
        check_live = self.var_liveness.get()

        result: AuthenticationResult = self.auth_service.authenticate(
            roll_number=roll,
            frame_bgr=self._current_raw_frame,
            session_id=session_id,
            check_liveness=check_live
        )

        self._last_result = result
        self._display_verdict(result)

    def _display_verdict(self, result: AuthenticationResult) -> None:
        """Update result panel with visually prominent verdict styling."""
        if result.success:
            # IDENTITY VERIFIED (Emerald Green Theme)
            bg_color = config.COLOR_SUCCESS_BG
            border_color = config.COLOR_SUCCESS
            text_color = config.COLOR_SUCCESS

            self.card_result.config(bg=bg_color, highlightbackground=border_color)
            self.card_inner.config(bg=bg_color)

            self.lbl_verdict_title.config(
                text="✓ IDENTITY VERIFIED",
                fg=text_color,
                bg=bg_color
            )
            self.lbl_student_info.config(
                text=f"Candidate: {result.student_name}\nRoll Number: {result.roll_number}",
                fg=config.COLOR_TEXT_PRIMARY,
                bg=bg_color
            )
            self.lbl_match_status.config(
                text=f"Face Match: {result.face_match_status}",
                fg=text_color,
                font=FONT_BODY_BOLD,
                bg=bg_color
            )
            live_text = f"Liveness: {result.liveness_status or 'VERIFIED'} (Score: {result.liveness_score:.2f})" if result.liveness_score is not None else "Liveness: VERIFIED"
            self.lbl_liveness_status.config(
                text=live_text,
                fg=text_color,
                font=FONT_BODY,
                bg=bg_color
            )
            att_display = (
                f"Attendance: {result.attendance_status} ✓ (Saved in Database)"
                if "PRESENT" in result.attendance_status
                else f"Attendance: {result.attendance_status}"
            )
            self.lbl_attendance_status.config(
                text=att_display,
                fg=text_color,
                font=FONT_BODY_BOLD,
                bg=bg_color
            )
            self.lbl_access_status.config(
                text=f"Exam Access: {result.exam_access}",
                fg=text_color,
                font=(FONT_BODY[0], 12, "bold"),
                bg=bg_color
            )
            score_txt = f"Cosine Similarity: {result.similarity_score:.3f} (Threshold: {result.threshold:.3f})" if result.similarity_score is not None else ""
            if result.exam_pass:
                score_txt += f"\nPass Token: {result.exam_pass.pass_code}"
            self.lbl_score_detail.config(
                text=score_txt,
                fg=config.COLOR_TEXT_MUTED,
                bg=bg_color
            )
            self.btn_box_verdict.config(bg=bg_color)
            self.btn_box_verdict.pack(anchor="w", pady=(10, 0), fill="x")
        else:
            self.btn_box_verdict.pack_forget()
            # AUTHENTICATION FAILED (Red Theme)
            bg_color = config.COLOR_DANGER_BG
            border_color = config.COLOR_DANGER
            text_color = config.COLOR_DANGER

            self.card_result.config(bg=bg_color, highlightbackground=border_color)
            self.card_inner.config(bg=bg_color)

            title_text = "⚠️ " + result.status_title if "SPOOF" in result.status_title else "✗ AUTHENTICATION FAILED"
            self.lbl_verdict_title.config(
                text=title_text,
                fg=text_color,
                bg=bg_color
            )
            self.lbl_student_info.config(
                text=f"Student: {result.student_name}\nRoll Number: {result.roll_number}",
                fg=config.COLOR_TEXT_PRIMARY,
                bg=bg_color
            )
            self.lbl_match_status.config(
                text=f"Face Match: {result.face_match_status}",
                fg=text_color,
                font=FONT_BODY_BOLD,
                bg=bg_color
            )
            live_text = f"Liveness: {result.liveness_status or 'FAILED'}" + (f" ({result.liveness_score:.2f})" if result.liveness_score is not None else "")
            self.lbl_liveness_status.config(
                text=live_text,
                fg=text_color,
                font=FONT_BODY,
                bg=bg_color
            )
            self.lbl_attendance_status.config(
                text=f"Attendance: {result.attendance_status}",
                fg=text_color,
                font=FONT_BODY_BOLD,
                bg=bg_color
            )
            self.lbl_access_status.config(
                text=f"Exam Access: {result.exam_access}",
                fg=text_color,
                font=(FONT_BODY[0], 12, "bold"),
                bg=bg_color
            )
            score_text = (
                f"Cosine Similarity: {result.similarity_score:.3f} (Threshold: {result.threshold:.3f})\n"
                if result.similarity_score is not None else ""
            )
            self.lbl_score_detail.config(
                text=f"{score_text}{result.message}",
                fg=config.COLOR_DANGER,
                bg=bg_color
            )

    def _reset_verdict(self) -> None:
        """Reset verdict card to default neutral state."""
        self.btn_box_verdict.pack_forget()
        self.card_result.config(bg="#F8FAFC", highlightbackground="#CBD5E1")
        self.card_inner.config(bg="#F8FAFC")
        self.lbl_verdict_title.config(text="AWAITING VERIFICATION", fg=config.COLOR_TEXT_MUTED, bg="#F8FAFC")
        self.lbl_student_info.config(text="Student: —\nRoll Number: —", fg=config.COLOR_TEXT_PRIMARY, bg="#F8FAFC")
        self.lbl_match_status.config(text="Face Match: —", fg=config.COLOR_TEXT_MUTED, font=FONT_BODY, bg="#F8FAFC")
        self.lbl_liveness_status.config(text="Liveness: —", fg=config.COLOR_TEXT_MUTED, font=FONT_BODY, bg="#F8FAFC")
        self.lbl_attendance_status.config(text="Attendance: —", fg=config.COLOR_TEXT_MUTED, font=FONT_BODY, bg="#F8FAFC")
        self.lbl_access_status.config(text="Exam Access: —", fg=config.COLOR_TEXT_MUTED, font=FONT_BODY_BOLD, bg="#F8FAFC")
        self.lbl_score_detail.config(text="Similarity Score: —", fg=config.COLOR_TEXT_MUTED, bg="#F8FAFC")

    def _on_show_exam_pass_modal(self) -> None:
        """Open popup window displaying the generated official exam pass."""
        if not self._last_result or not self._last_result.exam_pass:
            messagebox.showinfo("No Pass", "No exam pass is currently available.", parent=self)
            return

        exam_pass: ExamPass = self._last_result.exam_pass

        # Modal Window
        dlg = tk.Toplevel(self)
        dlg.title(f"Official Exam Entry Pass — {exam_pass.roll_number}")
        dlg.geometry("740x560")
        dlg.minsize(700, 520)
        dlg.transient(self)
        dlg.grab_set()

        hdr = tk.Frame(dlg, bg=config.COLOR_HEADER_BG, height=50)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        tk.Label(
            hdr,
            text=f"Verified Exam Entry Pass ({exam_pass.pass_code})",
            font=FONT_HEADER,
            fg="#FFFFFF",
            bg=config.COLOR_HEADER_BG
        ).pack(side="left", padx=20, pady=12)

        body = tk.Frame(dlg, bg=config.COLOR_BG, padx=20, pady=16)
        body.pack(fill="both", expand=True)

        # Render pass image
        if exam_pass.pil_image:
            pil_thumb = exam_pass.pil_image.copy()
            pil_thumb.thumbnail((680, 400), Image.Resampling.LANCZOS)
            self._pass_preview_img = ImageTk.PhotoImage(pil_thumb)

            canvas_pass = tk.Canvas(body, width=pil_thumb.width, height=pil_thumb.height, bg=config.COLOR_BG, highlightthickness=0)
            canvas_pass.pack(pady=(0, 14))
            canvas_pass.create_image(0, 0, anchor="nw", image=self._pass_preview_img)

        btn_row = tk.Frame(body, bg=config.COLOR_BG)
        btn_row.pack(fill="x", pady=4)

        ttk.Button(
            btn_row,
            text="🖨️ Open Printable HTML Receipt",
            style="Primary.TButton",
            command=lambda: self._open_html_pass(exam_pass)
        ).pack(side="left", padx=(0, 8))

        if exam_pass.image_path:
            ttk.Button(
                btn_row,
                text="💾 Save Pass Image...",
                style="Secondary.TButton",
                command=lambda: self._save_pass_image(exam_pass)
            ).pack(side="left", padx=4)

        ttk.Button(btn_row, text="Close", style="Secondary.TButton", command=dlg.destroy).pack(side="right")

    def _open_html_pass(self, exam_pass: ExamPass) -> None:
        """Generate and launch printable HTML in user's default browser."""
        try:
            html_path = self.pass_generator.generate_html_receipt(exam_pass)
            webbrowser.open(f"file:///{os.path.abspath(html_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open pass receipt: {e}", parent=self)

    def _save_pass_image(self, exam_pass: ExamPass) -> None:
        """Save pass image to custom user path."""
        if not exam_pass.pil_image:
            return
        save_path = filedialog.asksaveasfilename(
            parent=self,
            title="Save Exam Entry Pass Image",
            initialfile=f"exam_pass_{exam_pass.roll_number}.png",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")]
        )
        if save_path:
            exam_pass.pil_image.save(save_path, "PNG")
            messagebox.showinfo("Saved", f"Pass saved to:\n\n{save_path}", parent=self)

    def _on_view_attendance_clicked(self) -> None:
        """Navigate directly to the Attendance Records screen."""
        self._stop_camera_stream()
        self.navigate("attendance")
