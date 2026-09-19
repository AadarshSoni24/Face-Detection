"""
ui/dashboard.py - Main Dashboard Screen.

Displays application summary metrics (registered students, today's attendance,
failed authentication attempts, active exam sessions) and primary navigation tiles.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable
import config
from database.db import Database
from ui.styles import (
    FONT_TITLE, FONT_SUBHEADER, FONT_BODY, FONT_SMALL,
    FONT_HEADER, FONT_BODY_BOLD
)


class DashboardView(ttk.Frame):
    """Main application landing dashboard."""

    def __init__(self, parent: tk.Widget, db: Database, navigate_callback: Callable[[str], None]):
        super().__init__(parent, style="App.TFrame")
        self.db = db
        self.navigate = navigate_callback

        self._build_ui()
        self.refresh_stats()

    def _build_ui(self) -> None:
        """Construct dashboard components."""
        # Top Header Bar
        header_frame = tk.Frame(self, bg=config.COLOR_HEADER_BG, height=75)
        header_frame.pack(fill="x", side="top")
        header_frame.pack_propagate(False)

        title_box = tk.Frame(header_frame, bg=config.COLOR_HEADER_BG)
        title_box.pack(side="left", padx=24, pady=12)

        tk.Label(
            title_box,
            text=config.APP_TITLE,
            font=FONT_HEADER,
            fg="#FFFFFF",
            bg=config.COLOR_HEADER_BG
        ).pack(anchor="w")

        tk.Label(
            title_box,
            text=f"{config.APP_SUBTITLE} • SFace Biometrics & Anti-Spoofing",
            font=FONT_SMALL,
            fg="#94A3B8",
            bg=config.COLOR_HEADER_BG
        ).pack(anchor="w")

        # Top Right Buttons (About, Refresh)
        top_btn_box = tk.Frame(header_frame, bg=config.COLOR_HEADER_BG)
        top_btn_box.pack(side="right", padx=24, pady=16)

        ttk.Button(
            top_btn_box,
            text="🔄 Refresh Stats",
            style="Secondary.TButton",
            command=self.refresh_stats
        ).pack(side="left", padx=6)

        ttk.Button(
            top_btn_box,
            text="ℹ️ About System",
            style="Secondary.TButton",
            command=self._show_about_dialog
        ).pack(side="left", padx=6)

        # Main Scrollable / Padded Body
        content = tk.Frame(self, bg=config.COLOR_BG)
        content.pack(fill="both", expand=True, padx=32, pady=16)

        # -------------------------------------------------------------
        # Section 1: Metrics Overview Cards (4 Cards)
        # -------------------------------------------------------------
        stats_header = tk.Label(
            content,
            text="System Analytics & Today's Examination Overview",
            font=FONT_SUBHEADER,
            fg=config.COLOR_TEXT_PRIMARY,
            bg=config.COLOR_BG
        )
        stats_header.pack(anchor="w", pady=(0, 8))

        self.stats_container = tk.Frame(content, bg=config.COLOR_BG)
        self.stats_container.pack(fill="x", pady=(0, 16))

        self.card_total_students = self._create_stat_card(
            self.stats_container, "Total Enrolled Students", "0", config.COLOR_PRIMARY
        )
        self.card_total_students.pack(side="left", fill="both", expand=True, padx=(0, 8))

        self.card_attendance_today = self._create_stat_card(
            self.stats_container, "Present Today (Verified)", "0", config.COLOR_SUCCESS
        )
        self.card_attendance_today.pack(side="left", fill="both", expand=True, padx=8)

        self.card_failed_today = self._create_stat_card(
            self.stats_container, "Failed / Spoof Attempts", "0", config.COLOR_DANGER
        )
        self.card_failed_today.pack(side="left", fill="both", expand=True, padx=8)

        self.card_active_sessions = self._create_stat_card(
            self.stats_container, "Active Exam Sessions", "0", "#7C3AED"
        )
        self.card_active_sessions.pack(side="left", fill="both", expand=True, padx=(8, 0))

        # -------------------------------------------------------------
        # Section 2: Quick Action Modules (6 Core Tiles in Grid)
        # -------------------------------------------------------------
        actions_header = tk.Label(
            content,
            text="Examination Modules & Control Center",
            font=FONT_SUBHEADER,
            fg=config.COLOR_TEXT_PRIMARY,
            bg=config.COLOR_BG
        )
        actions_header.pack(anchor="w", pady=(0, 8))

        modules_grid = tk.Frame(content, bg=config.COLOR_BG)
        modules_grid.pack(fill="both", expand=True)

        modules_grid.columnconfigure(0, weight=1, uniform="col")
        modules_grid.columnconfigure(1, weight=1, uniform="col")
        modules_grid.columnconfigure(2, weight=1, uniform="col")

        # Row 0, Col 0: Exam Day Authentication
        self._create_action_card(
            modules_grid,
            row=0, col=0,
            title="🎯 Exam Day Authentication",
            desc="Verify student face with anti-spoofing liveness, mark attendance, and issue digital exam passes.",
            button_text="Open Authentication",
            btn_style="Primary.TButton",
            command=lambda: self.navigate("authentication")
        )

        # Row 0, Col 1: Register Student (Webcam)
        self._create_action_card(
            modules_grid,
            row=0, col=1,
            title="📸 Register Student (Webcam)",
            desc="Enroll new student manually with real-time lighting/focus guidance and SFace 128-d embedding.",
            button_text="Register Student",
            btn_style="Success.TButton",
            command=lambda: self.navigate("registration")
        )

        # Row 0, Col 2: Exam Sessions & Hall Seating
        self._create_action_card(
            modules_grid,
            row=0, col=2,
            title="🏛️ Exam Sessions & Halls",
            desc="Schedule exam sessions, configure course titles & halls, and export session-wise rosters.",
            button_text="Manage Sessions",
            btn_style="Secondary.TButton",
            command=lambda: self.navigate("sessions")
        )

        # Row 1, Col 0: Attendance Records
        self._create_action_card(
            modules_grid,
            row=1, col=0,
            title="📋 Attendance Records",
            desc="Review authenticated attendance entries, filter by session/date, and export CSV spreadsheets.",
            button_text="View Attendance",
            btn_style="Secondary.TButton",
            command=lambda: self.navigate("attendance")
        )

        # Row 1, Col 1: Student Records
        self._create_action_card(
            modules_grid,
            row=1, col=1,
            title="👥 Student Records & Profiles",
            desc="Manage registered students, inspect biometric enrollment details, and view attendance history.",
            button_text="Manage Students",
            btn_style="Secondary.TButton",
            command=lambda: self.navigate("students")
        )

        # Row 1, Col 2: Security & Audit Logs
        self._create_action_card(
            modules_grid,
            row=1, col=2,
            title="🛡️ Security Audit & Logs",
            desc="Inspect biometric verification logs, spoofing alerts, similarity scores, and forensic audit reports.",
            button_text="Inspect Audit Logs",
            btn_style="Secondary.TButton",
            command=lambda: self.navigate("audit_logs")
        )

        # Bottom Disclaimer Banner
        footer_frame = tk.Frame(self, bg="#FEF3C7", height=34)
        footer_frame.pack(fill="x", side="bottom")
        footer_frame.pack_propagate(False)

        tk.Label(
            footer_frame,
            text=f"⚠️ {config.LIVENESS_DISCLAIMER}",
            font=FONT_SMALL,
            fg="#92400E",
            bg="#FEF3C7"
        ).pack(side="left", padx=20, pady=6)

    def _create_stat_card(self, parent: tk.Widget, title: str, default_val: str, color: str) -> tk.Frame:
        """Create a rounded-styled summary metric card."""
        card = tk.Frame(parent, bg=config.COLOR_CARD_BG, bd=1, relief="solid", highlightthickness=0)
        card.config(highlightbackground=config.COLOR_BORDER)

        inner = tk.Frame(card, bg=config.COLOR_CARD_BG, padx=14, pady=12)
        inner.pack(fill="both", expand=True)

        tk.Label(inner, text=title, font=FONT_SMALL, fg=config.COLOR_TEXT_MUTED, bg=config.COLOR_CARD_BG).pack(anchor="w")
        val_lbl = tk.Label(inner, text=default_val, font=(FONT_BODY[0], 22, "bold"), fg=color, bg=config.COLOR_CARD_BG)
        val_lbl.pack(anchor="w", pady=(2, 0))
        card.val_label = val_lbl
        return card

    def _create_action_card(
        self,
        parent: tk.Widget,
        row: int,
        col: int,
        title: str,
        desc: str,
        button_text: str,
        btn_style: str,
        command: Callable[[], None]
    ) -> None:
        """Create an interactive action card within the dashboard grid."""
        card = tk.Frame(parent, bg=config.COLOR_CARD_BG, bd=1, relief="solid")
        card.config(highlightbackground=config.COLOR_BORDER)
        card.grid(row=row, column=col, sticky="nsew", padx=6, pady=6)

        inner = tk.Frame(card, bg=config.COLOR_CARD_BG, padx=16, pady=14)
        inner.pack(fill="both", expand=True)

        tk.Label(inner, text=title, font=FONT_SUBHEADER, fg=config.COLOR_TEXT_PRIMARY, bg=config.COLOR_CARD_BG).pack(anchor="w")
        tk.Label(
            inner,
            text=desc,
            font=FONT_SMALL,
            fg=config.COLOR_TEXT_MUTED,
            bg=config.COLOR_CARD_BG,
            wraplength=270,
            justify="left"
        ).pack(anchor="w", pady=(4, 10), fill="x", expand=True)

        ttk.Button(inner, text=button_text, style=btn_style, command=command).pack(anchor="w")

    def refresh_stats(self) -> None:
        """Fetch latest counts from SQLite and update dashboard metric cards."""
        try:
            stats = self.db.get_dashboard_stats()
            self.card_total_students.val_label.config(text=str(stats["total_students"]))
            self.card_attendance_today.val_label.config(text=str(stats["attendance_today"]))
            self.card_failed_today.val_label.config(text=str(stats["failed_auth_today"]))
            self.card_active_sessions.val_label.config(text=str(stats["active_sessions"]))
        except Exception as e:
            config.logger.error(f"Failed to refresh dashboard stats: {e}")

    def _show_about_dialog(self) -> None:
        """Display academic prototype technical information."""
        info = (
            "Biometric-Based Exam Authentication System\n\n"
            "Features & Technology Stack:\n"
            "• Face Detection: YuNet Deep Neural Network (ONNX)\n"
            "• Feature Extraction: SFace Deep Neural Network (128-d embeddings)\n"
            "• Matching Metric: Cosine Similarity (Threshold: 0.363)\n"
            "• Anti-Spoofing: Landmark Micro-Motion & Blink Liveness Detection\n"
            "• Exam Passes: Digital Hall Ticket & Verification Pass Generator\n"
            "• Sessions: Course & Hall Seating Allocation Engine\n"
            "• Audit Logging: Forensic Security Log & Integrity Inspector\n"
            "• Quality Analyzer: Illumination, Blur, and Pose Guidance\n"
            "• Database: SQLite (Parameterized & Thread-Safe)\n\n"
            "Privacy & Security:\n"
            "• Only mathematical 128-d feature embeddings are stored.\n"
            "• Raw biometric face templates are never exposed in user interface.\n"
            "• Zero cloud telemetry; 100% local processing."
        )
        messagebox.showinfo("About - Exam Authentication System", info, parent=self)
