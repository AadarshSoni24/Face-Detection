"""
ui/dashboard.py - Main Dashboard Screen.

Displays application summary metrics (registered students, today's attendance,
failed authentication attempts) and primary navigation tiles.
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
            text=f"{config.APP_SUBTITLE} • SFace 128-d Verification",
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
            text="ℹ️ About Project",
            style="Secondary.TButton",
            command=self._show_about_dialog
        ).pack(side="left", padx=6)

        # Main Scrollable / Padded Body
        content = tk.Frame(self, bg=config.COLOR_BG)
        content.pack(fill="both", expand=True, padx=32, pady=20)

        # -------------------------------------------------------------
        # Section 1: Metrics Overview Cards
        # -------------------------------------------------------------
        stats_header = tk.Label(
            content,
            text="System Analytics & Today's Overview",
            font=FONT_SUBHEADER,
            fg=config.COLOR_TEXT_PRIMARY,
            bg=config.COLOR_BG
        )
        stats_header.pack(anchor="w", pady=(0, 10))

        self.stats_container = tk.Frame(content, bg=config.COLOR_BG)
        self.stats_container.pack(fill="x", pady=(0, 24))

        self.card_total_students = self._create_stat_card(
            self.stats_container, "Total Registered Students", "0", config.COLOR_PRIMARY
        )
        self.card_total_students.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.card_attendance_today = self._create_stat_card(
            self.stats_container, "Present Today (Verified)", "0", config.COLOR_SUCCESS
        )
        self.card_attendance_today.pack(side="left", fill="both", expand=True, padx=10)

        self.card_failed_today = self._create_stat_card(
            self.stats_container, "Failed Auth Attempts Today", "0", config.COLOR_DANGER
        )
        self.card_failed_today.pack(side="left", fill="both", expand=True, padx=(10, 0))

        # -------------------------------------------------------------
        # Section 2: Quick Action Modules
        # -------------------------------------------------------------
        actions_header = tk.Label(
            content,
            text="Core Examination Modules",
            font=FONT_SUBHEADER,
            fg=config.COLOR_TEXT_PRIMARY,
            bg=config.COLOR_BG
        )
        actions_header.pack(anchor="w", pady=(0, 12))

        modules_grid = tk.Frame(content, bg=config.COLOR_BG)
        modules_grid.pack(fill="both", expand=True)

        # Grid configured with 3 columns
        modules_grid.columnconfigure(0, weight=1, uniform="col")
        modules_grid.columnconfigure(1, weight=1, uniform="col")
        modules_grid.columnconfigure(2, weight=1, uniform="col")

        # Tile 1: Authenticate Student (Exam Day)
        self._create_action_card(
            modules_grid,
            row=0, col=0,
            title="🎯 Exam Day Authentication",
            desc="Verify student live face against registered encoding. Marks attendance and grants exam access.",
            button_text="Open Authentication",
            btn_style="Primary.TButton",
            command=lambda: self.navigate("authentication")
        )

        # Tile 2: Register Student (Webcam)
        self._create_action_card(
            modules_grid,
            row=0, col=1,
            title="📸 Register Student (Webcam)",
            desc="Enroll a new student manually using live camera face capture and 128-d SFace embedding.",
            button_text="Register Student",
            btn_style="Success.TButton",
            command=lambda: self.navigate("registration")
        )

        # Tile 3: Bulk Import Student Photos
        self._create_action_card(
            modules_grid,
            row=0, col=2,
            title="📂 Bulk Photo Import",
            desc="Batch import student photos with CSV mapping, automated quality checks, and manual review tagging.",
            button_text="Import Photos",
            btn_style="Secondary.TButton",
            command=lambda: self.navigate("import_photos")
        )

        # Tile 4: Attendance Records
        self._create_action_card(
            modules_grid,
            row=1, col=0,
            title="📋 Attendance Records",
            desc="Review authenticated attendance entries, filter by date, and audit verification statuses.",
            button_text="View Attendance",
            btn_style="Secondary.TButton",
            command=lambda: self.navigate("attendance")
        )

        # Tile 5: Student Records
        self._create_action_card(
            modules_grid,
            row=1, col=1,
            title="👥 Student Records",
            desc="Manage registered students, view enrollment details, search by roll number, or delete records.",
            button_text="Manage Students",
            btn_style="Secondary.TButton",
            command=lambda: self.navigate("students")
        )

        # Tile 6: Academic Information & Disclaimer
        self._create_action_card(
            modules_grid,
            row=1, col=2,
            title="🛡️ System & Privacy Info",
            desc="Review academic design, biometric privacy policies, threshold benchmarks, and liveness notes.",
            button_text="Project Details",
            btn_style="Secondary.TButton",
            command=self._show_about_dialog
        )

        # Bottom Disclaimer Banner
        footer_frame = tk.Frame(self, bg="#FEF3C7", height=38)
        footer_frame.pack(fill="x", side="bottom")
        footer_frame.pack_propagate(False)

        tk.Label(
            footer_frame,
            text=f"⚠️ {config.LIVENESS_DISCLAIMER}",
            font=FONT_SMALL,
            fg="#92400E",
            bg="#FEF3C7"
        ).pack(side="left", padx=20, pady=8)

    def _create_stat_card(self, parent: tk.Widget, title: str, default_val: str, color: str) -> tk.Frame:
        """Create a rounded-styled summary metric card."""
        card = tk.Frame(parent, bg=config.COLOR_CARD_BG, bd=1, relief="solid", highlightthickness=0)
        card.config(highlightbackground=config.COLOR_BORDER)

        inner = tk.Frame(card, bg=config.COLOR_CARD_BG, padx=16, pady=14)
        inner.pack(fill="both", expand=True)

        tk.Label(inner, text=title, font=FONT_SMALL, fg=config.COLOR_TEXT_MUTED, bg=config.COLOR_CARD_BG).pack(anchor="w")
        val_lbl = tk.Label(inner, text=default_val, font=(FONT_BODY[0], 24, "bold"), fg=color, bg=config.COLOR_CARD_BG)
        val_lbl.pack(anchor="w", pady=(4, 0))
        card.val_label = val_lbl  # store reference
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
        card.grid(row=row, column=col, sticky="nsew", padx=8, pady=8)

        inner = tk.Frame(card, bg=config.COLOR_CARD_BG, padx=18, pady=16)
        inner.pack(fill="both", expand=True)

        tk.Label(inner, text=title, font=FONT_SUBHEADER, fg=config.COLOR_TEXT_PRIMARY, bg=config.COLOR_CARD_BG).pack(anchor="w")
        tk.Label(
            inner,
            text=desc,
            font=FONT_SMALL,
            fg=config.COLOR_TEXT_MUTED,
            bg=config.COLOR_CARD_BG,
            wraplength=280,
            justify="left"
        ).pack(anchor="w", pady=(6, 14), fill="x", expand=True)

        ttk.Button(inner, text=button_text, style=btn_style, command=command).pack(anchor="w")

    def refresh_stats(self) -> None:
        """Fetch latest counts from SQLite and update dashboard metric cards."""
        try:
            stats = self.db.get_dashboard_stats()
            self.card_total_students.val_label.config(text=str(stats["total_students"]))
            self.card_attendance_today.val_label.config(text=str(stats["attendance_today"]))
            self.card_failed_today.val_label.config(text=str(stats["failed_auth_today"]))
        except Exception as e:
            config.logger.error(f"Failed to refresh dashboard stats: {e}")

    def _show_about_dialog(self) -> None:
        """Display academic prototype technical information."""
        info = (
            "Biometric-Based Exam Authentication System\n"
            "Academic Prototype\n\n"
            "Technology Stack:\n"
            "• Face Detection: YuNet Deep Neural Network (ONNX)\n"
            "• Feature Extraction: SFace Deep Neural Network (128-d embeddings)\n"
            "• Verification Metric: Cosine Similarity (Threshold: 0.363)\n"
            "• Database: SQLite (Parameterized queries)\n"
            "• UI Framework: Python Tkinter / TTK\n\n"
            "Privacy & Security:\n"
            "• Only mathematical 128-d feature embeddings are stored.\n"
            "• Raw face templates are never exposed in user interface.\n"
            "• Localized execution; zero cloud transmission.\n\n"
            "Limitations:\n"
            "• Prototype does not include active liveness/anti-spoofing.\n"
            "• For evaluation & academic demonstration purposes only."
        )
        messagebox.showinfo("About - Exam Authentication System", info, parent=self)
