"""
ui/sessions.py - Examination Session & Hall Management Screen.

Allows administrators to schedule examination sessions, configure examination halls
and courses, and monitor session-wise biometric attendance rosters.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Callable
import datetime
import csv
import os
import config
from database.db import Database
from ui.styles import (
    FONT_TITLE, FONT_SUBHEADER, FONT_BODY, FONT_SMALL,
    FONT_HEADER, FONT_BODY_BOLD
)


class SessionsView(ttk.Frame):
    """Exam Sessions and Hall Allocation Management Screen."""

    def __init__(self, parent: tk.Widget, db: Database, navigate_callback: Callable[[str], None]):
        super().__init__(parent, style="App.TFrame")
        self.db = db
        self.navigate = navigate_callback

        self._build_ui()

    def _build_ui(self) -> None:
        """Construct the session management interface."""
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
            command=lambda: self.navigate("dashboard")
        ).pack(side="left", pady=12)

        tk.Label(
            header_inner,
            text="Exam Sessions & Hall Allocation",
            font=FONT_HEADER,
            fg="#FFFFFF",
            bg=config.COLOR_HEADER_BG
        ).pack(side="left", padx=20, pady=12)

        ttk.Button(
            header_inner,
            text="🎯 Biometric Authentication",
            style="Primary.TButton",
            command=lambda: self.navigate("authentication")
        ).pack(side="right", pady=12)

        # Main Body: 2 Columns (Left: Create Session Form, Right: Session Table)
        main_box = tk.Frame(self, bg=config.COLOR_BG)
        main_box.pack(fill="both", expand=True, padx=24, pady=20)

        # Left Panel: Create Session Form
        left_panel = tk.Frame(main_box, bg=config.COLOR_CARD_BG, bd=1, relief="solid")
        left_panel.config(highlightbackground=config.COLOR_BORDER)
        left_panel.pack(side="left", fill="y", padx=(0, 16), ipadx=14, ipady=14)

        tk.Label(
            left_panel,
            text="Schedule Exam Session",
            font=FONT_SUBHEADER,
            fg=config.COLOR_TEXT_PRIMARY,
            bg=config.COLOR_CARD_BG
        ).pack(anchor="w", padx=14, pady=(12, 10))

        # Session Code
        tk.Label(left_panel, text="Session Code *", font=FONT_BODY_BOLD, bg=config.COLOR_CARD_BG).pack(anchor="w", padx=14, pady=(4, 2))
        self.entry_code = ttk.Entry(left_panel, style="App.TEntry", width=30)
        self.entry_code.pack(fill="x", padx=14, pady=(0, 8))

        # Course Name
        tk.Label(left_panel, text="Course / Department *", font=FONT_BODY_BOLD, bg=config.COLOR_CARD_BG).pack(anchor="w", padx=14, pady=(4, 2))
        self.entry_course = ttk.Entry(left_panel, style="App.TEntry", width=30)
        self.entry_course.pack(fill="x", padx=14, pady=(0, 8))

        # Exam Title
        tk.Label(left_panel, text="Exam / Subject Title *", font=FONT_BODY_BOLD, bg=config.COLOR_CARD_BG).pack(anchor="w", padx=14, pady=(4, 2))
        self.entry_title = ttk.Entry(left_panel, style="App.TEntry", width=30)
        self.entry_title.pack(fill="x", padx=14, pady=(0, 8))

        # Hall Number
        tk.Label(left_panel, text="Exam Hall / Room *", font=FONT_BODY_BOLD, bg=config.COLOR_CARD_BG).pack(anchor="w", padx=14, pady=(4, 2))
        self.entry_hall = ttk.Entry(left_panel, style="App.TEntry", width=30)
        self.entry_hall.pack(fill="x", padx=14, pady=(0, 8))
        self.entry_hall.insert(0, config.DEFAULT_EXAM_HALL)

        # Date & Time Row
        tk.Label(left_panel, text="Exam Date (YYYY-MM-DD) *", font=FONT_BODY_BOLD, bg=config.COLOR_CARD_BG).pack(anchor="w", padx=14, pady=(4, 2))
        self.entry_date = ttk.Entry(left_panel, style="App.TEntry", width=30)
        self.entry_date.pack(fill="x", padx=14, pady=(0, 8))
        self.entry_date.insert(0, datetime.date.today().strftime("%Y-%m-%d"))

        time_frame = tk.Frame(left_panel, bg=config.COLOR_CARD_BG)
        time_frame.pack(fill="x", padx=14, pady=(0, 12))

        tk.Label(time_frame, text="Start:", font=FONT_SMALL, bg=config.COLOR_CARD_BG).pack(side="left", padx=(0, 4))
        self.entry_start = ttk.Entry(time_frame, width=9, style="App.TEntry")
        self.entry_start.pack(side="left", padx=(0, 8))
        self.entry_start.insert(0, "09:30 AM")

        tk.Label(time_frame, text="End:", font=FONT_SMALL, bg=config.COLOR_CARD_BG).pack(side="left", padx=(0, 4))
        self.entry_end = ttk.Entry(time_frame, width=9, style="App.TEntry")
        self.entry_end.pack(side="left")
        self.entry_end.insert(0, "12:30 PM")

        # Create Button
        ttk.Button(
            left_panel,
            text="➕ Create Exam Session",
            style="Primary.TButton",
            command=self._on_create_session
        ).pack(fill="x", padx=14, pady=(4, 8))

        ttk.Button(
            left_panel,
            text="🧹 Clear Form",
            style="Secondary.TButton",
            command=self._clear_form
        ).pack(fill="x", padx=14, pady=(0, 12))

        # Right Panel: Sessions Table & Actions
        right_panel = tk.Frame(main_box, bg=config.COLOR_CARD_BG, bd=1, relief="solid")
        right_panel.config(highlightbackground=config.COLOR_BORDER)
        right_panel.pack(side="right", fill="both", expand=True)

        # Filter Bar
        ctrl_bar = tk.Frame(right_panel, bg=config.COLOR_CARD_BG, padx=16, pady=12)
        ctrl_bar.pack(fill="x")

        tk.Label(ctrl_bar, text="Search Sessions:", font=FONT_BODY_BOLD, bg=config.COLOR_CARD_BG).pack(side="left", padx=(0, 8))
        self.entry_search = ttk.Entry(ctrl_bar, width=20, style="App.TEntry")
        self.entry_search.pack(side="left", padx=(0, 8))
        self.entry_search.bind("<KeyRelease>", lambda e: self.refresh_table())

        ttk.Button(ctrl_bar, text="🔍 Search", style="Primary.TButton", command=self.refresh_table).pack(side="left", padx=4)
        ttk.Button(ctrl_bar, text="🔄 Refresh", style="Secondary.TButton", command=self.refresh_table).pack(side="left", padx=4)
        ttk.Button(ctrl_bar, text="📥 Export Attendance", style="Success.TButton", command=self._on_export_session_attendance).pack(side="left", padx=6)
        ttk.Button(ctrl_bar, text="🗑 Delete Session", style="Danger.TButton", command=self._on_delete_session).pack(side="right", padx=4)

        # Table
        columns = ("id", "code", "course", "title", "hall", "date", "time", "status")
        self.tree = ttk.Treeview(right_panel, columns=columns, show="headings", selectmode="browse")

        self.tree.heading("id", text="ID")
        self.tree.heading("code", text="Session Code")
        self.tree.heading("course", text="Course / Dept")
        self.tree.heading("title", text="Exam Title")
        self.tree.heading("hall", text="Hall / Room")
        self.tree.heading("date", text="Date")
        self.tree.heading("time", text="Time Window")
        self.tree.heading("status", text="Status")

        self.tree.column("id", width=40, anchor="center")
        self.tree.column("code", width=110, anchor="center")
        self.tree.column("course", width=140, anchor="w")
        self.tree.column("title", width=180, anchor="w")
        self.tree.column("hall", width=120, anchor="w")
        self.tree.column("date", width=100, anchor="center")
        self.tree.column("time", width=140, anchor="center")
        self.tree.column("status", width=80, anchor="center")

        scrollbar = ttk.Scrollbar(right_panel, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True, padx=(16, 0), pady=12)
        scrollbar.pack(side="right", fill="y", padx=(0, 16), pady=12)

    def on_show(self) -> None:
        """Invoked when navigating to screen."""
        self.refresh_table()

    def _clear_form(self) -> None:
        """Reset form fields."""
        self.entry_code.delete(0, "end")
        self.entry_course.delete(0, "end")
        self.entry_title.delete(0, "end")
        self.entry_hall.delete(0, "end")
        self.entry_hall.insert(0, config.DEFAULT_EXAM_HALL)
        self.entry_date.delete(0, "end")
        self.entry_date.insert(0, datetime.date.today().strftime("%Y-%m-%d"))

    def _on_create_session(self) -> None:
        """Create new exam session in SQLite."""
        code = self.entry_code.get().strip().upper()
        course = self.entry_course.get().strip()
        title = self.entry_title.get().strip()
        hall = self.entry_hall.get().strip()
        date_str = self.entry_date.get().strip()
        start_t = self.entry_start.get().strip()
        end_t = self.entry_end.get().strip()

        if not code or not course or not title or not hall or not date_str:
            messagebox.showwarning("Validation Error", "All fields are required to create an exam session.", parent=self)
            return

        try:
            sid = self.db.create_exam_session(
                session_code=code,
                course_name=course,
                exam_title=title,
                hall_number=hall,
                exam_date=date_str,
                start_time=start_t,
                end_time=end_t
            )
            messagebox.showinfo("Session Created", f"Successfully created Exam Session:\n\n{code} - {title}\nHall: {hall}\nDate: {date_str}", parent=self)
            self._clear_form()
            self.refresh_table()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create session:\n{e}", parent=self)

    def refresh_table(self) -> None:
        """Reload sessions from database."""
        for row in self.tree.get_children():
            self.tree.delete(row)

        search_q = self.entry_search.get().strip()
        sessions = self.db.get_all_sessions(search_query=search_q)

        for s in sessions:
            time_window = f"{s.get('start_time', '')} - {s.get('end_time', '')}"
            self.tree.insert(
                "",
                "end",
                iid=str(s["id"]),
                values=(
                    s["id"],
                    s["session_code"],
                    s["course_name"],
                    s["exam_title"],
                    s["hall_number"],
                    s["exam_date"],
                    time_window,
                    s["status"]
                )
            )

    def _on_delete_session(self) -> None:
        """Delete selected session."""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Select Session", "Please select a session to delete.", parent=self)
            return

        sid = int(selected[0])
        session = self.db.get_session_by_id(sid)
        if not session:
            return

        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete session:\n\n{session['session_code']} - {session['exam_title']}?",
            parent=self
        )
        if confirm:
            self.db.delete_session(sid)
            self.refresh_table()

    def _on_export_session_attendance(self) -> None:
        """Export attendance roster for selected session."""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Select Session", "Please select a session to export attendance for.", parent=self)
            return

        sid = int(selected[0])
        session = self.db.get_session_by_id(sid)
        if not session:
            return

        records = self.db.get_attendance_records(session_id=sid)
        if not records:
            messagebox.showinfo("No Attendance", f"No verified attendance records found for session {session['session_code']}.", parent=self)
            return

        save_path = filedialog.asksaveasfilename(
            parent=self,
            title=f"Export Attendance for {session['session_code']}",
            initialfile=f"attendance_{session['session_code']}.csv",
            defaultextension=".csv",
            filetypes=[("CSV File", "*.csv"), ("All Files", "*.*")]
        )
        if not save_path:
            return

        try:
            with open(save_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Session Code", "Exam Title", "Hall", "Roll Number", "Student Name", "Course", "Date", "Time", "Status", "Pass Token"])
                for r in records:
                    writer.writerow([
                        session["session_code"],
                        session["exam_title"],
                        session["hall_number"],
                        r["roll_number"],
                        r["name"],
                        r.get("course", ""),
                        r["date"],
                        r["time"],
                        r["status"],
                        r.get("pass_code", "")
                    ])
            messagebox.showinfo("Export Complete", f"Successfully exported {len(records)} records to:\n\n{save_path}", parent=self)
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {e}", parent=self)
