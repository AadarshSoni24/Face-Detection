"""
ui/attendance.py - Attendance Records Screen.

Displays verified examination attendance entries with date, session, and identity filtering.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Callable, Dict
import datetime
import csv
import os
import config
from database.db import Database
from ui.styles import FONT_SUBHEADER, FONT_BODY, FONT_SMALL, FONT_HEADER, FONT_BODY_BOLD


class AttendanceView(ttk.Frame):
    """Attendance auditing screen with filtering and tabular view."""

    def __init__(self, parent: tk.Widget, db: Database, navigate_callback: Callable[[str], None]):
        super().__init__(parent, style="App.TFrame")
        self.db = db
        self.navigate = navigate_callback

        self._build_ui()

    def _build_ui(self) -> None:
        """Construct attendance UI."""
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
            text="Examination Attendance Log",
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

        # Main Body
        content = tk.Frame(self, bg=config.COLOR_BG)
        content.pack(fill="both", expand=True, padx=24, pady=20)

        # Control / Filter Bar
        ctrl_card = tk.Frame(content, bg=config.COLOR_CARD_BG, bd=1, relief="solid")
        ctrl_card.config(highlightbackground=config.COLOR_BORDER)
        ctrl_card.pack(fill="x", pady=(0, 16), padx=2, ipady=8)

        ctrl_inner = tk.Frame(ctrl_card, bg=config.COLOR_CARD_BG, padx=16, pady=8)
        ctrl_inner.pack(fill="x")

        # Date Filter
        tk.Label(ctrl_inner, text="Date (YYYY-MM-DD):", font=FONT_BODY_BOLD, bg=config.COLOR_CARD_BG).pack(side="left", padx=(0, 6))
        self.entry_date = ttk.Entry(ctrl_inner, width=12, style="App.TEntry")
        self.entry_date.pack(side="left", padx=(0, 10))
        self.entry_date.insert(0, datetime.date.today().strftime("%Y-%m-%d"))

        # Session Filter
        tk.Label(ctrl_inner, text="Session:", font=FONT_BODY_BOLD, bg=config.COLOR_CARD_BG).pack(side="left", padx=(4, 6))
        self.combo_session = ttk.Combobox(ctrl_inner, state="readonly", width=14)
        self.combo_session.pack(side="left", padx=(0, 10))
        self.combo_session.bind("<<ComboboxSelected>>", lambda e: self.refresh_table())

        # Search Query Filter
        tk.Label(ctrl_inner, text="Search Roll/Name:", font=FONT_BODY_BOLD, bg=config.COLOR_CARD_BG).pack(side="left", padx=(4, 6))
        self.entry_search = ttk.Entry(ctrl_inner, width=16, style="App.TEntry")
        self.entry_search.pack(side="left", padx=(0, 10))
        self.entry_search.bind("<KeyRelease>", lambda e: self.refresh_table())

        ttk.Button(ctrl_inner, text="🔍 Filter", style="Primary.TButton", command=self.refresh_table).pack(side="left", padx=4)
        ttk.Button(ctrl_inner, text="All Dates", style="Secondary.TButton", command=self._on_show_all_dates).pack(side="left", padx=4)
        ttk.Button(ctrl_inner, text="🔄 Reset", style="Secondary.TButton", command=self._on_reset_filters).pack(side="left", padx=4)
        ttk.Button(ctrl_inner, text="📥 Export CSV", style="Success.TButton", command=self._on_export_csv).pack(side="left", padx=(6, 4))

        # Counter on Right
        self.lbl_count = tk.Label(ctrl_inner, text="Records: 0", font=FONT_SMALL, fg=config.COLOR_TEXT_MUTED, bg=config.COLOR_CARD_BG)
        self.lbl_count.pack(side="right", padx=8)

        # Table Panel
        table_card = tk.Frame(content, bg=config.COLOR_CARD_BG, bd=1, relief="solid")
        table_card.config(highlightbackground=config.COLOR_BORDER)
        table_card.pack(fill="both", expand=True, padx=2)

        # Treeview Table
        columns = ("id", "roll_number", "name", "course", "session", "date", "time", "status", "pass")
        self.tree = ttk.Treeview(table_card, columns=columns, show="headings", selectmode="browse")

        self.tree.heading("id", text="Entry ID")
        self.tree.heading("roll_number", text="Roll Number")
        self.tree.heading("name", text="Student Name")
        self.tree.heading("course", text="Course / Dept")
        self.tree.heading("session", text="Session Code")
        self.tree.heading("date", text="Exam Date")
        self.tree.heading("time", text="Auth Time")
        self.tree.heading("status", text="Status")
        self.tree.heading("pass", text="Pass Token")

        self.tree.column("id", width=60, anchor="center")
        self.tree.column("roll_number", width=120, anchor="center")
        self.tree.column("name", width=180, anchor="w")
        self.tree.column("course", width=130, anchor="w")
        self.tree.column("session", width=110, anchor="center")
        self.tree.column("date", width=100, anchor="center")
        self.tree.column("time", width=90, anchor="center")
        self.tree.column("status", width=90, anchor="center")
        self.tree.column("pass", width=130, anchor="center")

        scrollbar = ttk.Scrollbar(table_card, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.tag_configure("present", foreground="#047857")

        self.tree.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=12)
        scrollbar.pack(side="right", fill="y", padx=(0, 12), pady=12)

    def on_show(self) -> None:
        """Invoked when navigating to this screen."""
        self._populate_sessions()
        self.refresh_table()

    def _populate_sessions(self) -> None:
        """Populate session filter dropdown."""
        sessions = self.db.get_all_sessions()
        self._session_map: Dict[str, int] = {}
        vals = ["ALL SESSIONS"]
        for s in sessions:
            lbl = f"{s['session_code']}"
            vals.append(lbl)
            self._session_map[lbl] = s["id"]
        self.combo_session["values"] = vals
        self.combo_session.set("ALL SESSIONS")

    def _on_show_all_dates(self) -> None:
        """Clear date filter to show all recorded dates."""
        self.entry_date.delete(0, "end")
        self.refresh_table()

    def _on_reset_filters(self) -> None:
        """Reset date to today and clear search."""
        self.entry_date.delete(0, "end")
        self.entry_date.insert(0, datetime.date.today().strftime("%Y-%m-%d"))
        self.combo_session.set("ALL SESSIONS")
        self.entry_search.delete(0, "end")
        self.refresh_table()

    def refresh_table(self) -> None:
        """Query database and render attendance records."""
        for row in self.tree.get_children():
            self.tree.delete(row)

        date_val = self.entry_date.get().strip()
        search_val = self.entry_search.get().strip()
        sel_session = self.combo_session.get()
        session_id = self._session_map.get(sel_session) if sel_session != "ALL SESSIONS" else None

        records = self.db.get_attendance_records(
            date_str=date_val if date_val else None,
            search_query=search_val if search_val else None,
            session_id=session_id
        )

        for rec in records:
            status = rec.get("status", "")
            row_tag = ("present",) if status == "PRESENT" else ()
            self.tree.insert(
                "",
                "end",
                iid=str(rec["id"]),
                values=(
                    rec["id"],
                    rec["roll_number"],
                    rec["name"],
                    rec.get("course") or "—",
                    rec.get("session_code") or "—",
                    rec["date"],
                    rec["time"],
                    status,
                    rec.get("pass_code") or "—"
                ),
                tags=row_tag
            )

        self.lbl_count.config(text=f"Records: {len(records)} present")

    def _on_export_csv(self) -> None:
        """Export displayed attendance records to a CSV spreadsheet file."""
        records = []
        for item_id in self.tree.get_children():
            records.append(self.tree.item(item_id)["values"])

        if not records:
            messagebox.showinfo("No Records", "No attendance records to export.", parent=self)
            return

        today = datetime.date.today().strftime("%Y-%m-%d")
        default_name = f"exam_attendance_{today}.csv"

        save_path = filedialog.asksaveasfilename(
            parent=self,
            title="Save Attendance Spreadsheet",
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV Spreadsheet", "*.csv"), ("All Files", "*.*")]
        )
        if not save_path:
            return

        try:
            with open(save_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Entry ID", "Roll Number", "Student Name", "Course / Department", "Session Code", "Exam Date", "Auth Time", "Status", "Pass Token"])
                for r in records:
                    writer.writerow(r)

            messagebox.showinfo(
                "Export Successful",
                f"Successfully exported {len(records)} attendance records to:\n\n{save_path}",
                parent=self
            )
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save CSV file:\n{e}", parent=self)
