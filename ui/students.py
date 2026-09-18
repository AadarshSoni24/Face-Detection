"""
ui/students.py - Student Records Management Screen.

Provides listing, filtering, searching, and confirmed deletion of registered students.
Never exposes raw biometric feature embeddings in the visual interface.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable
import config
from database.db import Database
from ui.styles import FONT_SUBHEADER, FONT_BODY, FONT_SMALL, FONT_HEADER, FONT_BODY_BOLD


class StudentsView(ttk.Frame):
    """Student management screen with searchable tabular view."""

    def __init__(self, parent: tk.Widget, db: Database, navigate_callback: Callable[[str], None]):
        super().__init__(parent, style="App.TFrame")
        self.db = db
        self.navigate = navigate_callback

        self._build_ui()

    def _build_ui(self) -> None:
        """Construct student management UI."""
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
            text="Registered Student Records",
            font=FONT_HEADER,
            fg="#FFFFFF",
            bg=config.COLOR_HEADER_BG
        ).pack(side="left", padx=20, pady=12)

        # Main Body
        content = tk.Frame(self, bg=config.COLOR_BG)
        content.pack(fill="both", expand=True, padx=24, pady=20)

        # Control / Filter Bar
        ctrl_card = tk.Frame(content, bg=config.COLOR_CARD_BG, bd=1, relief="solid")
        ctrl_card.config(highlightbackground=config.COLOR_BORDER)
        ctrl_card.pack(fill="x", pady=(0, 16), padx=2, ipady=8)

        ctrl_inner = tk.Frame(ctrl_card, bg=config.COLOR_CARD_BG, padx=16, pady=8)
        ctrl_inner.pack(fill="x")

        tk.Label(ctrl_inner, text="Search Students:", font=FONT_BODY_BOLD, bg=config.COLOR_CARD_BG).pack(side="left", padx=(0, 8))
        self.entry_search = ttk.Entry(ctrl_inner, width=28, style="App.TEntry")
        self.entry_search.pack(side="left", padx=(0, 8))
        self.entry_search.bind("<KeyRelease>", lambda e: self.refresh_table())

        ttk.Button(ctrl_inner, text="🔍 Search", style="Primary.TButton", command=self.refresh_table).pack(side="left", padx=4)
        ttk.Button(ctrl_inner, text="🔄 Refresh", style="Secondary.TButton", command=self._on_clear_search).pack(side="left", padx=4)

        # Delete Button & Counter on Right
        ttk.Button(ctrl_inner, text="🗑 Delete Selected", style="Danger.TButton", command=self._on_delete_clicked).pack(side="right", padx=(8, 0))

        self.lbl_count = tk.Label(ctrl_inner, text="Total: 0 students", font=FONT_SMALL, fg=config.COLOR_TEXT_MUTED, bg=config.COLOR_CARD_BG)
        self.lbl_count.pack(side="right", padx=12)

        # Table Panel
        table_card = tk.Frame(content, bg=config.COLOR_CARD_BG, bd=1, relief="solid")
        table_card.config(highlightbackground=config.COLOR_BORDER)
        table_card.pack(fill="both", expand=True, padx=2)

        # Treeview Table
        columns = ("id", "roll_number", "name", "course", "reg_type", "created_at")
        self.tree = ttk.Treeview(table_card, columns=columns, show="headings", selectmode="browse")

        self.tree.heading("id", text="ID")
        self.tree.heading("roll_number", text="Roll Number")
        self.tree.heading("name", text="Student Name")
        self.tree.heading("course", text="Course / Class")
        self.tree.heading("reg_type", text="Registration Method")
        self.tree.heading("created_at", text="Enrolled On")

        self.tree.column("id", width=60, anchor="center")
        self.tree.column("roll_number", width=140, anchor="center")
        self.tree.column("name", width=220, anchor="w")
        self.tree.column("course", width=160, anchor="w")
        self.tree.column("reg_type", width=140, anchor="center")
        self.tree.column("created_at", width=180, anchor="center")

        # Scrollbar
        scrollbar = ttk.Scrollbar(table_card, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=12)
        scrollbar.pack(side="right", fill="y", padx=(0, 12), pady=12)

    def on_show(self) -> None:
        """Invoked when navigating to this screen."""
        self.refresh_table()

    def _on_clear_search(self) -> None:
        """Clear search input and refresh all records."""
        self.entry_search.delete(0, "end")
        self.refresh_table()

    def refresh_table(self) -> None:
        """Fetch records from SQLite and populate the treeview."""
        # Clear existing rows
        for row in self.tree.get_children():
            self.tree.delete(row)

        search_query = self.entry_search.get().strip()
        students = self.db.get_all_students(search_query=search_query)

        for s in students:
            self.tree.insert(
                "",
                "end",
                iid=str(s["id"]),
                values=(
                    s["id"],
                    s["roll_number"],
                    s["name"],
                    s.get("course") or "—",
                    s.get("registration_type") or "WEBCAM",
                    s.get("created_at") or "—"
                )
            )

        self.lbl_count.config(text=f"Total: {len(students)} students")

    def _on_delete_clicked(self) -> None:
        """Delete selected student with confirmation."""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Select Student", "Please select a student from the list to delete.", parent=self)
            return

        student_id = int(selected[0])
        student = self.db.get_student_by_id(student_id)
        if not student:
            return

        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete student:\n\n"
            f"Name: {student['name']}\n"
            f"Roll Number: {student['roll_number']}\n\n"
            f"This will also delete associated reference biometric templates and attendance records.",
            icon="warning",
            parent=self
        )

        if confirm:
            success = self.db.delete_student(student_id)
            if success:
                messagebox.showinfo("Deleted", f"Student '{student['name']}' was successfully deleted.", parent=self)
                self.refresh_table()
            else:
                messagebox.showerror("Error", "Could not delete student record.", parent=self)
