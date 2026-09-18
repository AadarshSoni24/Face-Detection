"""
ui/import_photos.py - Bulk Student Photo Import UI.

Enables administrators to batch-import pre-existing student photographs from a local
folder with CSV mapping, automated quality checks, and manual-review tagging.
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Callable, Optional
from pathlib import Path
import config
from database.db import Database
from recognition.face_detector import FaceDetector
from recognition.face_encoder import FaceEncoder
from services.import_service import ImportService, ImportSummary, ImportItemResult
from ui.styles import (
    FONT_TITLE, FONT_SUBHEADER, FONT_BODY, FONT_SMALL,
    FONT_HEADER, FONT_BODY_BOLD
)


class ImportPhotosView(ttk.Frame):
    """Bulk Photo Import Screen with progress tracking and categorized reporting."""

    def __init__(
        self,
        parent: tk.Widget,
        db: Database,
        detector: FaceDetector,
        encoder: FaceEncoder,
        navigate_callback: Callable[[str], None]
    ):
        super().__init__(parent, style="App.TFrame")
        self.db = db
        self.detector = detector
        self.encoder = encoder
        self.navigate = navigate_callback

        self.import_service = ImportService(
            db=self.db,
            detector=self.detector,
            encoder=self.encoder
        )

        self._is_importing = False
        self._last_summary: Optional[ImportSummary] = None

        self._build_ui()

    def _build_ui(self) -> None:
        """Construct bulk photo import layout."""
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
            text="Bulk Student Photo & CSV Import",
            font=FONT_HEADER,
            fg="#FFFFFF",
            bg=config.COLOR_HEADER_BG
        ).pack(side="left", padx=20, pady=12)

        # Main Scrollable / Padded Body
        content = tk.Frame(self, bg=config.COLOR_BG)
        content.pack(fill="both", expand=True, padx=24, pady=16)

        # -------------------------------------------------------------
        # Section 1: File & Directory Selection Card
        # -------------------------------------------------------------
        picker_card = tk.Frame(content, bg=config.COLOR_CARD_BG, bd=1, relief="solid")
        picker_card.config(highlightbackground=config.COLOR_BORDER)
        picker_card.pack(fill="x", pady=(0, 14), ipady=6)

        picker_inner = tk.Frame(picker_card, bg=config.COLOR_CARD_BG, padx=16, pady=10)
        picker_inner.pack(fill="x")

        # Row 1: Photos Directory
        row1 = tk.Frame(picker_inner, bg=config.COLOR_CARD_BG)
        row1.pack(fill="x", pady=4)
        tk.Label(row1, text="Photos Folder: *", font=FONT_BODY_BOLD, width=18, anchor="w", bg=config.COLOR_CARD_BG).pack(side="left")
        self.entry_folder = ttk.Entry(row1, style="App.TEntry")
        self.entry_folder.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row1, text="Browse Folder...", style="Secondary.TButton", command=self._browse_folder).pack(side="right")

        # Row 2: CSV Mapping File
        row2 = tk.Frame(picker_inner, bg=config.COLOR_CARD_BG)
        row2.pack(fill="x", pady=4)
        tk.Label(row2, text="Mapping CSV: (opt)", font=FONT_BODY_BOLD, width=18, anchor="w", bg=config.COLOR_CARD_BG).pack(side="left")
        self.entry_csv = ttk.Entry(row2, style="App.TEntry")
        self.entry_csv.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row2, text="Browse CSV...", style="Secondary.TButton", command=self._browse_csv).pack(side="right")

        # Row 3: Action Buttons & Progress Bar
        row3 = tk.Frame(picker_inner, bg=config.COLOR_CARD_BG)
        row3.pack(fill="x", pady=(10, 2))

        self.btn_start = ttk.Button(row3, text="▶ Start Import", style="Primary.TButton", command=self._start_import)
        self.btn_start.pack(side="left", padx=(0, 8))

        self.btn_cancel = ttk.Button(row3, text="⏹ Cancel", style="Secondary.TButton", state="disabled", command=self._cancel_import)
        self.btn_cancel.pack(side="left", padx=(0, 16))

        self.progress_bar = ttk.Progressbar(row3, orient="horizontal", mode="determinate")
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=(0, 12))

        self.lbl_progress_status = tk.Label(row3, text="Ready to scan folder.", font=FONT_SMALL, fg=config.COLOR_TEXT_MUTED, bg=config.COLOR_CARD_BG)
        self.lbl_progress_status.pack(side="right")

        # -------------------------------------------------------------
        # Section 2: Summary Metric Cards (Dynamic Results)
        # -------------------------------------------------------------
        self.stats_box = tk.Frame(content, bg=config.COLOR_BG)
        self.stats_box.pack(fill="x", pady=(0, 14))

        self.card_total = self._create_summary_card(self.stats_box, "Total Files", "0", config.COLOR_TEXT_PRIMARY)
        self.card_total.pack(side="left", fill="both", expand=True, padx=(0, 8))

        self.card_success = self._create_summary_card(self.stats_box, "Processed Successfully", "0", config.COLOR_SUCCESS)
        self.card_success.pack(side="left", fill="both", expand=True, padx=8)

        self.card_review = self._create_summary_card(self.stats_box, "Manual Review Required", "0", config.COLOR_WARNING)
        self.card_review.pack(side="left", fill="both", expand=True, padx=8)

        self.card_failed = self._create_summary_card(self.stats_box, "Import Failed", "0", config.COLOR_DANGER)
        self.card_failed.pack(side="left", fill="both", expand=True, padx=(8, 0))

        # -------------------------------------------------------------
        # Section 3: Itemized Report Table
        # -------------------------------------------------------------
        table_card = tk.Frame(content, bg=config.COLOR_CARD_BG, bd=1, relief="solid")
        table_card.config(highlightbackground=config.COLOR_BORDER)
        table_card.pack(fill="both", expand=True)

        tbl_top = tk.Frame(table_card, bg=config.COLOR_CARD_BG, padx=16, pady=8)
        tbl_top.pack(fill="x")

        tk.Label(tbl_top, text="Itemized Import Report", font=FONT_SUBHEADER, bg=config.COLOR_CARD_BG).pack(side="left")

        # Category Filter Dropdown
        tk.Label(tbl_top, text="Show:", font=FONT_BODY, bg=config.COLOR_CARD_BG).pack(side="left", padx=(20, 6))
        self.combo_filter = ttk.Combobox(
            tbl_top,
            values=["All Outcomes", "Processed Successfully", "Manual Review Required", "Failed"],
            state="readonly",
            width=22
        )
        self.combo_filter.set("All Outcomes")
        self.combo_filter.pack(side="left")
        self.combo_filter.bind("<<ComboboxSelected>>", lambda e: self._filter_report_table())

        # Treeview for Import Report
        columns = ("filename", "roll_number", "student_name", "status", "notes")
        self.tree = ttk.Treeview(table_card, columns=columns, show="headings", selectmode="browse")

        self.tree.heading("filename", text="Image Filename")
        self.tree.heading("roll_number", text="Mapped Roll No")
        self.tree.heading("student_name", text="Student Name")
        self.tree.heading("status", text="Outcome Status")
        self.tree.heading("notes", text="Validation Diagnostic Notes")

        self.tree.column("filename", width=160, anchor="w")
        self.tree.column("roll_number", width=130, anchor="center")
        self.tree.column("student_name", width=180, anchor="w")
        self.tree.column("status", width=160, anchor="center")
        self.tree.column("notes", width=380, anchor="w")

        # Scrollbars
        scrollbar_y = ttk.Scrollbar(table_card, orient="vertical", command=self.tree.yview)
        scrollbar_x = ttk.Scrollbar(table_card, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        self.tree.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=10)
        scrollbar_y.pack(side="right", fill="y", padx=(0, 12), pady=10)

    def _create_summary_card(self, parent: tk.Widget, title: str, default_val: str, color: str) -> tk.Frame:
        """Create summary statistic card."""
        card = tk.Frame(parent, bg=config.COLOR_CARD_BG, bd=1, relief="solid")
        card.config(highlightbackground=config.COLOR_BORDER)
        inner = tk.Frame(card, bg=config.COLOR_CARD_BG, padx=12, pady=10)
        inner.pack(fill="both", expand=True)

        tk.Label(inner, text=title, font=FONT_SMALL, fg=config.COLOR_TEXT_MUTED, bg=config.COLOR_CARD_BG).pack(anchor="w")
        val_lbl = tk.Label(inner, text=default_val, font=(FONT_BODY[0], 20, "bold"), fg=color, bg=config.COLOR_CARD_BG)
        val_lbl.pack(anchor="w", pady=(2, 0))
        card.val_label = val_lbl
        return card

    def on_show(self) -> None:
        """Invoked when navigating to this screen."""
        pass

    def _on_back_clicked(self) -> None:
        """Navigate back to dashboard, warning if import is active."""
        if self._is_importing:
            if not messagebox.askyesno("Cancel Import", "An import is currently running. Cancel and return to dashboard?", parent=self):
                return
            self._cancel_import()
        self.navigate("dashboard")

    def _browse_folder(self) -> None:
        """Select directory containing student photographs."""
        folder = filedialog.askdirectory(parent=self, title="Select College Photos Folder")
        if folder:
            self.entry_folder.delete(0, "end")
            self.entry_folder.insert(0, folder)
            # Check if a mapping.csv or students.csv exists in that folder automatically
            potential_csv = os.path.join(folder, "mapping.csv")
            if not os.path.exists(potential_csv):
                potential_csv = os.path.join(folder, "students.csv")
            if os.path.exists(potential_csv) and not self.entry_csv.get():
                self.entry_csv.delete(0, "end")
                self.entry_csv.insert(0, potential_csv)

    def _browse_csv(self) -> None:
        """Select student mapping CSV file."""
        csv_file = filedialog.askopenfilename(
            parent=self,
            title="Select Student Mapping CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if csv_file:
            self.entry_csv.delete(0, "end")
            self.entry_csv.insert(0, csv_file)

    def _start_import(self) -> None:
        """Begin bulk photo import on a background worker thread."""
        photos_folder = self.entry_folder.get().strip()
        csv_file = self.entry_csv.get().strip()

        if not photos_folder or not os.path.isdir(photos_folder):
            messagebox.showwarning("Invalid Folder", "Please select a valid folder containing student photos.", parent=self)
            return

        if csv_file and not os.path.isfile(csv_file):
            messagebox.showwarning("Invalid CSV", "The specified CSV mapping file does not exist.", parent=self)
            return

        # Reset UI
        self._is_importing = True
        self.btn_start.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self.progress_bar["value"] = 0
        self.lbl_progress_status.config(text="Starting import...")

        # Clear report table
        for row in self.tree.get_children():
            self.tree.delete(row)

        # Reset summary cards
        self.card_total.val_label.config(text="0")
        self.card_success.val_label.config(text="0")
        self.card_review.val_label.config(text="0")
        self.card_failed.val_label.config(text="0")

        # Launch background thread
        worker = threading.Thread(
            target=self._run_import_thread,
            args=(photos_folder, csv_file if csv_file else None),
            daemon=True
        )
        worker.start()

    def _cancel_import(self) -> None:
        """Cancel running import."""
        self.import_service.request_cancel()
        self.lbl_progress_status.config(text="Cancelling import...")
        self.btn_cancel.config(state="disabled")

    def _run_import_thread(self, photos_folder: str, csv_path: Optional[str]) -> None:
        """Background execution of import pipeline."""
        try:
            summary = self.import_service.run_import(
                photos_folder=photos_folder,
                csv_mapping_path=csv_path,
                progress_callback=self._on_progress_update
            )
            self._last_summary = summary
            self.after(0, self._on_import_finished, summary)
        except Exception as e:
            config.logger.error(f"Bulk import error: {e}", exc_info=True)
            self.after(0, lambda: messagebox.showerror("Import Error", f"Import failed with error:\n{e}", parent=self))
            self.after(0, self._reset_import_controls)

    def _on_progress_update(self, current: int, total: int, filename: str) -> None:
        """Thread-safe update of progress bar and status text."""
        pct = (current / total) * 100 if total > 0 else 0
        self.after(0, self._apply_progress, pct, f"Processing ({current}/{total}): {filename}")

    def _apply_progress(self, percent: float, label_text: str) -> None:
        """GUI thread update for progress."""
        self.progress_bar["value"] = percent
        self.lbl_progress_status.config(text=label_text)

    def _on_import_finished(self, summary: ImportSummary) -> None:
        """Called on GUI thread upon bulk import completion."""
        self._reset_import_controls()
        self.progress_bar["value"] = 100
        self.lbl_progress_status.config(
            text=f"Complete! Processed: {summary.processed_successfully}, Review: {summary.needs_review}, Failed: {summary.failed}"
        )

        # Update summary cards with REAL numbers from execution
        self.card_total.val_label.config(text=str(summary.total_files))
        self.card_success.val_label.config(text=str(summary.processed_successfully))
        self.card_review.val_label.config(text=str(summary.needs_review))
        self.card_failed.val_label.config(text=str(summary.failed))

        # Populate report table
        self._filter_report_table()

        # Dialog notification
        msg = (
            f"IMPORT COMPLETE\n\n"
            f"Total Files Scanned: {summary.total_files}\n"
            f"Processed Successfully: {summary.processed_successfully}\n"
            f"Needs Manual Review: {summary.needs_review}\n"
            f"Failed: {summary.failed}\n"
        )
        messagebox.showinfo("Import Complete", msg, parent=self)

    def _reset_import_controls(self) -> None:
        """Re-enable start button and disable cancel."""
        self._is_importing = False
        self.btn_start.config(state="normal")
        self.btn_cancel.config(state="disabled")

    def _filter_report_table(self) -> None:
        """Filter rows in the itemized table based on selected category."""
        if not self._last_summary:
            return

        for row in self.tree.get_children():
            self.tree.delete(row)

        filter_choice = self.combo_filter.get()

        for item in self._last_summary.items:
            # Check filter criteria
            if filter_choice == "Processed Successfully" and item.status != "SUCCESS":
                continue
            if filter_choice == "Manual Review Required" and item.status != "NEEDS_REVIEW":
                continue
            if filter_choice == "Failed" and item.status != "FAILED":
                continue

            # Format status tag
            display_status = (
                "✓ PROCESSED" if item.status == "SUCCESS"
                else "⚠️ NEEDS REVIEW" if item.status == "NEEDS_REVIEW"
                else "✗ FAILED"
            )

            self.tree.insert(
                "",
                "end",
                values=(
                    item.filename,
                    item.roll_number,
                    item.student_name,
                    display_status,
                    item.message
                )
            )
