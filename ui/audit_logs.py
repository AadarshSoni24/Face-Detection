"""
ui/audit_logs.py - Security & Authentication Audit Logs Inspector.

Provides administrators with complete forensic audit capabilities: inspect every
biometric authentication attempt, detect spoofing/tampering events, review cosine
similarities and liveness scores, and export audit reports to CSV.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Callable
import datetime
import csv
import config
from database.db import Database
from ui.styles import (
    FONT_TITLE, FONT_SUBHEADER, FONT_BODY, FONT_SMALL,
    FONT_HEADER, FONT_BODY_BOLD
)


class AuditLogsView(ttk.Frame):
    """Security Audit Log Viewer and Biometric Analytics Screen."""

    def __init__(self, parent: tk.Widget, db: Database, navigate_callback: Callable[[str], None]):
        super().__init__(parent, style="App.TFrame")
        self.db = db
        self.navigate = navigate_callback

        self._build_ui()

    def _build_ui(self) -> None:
        """Construct the audit log viewer interface."""
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
            text="Biometric Security & Authentication Audit Logs",
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

        # Filter Card
        ctrl_card = tk.Frame(content, bg=config.COLOR_CARD_BG, bd=1, relief="solid")
        ctrl_card.config(highlightbackground=config.COLOR_BORDER)
        ctrl_card.pack(fill="x", pady=(0, 16), padx=2, ipady=8)

        ctrl_inner = tk.Frame(ctrl_card, bg=config.COLOR_CARD_BG, padx=16, pady=8)
        ctrl_inner.pack(fill="x")

        # Result Filter Dropdown
        tk.Label(ctrl_inner, text="Result Filter:", font=FONT_BODY_BOLD, bg=config.COLOR_CARD_BG).pack(side="left", padx=(0, 6))
        self.combo_filter = ttk.Combobox(
            ctrl_inner,
            values=["ALL", "SUCCESS_ONLY", "FAILED_ONLY", "SPOOF_ONLY", "NO_FACE", "MULTI_FACE"],
            state="readonly",
            width=14
        )
        self.combo_filter.set("ALL")
        self.combo_filter.pack(side="left", padx=(0, 12))
        self.combo_filter.bind("<<ComboboxSelected>>", lambda e: self.refresh_table())

        # Date Filter
        tk.Label(ctrl_inner, text="Date (YYYY-MM-DD):", font=FONT_BODY_BOLD, bg=config.COLOR_CARD_BG).pack(side="left", padx=(0, 6))
        self.entry_date = ttk.Entry(ctrl_inner, width=12, style="App.TEntry")
        self.entry_date.pack(side="left", padx=(0, 12))

        # Search Query
        tk.Label(ctrl_inner, text="Search Roll/Name:", font=FONT_BODY_BOLD, bg=config.COLOR_CARD_BG).pack(side="left", padx=(0, 6))
        self.entry_search = ttk.Entry(ctrl_inner, width=16, style="App.TEntry")
        self.entry_search.pack(side="left", padx=(0, 12))
        self.entry_search.bind("<KeyRelease>", lambda e: self.refresh_table())

        ttk.Button(ctrl_inner, text="🔍 Filter", style="Primary.TButton", command=self.refresh_table).pack(side="left", padx=4)
        ttk.Button(ctrl_inner, text="🔄 Reset", style="Secondary.TButton", command=self._on_reset_filters).pack(side="left", padx=4)
        ttk.Button(ctrl_inner, text="📥 Export CSV", style="Success.TButton", command=self._on_export_csv).pack(side="left", padx=6)

        self.lbl_count = tk.Label(ctrl_inner, text="Logs: 0 events", font=FONT_SMALL, fg=config.COLOR_TEXT_MUTED, bg=config.COLOR_CARD_BG)
        self.lbl_count.pack(side="right", padx=8)

        # Table Panel
        table_card = tk.Frame(content, bg=config.COLOR_CARD_BG, bd=1, relief="solid")
        table_card.config(highlightbackground=config.COLOR_BORDER)
        table_card.pack(fill="both", expand=True, padx=2)

        columns = ("id", "timestamp", "roll", "name", "session", "result", "sim", "liveness", "details")
        self.tree = ttk.Treeview(table_card, columns=columns, show="headings", selectmode="browse")

        self.tree.heading("id", text="Log ID")
        self.tree.heading("timestamp", text="Timestamp")
        self.tree.heading("roll", text="Roll Number")
        self.tree.heading("name", text="Candidate Name")
        self.tree.heading("session", text="Session Code")
        self.tree.heading("result", text="Verdict Result")
        self.tree.heading("sim", text="Similarity")
        self.tree.heading("liveness", text="Liveness")
        self.tree.heading("details", text="Security Details")

        self.tree.column("id", width=60, anchor="center")
        self.tree.column("timestamp", width=140, anchor="center")
        self.tree.column("roll", width=110, anchor="center")
        self.tree.column("name", width=140, anchor="w")
        self.tree.column("session", width=100, anchor="center")
        self.tree.column("result", width=110, anchor="center")
        self.tree.column("sim", width=90, anchor="center")
        self.tree.column("liveness", width=90, anchor="center")
        self.tree.column("details", width=260, anchor="w")

        scrollbar = ttk.Scrollbar(table_card, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Tags styling
        self.tree.tag_configure("success", foreground="#047857")
        self.tree.tag_configure("failed", foreground="#DC2626")
        self.tree.tag_configure("spoof", foreground="#7C3AED")
        self.tree.tag_configure("warning", foreground="#D97706")

        self.tree.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=12)
        scrollbar.pack(side="right", fill="y", padx=(0, 12), pady=12)

    def on_show(self) -> None:
        """Invoked when navigating to screen."""
        self.refresh_table()

    def _on_reset_filters(self) -> None:
        """Clear filter fields."""
        self.combo_filter.set("ALL")
        self.entry_date.delete(0, "end")
        self.entry_search.delete(0, "end")
        self.refresh_table()

    def refresh_table(self) -> None:
        """Query SQLite and populate audit logs."""
        for row in self.tree.get_children():
            self.tree.delete(row)

        filter_res = self.combo_filter.get()
        date_val = self.entry_date.get().strip()
        search_val = self.entry_search.get().strip()

        logs = self.db.get_audit_logs(
            filter_result=filter_res,
            date_str=date_val if date_val else None,
            search_query=search_val if search_val else None
        )

        for l in logs:
            res = l["result"]
            tag = "success" if res == "SUCCESS" else ("spoof" if "SPOOF" in res else ("warning" if res in ["MULTI_FACE", "NO_FACE"] else "failed"))
            sim_str = f"{l['similarity_score']:.3f}" if l.get("similarity_score") is not None else "—"
            live_str = f"{l['liveness_score']:.2f}" if l.get("liveness_score") is not None else "—"

            self.tree.insert(
                "",
                "end",
                iid=str(l["id"]),
                values=(
                    l["id"],
                    l["timestamp"],
                    l["roll_number"],
                    l.get("student_name") or "—",
                    l.get("session_code") or "—",
                    res,
                    sim_str,
                    live_str,
                    l.get("details") or ""
                ),
                tags=(tag,)
            )

        self.lbl_count.config(text=f"Logs: {len(logs)} events")

    def _on_export_csv(self) -> None:
        """Export displayed audit logs to CSV."""
        rows = []
        for item_id in self.tree.get_children():
            rows.append(self.tree.item(item_id)["values"])

        if not rows:
            messagebox.showinfo("No Logs", "No audit logs to export.", parent=self)
            return

        today = datetime.date.today().strftime("%Y-%m-%d")
        save_path = filedialog.asksaveasfilename(
            parent=self,
            title="Export Security Audit Logs",
            initialfile=f"security_audit_logs_{today}.csv",
            defaultextension=".csv",
            filetypes=[("CSV File", "*.csv"), ("All Files", "*.*")]
        )
        if not save_path:
            return

        try:
            with open(save_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Log ID", "Timestamp", "Roll Number", "Candidate Name", "Session Code", "Result", "Similarity", "Liveness Score", "Details"])
                for r in rows:
                    writer.writerow(r)
            messagebox.showinfo("Export Successful", f"Exported {len(rows)} audit log records to:\n\n{save_path}", parent=self)
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {e}", parent=self)
