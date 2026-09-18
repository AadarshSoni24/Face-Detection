"""
ui/styles.py - Visual styling, colors, and TTK widget themes for the application.
"""

import tkinter as tk
from tkinter import ttk
import config

FONT_FAMILY = "Segoe UI"
FONT_TITLE = (FONT_FAMILY, 16, "bold")
FONT_HEADER = (FONT_FAMILY, 13, "bold")
FONT_SUBHEADER = (FONT_FAMILY, 11, "bold")
FONT_BODY = (FONT_FAMILY, 10)
FONT_BODY_BOLD = (FONT_FAMILY, 10, "bold")
FONT_SMALL = (FONT_FAMILY, 9)
FONT_CODE = ("Consolas", 10)
FONT_VERDICT = (FONT_FAMILY, 14, "bold")


def apply_theme(root: tk.Tk) -> ttk.Style:
    """Configure modern TTK styling across the application."""
    style = ttk.Style(root)
    # Use 'clam' as base theme because it permits rich color customization
    style.theme_use("clam")

    # Base window background
    root.configure(bg=config.COLOR_BG)

    # Frame styles
    style.configure("App.TFrame", background=config.COLOR_BG)
    style.configure("Card.TFrame", background=config.COLOR_CARD_BG, relief="flat")
    style.configure("Header.TFrame", background=config.COLOR_HEADER_BG)
    style.configure("Sidebar.TFrame", background=config.COLOR_SIDEBAR_BG)

    # Label styles
    style.configure("App.TLabel", background=config.COLOR_BG, foreground=config.COLOR_TEXT_PRIMARY, font=FONT_BODY)
    style.configure("Card.TLabel", background=config.COLOR_CARD_BG, foreground=config.COLOR_TEXT_PRIMARY, font=FONT_BODY)
    style.configure("Title.TLabel", background=config.COLOR_BG, foreground=config.COLOR_TEXT_PRIMARY, font=FONT_TITLE)
    style.configure("Header.TLabel", background=config.COLOR_HEADER_BG, foreground="#FFFFFF", font=FONT_HEADER)
    style.configure("HeaderSub.TLabel", background=config.COLOR_HEADER_BG, foreground="#94A3B8", font=FONT_SMALL)
    style.configure("Muted.TLabel", background=config.COLOR_BG, foreground=config.COLOR_TEXT_MUTED, font=FONT_SMALL)
    style.configure("CardMuted.TLabel", background=config.COLOR_CARD_BG, foreground=config.COLOR_TEXT_MUTED, font=FONT_SMALL)

    # Stat Card styles
    style.configure("StatValue.TLabel", background=config.COLOR_CARD_BG, foreground=config.COLOR_PRIMARY, font=(FONT_FAMILY, 22, "bold"))
    style.configure("StatTitle.TLabel", background=config.COLOR_CARD_BG, foreground=config.COLOR_TEXT_MUTED, font=FONT_SMALL)

    # Button styles
    # Primary (Blue)
    style.configure(
        "Primary.TButton",
        background=config.COLOR_PRIMARY,
        foreground="#FFFFFF",
        font=FONT_BODY_BOLD,
        borderwidth=0,
        focuscolor="none",
        padding=(14, 8)
    )
    style.map(
        "Primary.TButton",
        background=[("active", config.COLOR_PRIMARY_HOVER), ("disabled", "#94A3B8")]
    )

    # Success (Green)
    style.configure(
        "Success.TButton",
        background=config.COLOR_SUCCESS,
        foreground="#FFFFFF",
        font=FONT_BODY_BOLD,
        borderwidth=0,
        focuscolor="none",
        padding=(14, 8)
    )
    style.map(
        "Success.TButton",
        background=[("active", "#047857"), ("disabled", "#94A3B8")]
    )

    # Danger (Red)
    style.configure(
        "Danger.TButton",
        background=config.COLOR_DANGER,
        foreground="#FFFFFF",
        font=FONT_BODY_BOLD,
        borderwidth=0,
        focuscolor="none",
        padding=(14, 8)
    )
    style.map(
        "Danger.TButton",
        background=[("active", "#B91C1C"), ("disabled", "#94A3B8")]
    )

    # Secondary (Slate / Outlined)
    style.configure(
        "Secondary.TButton",
        background="#E2E8F0",
        foreground=config.COLOR_TEXT_PRIMARY,
        font=FONT_BODY_BOLD,
        borderwidth=0,
        focuscolor="none",
        padding=(14, 8)
    )
    style.map(
        "Secondary.TButton",
        background=[("active", "#CBD5E1")]
    )

    # Entry styles
    style.configure("App.TEntry", padding=6, font=FONT_BODY)

    # Treeview / Table styles
    style.configure(
        "Treeview",
        background="#FFFFFF",
        foreground=config.COLOR_TEXT_PRIMARY,
        fieldbackground="#FFFFFF",
        font=FONT_BODY,
        rowheight=28
    )
    style.configure(
        "Treeview.Heading",
        background="#F1F5F9",
        foreground=config.COLOR_TEXT_PRIMARY,
        font=FONT_SUBHEADER,
        padding=6
    )
    style.map(
        "Treeview",
        background=[("selected", "#DBEAFE")],
        foreground=[("selected", "#1E40AF")]
    )

    return style
