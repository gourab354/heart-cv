"""
report_generator.py - 1-Minute Biometric PDF Report Generator.
Heart Vision | Remote Photoplethysmography

Generates a professional clinical-style PDF report from the last 60 seconds of
rPPG session data. Includes:
  - BPM timeline with confidence envelope and HR-zone colour coding
  - HRV (RMSSD) trend chart
  - Signal quality / confidence chart
  - IBI scatter chart
  - SNR bar chart
  - Full summary statistics table with normal-range assessment
  - Medical disclaimer

Usage:
    from report_generator import generate_report
    path = generate_report(records, output_dir="reports")
"""

import datetime
import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np


# ── Colour palette (professional white theme, hex) ───────────────────────────
BG     = "#FFFFFF"   # page white
PANEL  = "#F5F7FA"   # very light grey for chart backgrounds
CYAN   = "#0099CC"   # deep teal-blue
GREEN  = "#2E8B57"   # professional green
AMBER  = "#D97706"   # warm amber
RED    = "#DC2626"   # clear red
VIOLET = "#7C3AED"   # indigo-violet
WHITE  = "#1A1A2E"   # near-black for text (on white bg)
MUTED  = "#6B7280"   # medium grey for labels/muted text


def _hr_zone(bpm: float):
    """Return (label, colour) for a given heart rate."""
    if bpm < 60:
        return "Bradycardia", CYAN
    elif bpm <= 100:
        return "Normal", GREEN
    elif bpm <= 130:
        return "Elevated", AMBER
    else:
        return "Tachycardia", RED


def _assess(val, lo, hi):
    """Return (text, colour) based on whether val is inside [lo, hi]."""
    if val <= 0:
        return "N/A", MUTED
    if lo <= val <= hi:
        return "Normal  ✓", GREEN
    elif val < lo:
        return "Below range", CYAN
    else:
        return "Above range", AMBER


def generate_report(
    records: List[dict],
    output_dir: str = "reports",
    algorithm: str = "POS",
    session_start: Optional[datetime.datetime] = None,
) -> str:
    """
    Generate a PDF report from the last-60-seconds record list.

    Each record dict must have:
        t          – float, seconds since session start
        bpm        – float, smoothed BPM
        confidence – float, 0-100
        hrv_rmssd  – float, ms
        ibi_ms     – float, ms
        snr_db     – float, dB

    Returns the absolute path to the saved PDF.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not records:
        raise ValueError("No data to generate report.")

    if session_start is None:
        session_start = datetime.datetime.now()

    ts_label = session_start.strftime("%Y-%m-%d %H:%M:%S")
    file_ts  = session_start.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"heart_report_{file_ts}.pdf")

    # ── Extract arrays ─────────────────────────────────────────────────────
    t_arr    = np.array([r["t"]          for r in records])
    bpm_arr  = np.array([r["bpm"]        for r in records])
    conf_arr = np.array([r["confidence"] for r in records])
    hrv_arr  = np.array([r["hrv_rmssd"]  for r in records])
    ibi_arr  = np.array([r["ibi_ms"]     for r in records])
    snr_arr  = np.array([r["snr_db"]     for r in records])

    valid     = bpm_arr > 0
    bpm_valid = bpm_arr[valid]

    avg_bpm = float(np.mean(bpm_valid))                   if len(bpm_valid) else 0.0
    min_bpm = float(np.min(bpm_valid))                    if len(bpm_valid) else 0.0
    max_bpm = float(np.max(bpm_valid))                    if len(bpm_valid) else 0.0
    std_bpm = float(np.std(bpm_valid))                    if len(bpm_valid) else 0.0
    avg_hrv = float(np.mean(hrv_arr[hrv_arr > 0]))        if (hrv_arr > 0).any() else 0.0
    avg_ibi = float(np.mean(ibi_arr[(ibi_arr > 0) & (ibi_arr < 1600)])) \
              if ((ibi_arr > 0) & (ibi_arr < 1600)).any() else 0.0
    avg_snr  = float(np.mean(snr_arr))
    avg_conf = float(np.mean(conf_arr))
    duration = float(t_arr[-1] - t_arr[0]) if len(t_arr) > 1 else 0.0
    zone_label, zone_color = _hr_zone(avg_bpm)

    # ── Figure ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 15), facecolor=BG)
    gs  = gridspec.GridSpec(
        5, 2,
        figure=fig,
        left=0.08, right=0.95,
        top=0.93, bottom=0.04,
        hspace=0.55, wspace=0.35,
    )

    # Header
    fig.text(0.5, 0.965,
             "HEART VISION  ·  Remote Photoplethysmography Report",
             ha="center", fontsize=16, fontweight="bold",
             color=WHITE, fontfamily="monospace")
    fig.text(0.5, 0.950,
             f"Session: {ts_label}    |    Algorithm: {algorithm}    |    Duration: {duration:.1f} s",
             ha="center", fontsize=9, color=MUTED, fontfamily="monospace")
    fig.add_artist(
        plt.Line2D([0.06, 0.94], [0.945, 0.945],
                   transform=fig.transFigure, color=MUTED, linewidth=0.5)
    )

    # ── Chart 1: BPM Timeline (full width) ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(PANEL)
    for sp in ax1.spines.values():
        sp.set_color(MUTED)
    ax1.tick_params(colors=MUTED, labelsize=8)

    if len(t_arr) > 1:
        conf_norm = conf_arr / 100.0
        ax1.fill_between(t_arr, bpm_arr - conf_norm * 3, bpm_arr + conf_norm * 3,
                         alpha=0.18, color=GREEN, linewidth=0)
        pts  = np.array([t_arr, bpm_arr]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        seg_colors = [_hr_zone(b)[1] for b in bpm_arr[:-1]]
        lc = LineCollection(segs, colors=seg_colors, linewidths=2, zorder=3)
        ax1.add_collection(lc)
        if avg_bpm > 0:
            ax1.axhline(avg_bpm, color=AMBER, linewidth=1.0, linestyle="--", alpha=0.8)
            ax1.text(t_arr[-1], avg_bpm + 0.8, f"  avg {avg_bpm:.1f}",
                     color=AMBER, fontsize=7, va="bottom")
        ax1.set_xlim(t_arr[0], t_arr[-1])
        ax1.set_ylim(max(30, min_bpm - 12), min(200, max_bpm + 12))

    ax1.set_title("Heart Rate Timeline (BPM)", color=WHITE, fontsize=10, pad=6)
    ax1.set_xlabel("Time (s)", color=MUTED, fontsize=8)
    ax1.set_ylabel("BPM", color=MUTED, fontsize=8)
    ax1.grid(True, color=MUTED, alpha=0.15, linewidth=0.5)
    patches = [
        mpatches.Patch(color=CYAN,  label="Bradycardia (<60)"),
        mpatches.Patch(color=GREEN, label="Normal (60-100)"),
        mpatches.Patch(color=AMBER, label="Elevated (101-130)"),
        mpatches.Patch(color=RED,   label="Tachycardia (>130)"),
    ]
    ax1.legend(handles=patches, loc="upper right", fontsize=7,
               facecolor=PANEL, edgecolor=MUTED, labelcolor=WHITE, ncol=4)

    # ── Chart 2: HRV Trend (row 1, left) ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(PANEL)
    for sp in ax2.spines.values(): sp.set_color(MUTED)
    ax2.tick_params(colors=MUTED, labelsize=8)
    hrv_mask = hrv_arr > 0
    if hrv_mask.any():
        ax2.plot(t_arr[hrv_mask], hrv_arr[hrv_mask], color=VIOLET, linewidth=1.5)
        ax2.fill_between(t_arr[hrv_mask], 0, hrv_arr[hrv_mask], alpha=0.15, color=VIOLET)
        if avg_hrv > 0:
            ax2.axhline(avg_hrv, color=AMBER, linewidth=1.0, linestyle="--", alpha=0.8)
    else:
        ax2.text(0.5, 0.5, "HRV data not available", ha="center", va="center",
                 color=MUTED, fontsize=9, transform=ax2.transAxes)
    ax2.set_title("HRV — RMSSD (ms)", color=WHITE, fontsize=9, pad=5)
    ax2.set_xlabel("Time (s)", color=MUTED, fontsize=7)
    ax2.set_ylabel("RMSSD (ms)", color=MUTED, fontsize=7)
    ax2.grid(True, color=MUTED, alpha=0.15, linewidth=0.5)

    # ── Chart 3: Signal Quality (row 1, right) ───────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(PANEL)
    for sp in ax3.spines.values(): sp.set_color(MUTED)
    ax3.tick_params(colors=MUTED, labelsize=8)
    ax3.plot(t_arr, conf_arr, color=CYAN, linewidth=1.5)
    ax3.fill_between(t_arr, 0, conf_arr, alpha=0.15, color=CYAN)
    ax3.axhline(60, color=GREEN, linewidth=0.8, linestyle=":", alpha=0.7, label="Good (60%)")
    ax3.axhline(30, color=AMBER, linewidth=0.8, linestyle=":", alpha=0.7, label="Fair (30%)")
    ax3.set_ylim(0, 105)
    ax3.set_title("Signal Quality / Confidence (%)", color=WHITE, fontsize=9, pad=5)
    ax3.set_xlabel("Time (s)", color=MUTED, fontsize=7)
    ax3.set_ylabel("Confidence (%)", color=MUTED, fontsize=7)
    ax3.grid(True, color=MUTED, alpha=0.15, linewidth=0.5)
    ax3.legend(fontsize=7, facecolor=PANEL, edgecolor=MUTED, labelcolor=WHITE)

    # ── Chart 4: IBI Scatter (row 2, left) ──────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor(PANEL)
    for sp in ax4.spines.values(): sp.set_color(MUTED)
    ax4.tick_params(colors=MUTED, labelsize=8)
    ibi_mask = (ibi_arr > 0) & (ibi_arr < 1600)
    if ibi_mask.any():
        ax4.scatter(t_arr[ibi_mask], ibi_arr[ibi_mask], color=AMBER, s=12, alpha=0.75)
        if avg_ibi > 0:
            ax4.axhline(avg_ibi, color=AMBER, linewidth=1.0, linestyle="--", alpha=0.8)
    else:
        ax4.text(0.5, 0.5, "IBI data not available", ha="center", va="center",
                 color=MUTED, fontsize=9, transform=ax4.transAxes)
    ax4.set_title("Inter-Beat Interval (ms)", color=WHITE, fontsize=9, pad=5)
    ax4.set_xlabel("Time (s)", color=MUTED, fontsize=7)
    ax4.set_ylabel("IBI (ms)", color=MUTED, fontsize=7)
    ax4.grid(True, color=MUTED, alpha=0.15, linewidth=0.5)

    # ── Chart 5: SNR Bars (row 2, right) ────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(PANEL)
    for sp in ax5.spines.values(): sp.set_color(MUTED)
    ax5.tick_params(colors=MUTED, labelsize=8)
    bar_w = max(0.4, duration / max(len(snr_arr), 1))
    bar_colors = [GREEN if s >= 4 else AMBER if s >= 0 else RED for s in snr_arr]
    ax5.bar(t_arr, snr_arr, color=bar_colors, width=bar_w, alpha=0.75)
    ax5.axhline(4.0, color=GREEN, linewidth=0.8, linestyle=":", alpha=0.7, label="Good (4 dB)")
    ax5.axhline(0.0, color=AMBER, linewidth=0.8, linestyle=":", alpha=0.7, label="Fair (0 dB)")
    ax5.set_title("Signal-to-Noise Ratio (dB)", color=WHITE, fontsize=9, pad=5)
    ax5.set_xlabel("Time (s)", color=MUTED, fontsize=7)
    ax5.set_ylabel("SNR (dB)", color=MUTED, fontsize=7)
    ax5.grid(True, color=MUTED, alpha=0.15, linewidth=0.5)
    ax5.legend(fontsize=7, facecolor=PANEL, edgecolor=MUTED, labelcolor=WHITE)

    # ── Summary Table (row 3, full width) ───────────────────────────────────
    ax_tbl = fig.add_subplot(gs[3, :])
    ax_tbl.axis("off")

    tbl_rows = [
        ["Avg Heart Rate",      f"{avg_bpm:.1f} BPM",  "60 – 100 BPM",  _assess(avg_bpm, 60, 100)],
        ["Min Heart Rate",      f"{min_bpm:.1f} BPM",  "—",             ("—", MUTED)],
        ["Max Heart Rate",      f"{max_bpm:.1f} BPM",  "—",             ("—", MUTED)],
        ["HR Variability (SD)", f"{std_bpm:.1f} BPM",  "< 10 BPM",      _assess(std_bpm, 0, 10)],
        ["HRV RMSSD",           f"{avg_hrv:.1f} ms" if avg_hrv > 0 else "—",
                                                        "20 – 80 ms",    _assess(avg_hrv, 20, 80) if avg_hrv > 0 else ("N/A", MUTED)],
        ["Avg IBI",             f"{avg_ibi:.0f} ms" if avg_ibi > 0 else "—",
                                                        "600 – 1000 ms", _assess(avg_ibi, 600, 1000) if avg_ibi > 0 else ("N/A", MUTED)],
        ["Avg Signal Quality",  f"{avg_conf:.1f}%",    "> 60%",         _assess(avg_conf, 60, 100)],
        ["Avg SNR",             f"{avg_snr:.1f} dB",   "> 4 dB",        _assess(avg_snr, 4, 20)],
        ["HR Zone",             zone_label,            "Normal",        (zone_label, zone_color)],
        ["Recording Duration",  f"{duration:.1f} s",   "≥ 60 s",
         ("Complete ✓", GREEN) if duration >= 55 else ("Partial", AMBER)],
    ]

    cell_text   = [[r[0], r[1], r[2], r[3][0]] for r in tbl_rows]
    tbl = ax_tbl.table(
        cellText=cell_text,
        colLabels=["Metric", "Value", "Normal Range", "Assessment"],
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    for (ri, ci), cell in tbl.get_celld().items():
        cell.set_facecolor("#E8EDF5" if ri == 0 else PANEL)
        cell.set_edgecolor("#D1D5DB")
        if ri == 0:
            cell.get_text().set_color("#1A1A2E")
            cell.get_text().set_fontweight("bold")
        elif ci == 3:
            cell.get_text().set_color(tbl_rows[ri - 1][3][1])
        else:
            cell.get_text().set_color("#1A1A2E" if ci == 1 else MUTED)

    ax_tbl.set_title("Summary Statistics", color=WHITE, fontsize=10, pad=8)

    # ── Disclaimer (row 4, full width) ──────────────────────────────────────
    ax_disc = fig.add_subplot(gs[4, :])
    ax_disc.axis("off")
    disclaimer = (
        "⚠  MEDICAL DISCLAIMER:  This report is generated from remote photoplethysmography (rPPG) — a research-grade\n"
        "optical technique. Results are NOT a substitute for a certified medical diagnosis or clinical-grade ECG.\n"
        "For any health concerns, please consult a qualified healthcare professional."
    )
    ax_disc.text(0.5, 0.5, disclaimer, ha="center", va="center",
                 fontsize=7.5, color=MUTED, style="italic",
                 transform=ax_disc.transAxes,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF8E7",
                           edgecolor="#D97706", alpha=0.9))

    # ── Save ────────────────────────────────────────────────────────────────
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(out_path)
