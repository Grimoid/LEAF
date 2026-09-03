#!/usr/bin/env python3
"""Matplotlib figures for the fork-fire study.

Produces two PDFs (and PNGs) suitable for the paper:
  fig_fork_fire_by_task.{pdf,png}        2x2 grid; one panel per task; 3 backbones overlaid.
                                          -> visual argument for *backbone-invariance*.
  fig_fork_fire_by_backbone.{pdf,png}    1x3 grid; one panel per backbone; 4 tasks overlaid.
                                          -> visual argument for *task-monotonicity*.

Improvements over the previous PIL-based plotter:
  - Wilson 95% CI bands per position (so high-variance tail positions visibly widen).
  - min_selected=15 filter to drop positions with too few attempts for any signal.
  - x-axis clipped to position <= MAX_POSITION (most fork activity is early).
  - colour-blind-safe palette, consistent across panels.
  - Vector PDF output (and 300 dpi PNG fallback).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
# ART (input dir with the 12 {backbone}_{task}_fork_fire_stats.json) and the output
# dir are resolved in main() from --input_dir / --out_dir; this is the import-time
# fall-through only.
ART  = HERE.parent


def _default_fork_stats_dir() -> Path:
    """$FORK_STATS_DIR, else the newest subdirectory of $FORK_STATS_ROOT (default
    results/fork_stats) holding the fork-fire JSONs, else this script's parent."""
    env = os.environ.get("FORK_STATS_DIR")
    if env:
        return Path(env)
    root = Path(os.environ.get("FORK_STATS_ROOT", "results/fork_stats"))
    cands = sorted(root.glob("*/granite3_2b_librisqa_fork_fire_stats.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0].parent if cands else HERE.parent

BACKBONES = [
    ("granite4_1b",  "Granite 4.0-1B"),
    ("granite3_2b",  "Granite 3.3-2B"),
    ("granite3_8b",  "Granite 3.3-8B"),
]
TASKS = [
    ("librisqa",                   "LibriSQA"),
    ("dailytalk_longaudio_nomcq",  "DailyTalk"),
    ("covost2",                    "CoVoST2"),
    ("speech_longaudio",           "LongAudio"),
]
# Wong / Okabe colour-blind safe.
BACKBONE_COLORS = {
    "granite4_1b": "#0072B2",
    "granite3_2b": "#D55E00",
    "granite3_8b": "#009E73",
}
TASK_COLORS = {
    "librisqa":                   "#0072B2",
    "dailytalk_longaudio_nomcq":  "#009E73",
    "covost2":                    "#D55E00",
    "speech_longaudio":           "#CC79A7",
}

MIN_SELECTED = 15
MAX_POSITION = 40
CI_ALPHA     = 0.15
LINE_LW      = 3.0
MARKER       = "o"
MARKER_SIZE  = 7.0

# Paper figures are scaled to ~3.5" wide in a double-column layout, so
# fonts rendered here must be large enough to remain legible after
# scaling. The main-text figure (per-task) gets the bigger boost.
FS_MAIN_TITLE  = 24
FS_MAIN_AXIS   = 22
FS_MAIN_TICK   = 20
FS_MAIN_LEGEND = 20

FS_APP_TITLE   = 22
FS_APP_AXIS    = 20
FS_APP_TICK    = 18
FS_APP_LEGEND  = 17


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1.0 + z*z/n
    center = (p + z*z/(2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def load_curve(bb: str, task: str,
               min_selected: int = MIN_SELECTED,
               max_position: int = MAX_POSITION):
    p = ART / f"{bb}_{task}_fork_fire_stats.json"
    d = json.loads(p.read_text())
    rows = []
    for s in d["position_stats"]:
        pos = int(s["position"])
        sel = int(s["selected_count"])
        use = int(s["usable_count"])
        if sel < min_selected:
            continue
        if max_position and pos > max_position:
            continue
        mean, lo, hi = wilson_ci(use, sel)
        rows.append((pos, mean, lo, hi))
    rows.sort()
    if not rows:
        return np.array([]), np.array([]), np.array([]), np.array([])
    pos = np.array([r[0] for r in rows])
    mean = np.array([r[1] for r in rows])
    lo = np.array([r[2] for r in rows])
    hi = np.array([r[3] for r in rows])
    return pos, mean, lo, hi


def _style_axis(ax, tick_fs):
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.5, MAX_POSITION + 0.5)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.tick_params(axis="both", labelsize=tick_fs, width=1.2, length=5)


def fig_per_task_grid(out_stub: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), sharey=True)
    axes_flat = axes.flatten()
    for ax, (task, task_lbl) in zip(axes_flat, TASKS):
        for bb, bb_lbl in BACKBONES:
            pos, mean, lo, hi = load_curve(bb, task)
            if pos.size == 0:
                continue
            color = BACKBONE_COLORS[bb]
            ax.fill_between(pos, lo*100, hi*100, color=color, alpha=CI_ALPHA, linewidth=0)
            ax.plot(pos, mean*100, color=color, lw=LINE_LW,
                    marker=MARKER, markersize=MARKER_SIZE, label=bb_lbl)
        ax.set_title(task_lbl, fontsize=FS_MAIN_TITLE, pad=8)
        _style_axis(ax, FS_MAIN_TICK)
    for ax in axes_flat[2:]:
        ax.set_xlabel("Token position", fontsize=FS_MAIN_AXIS, labelpad=6)
    for ax in (axes_flat[0], axes_flat[2]):
        ax.set_ylabel("Usable fork-fire rate (%)", fontsize=FS_MAIN_AXIS, labelpad=6)
    leg = axes_flat[0].legend(loc="lower left", framealpha=0.9,
                              fontsize=FS_MAIN_LEGEND,
                              title="Backbone",
                              title_fontsize=FS_MAIN_LEGEND,
                              borderpad=0.6, handlelength=2.2, handletextpad=0.6)
    leg.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(out_stub.with_suffix(".pdf"))
    fig.savefig(out_stub.with_suffix(".png"), dpi=300)
    plt.close(fig)


def fig_per_backbone_grid(out_stub: Path):
    # Width bumped to make room for an external legend on the right.
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.4), sharey=True)
    handles, labels = [], []
    for ax, (bb, bb_lbl) in zip(axes, BACKBONES):
        for task, task_lbl in TASKS:
            pos, mean, lo, hi = load_curve(bb, task)
            if pos.size == 0:
                continue
            color = TASK_COLORS[task]
            ax.fill_between(pos, lo*100, hi*100, color=color, alpha=CI_ALPHA, linewidth=0)
            (line,) = ax.plot(pos, mean*100, color=color, lw=LINE_LW,
                              marker=MARKER, markersize=MARKER_SIZE, label=task_lbl)
            if task_lbl not in labels:
                handles.append(line); labels.append(task_lbl)
        ax.set_title(bb_lbl, fontsize=FS_APP_TITLE, pad=8)
        ax.set_xlabel("Token position", fontsize=FS_APP_AXIS, labelpad=6)
        _style_axis(ax, FS_APP_TICK)
    axes[0].set_ylabel("Usable fork-fire rate (%)", fontsize=FS_APP_AXIS, labelpad=6)
    # External legend, vertically centred to the right of the last panel.
    leg = fig.legend(handles, labels, loc="center left",
                     bbox_to_anchor=(0.87, 0.5),
                     framealpha=0.9,
                     fontsize=FS_APP_LEGEND,
                     title="Task",
                     title_fontsize=FS_APP_LEGEND,
                     borderpad=0.6, handlelength=2.2, handletextpad=0.6)
    leg.get_frame().set_linewidth(0.8)
    fig.tight_layout(rect=[0, 0, 0.87, 1])
    fig.savefig(out_stub.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stub.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    global ART
    ap = argparse.ArgumentParser(
        description="Paper figures 2 & 3 (fork-fire by task / by backbone) from the 12 merged JSONs.")
    ap.add_argument("--input_dir", default="",
                    help="dir with {backbone}_{task}_fork_fire_stats.json "
                         "(default: $FORK_STATS_DIR / auto-discover)")
    ap.add_argument("--out_dir", default="",
                    help="output dir for the figures (default: ./fork_paper_outputs)")
    args = ap.parse_args()
    ART = Path(args.input_dir) if args.input_dir else _default_fork_stats_dir()
    out = Path(args.out_dir) if args.out_dir else (Path.cwd() / "fork_paper_outputs")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[plot_fork_fire_clean] input_dir={ART}  out_dir={out}")
    fig_per_task_grid(out / "fig_fork_fire_by_task")
    fig_per_backbone_grid(out / "fig_fork_fire_by_backbone")
    print("Wrote PDFs and PNGs to", out)


if __name__ == "__main__":
    main()
