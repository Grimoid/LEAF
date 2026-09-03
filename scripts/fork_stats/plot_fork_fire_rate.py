#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PALETTE = ["#0f766e", "#c26d16", "#2563eb", "#7c3aed"]
BG = "#faf7f1"
PANEL = "#fffdf8"
FG = "#2d261d"
MUTED = "#7a6d5c"
GRID = "#dfd6c8"
AXIS = "#6b5f52"


def load_font(size: int, *, bold: bool = False):
    candidates = ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf"] if bold else ["DejaVuSans.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def load_series(spec: str) -> dict:
    if "=" not in spec:
        raise SystemExit(f"Expected --series label=/path/to/json, got: {spec}")
    label, path = spec.split("=", 1)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = payload["position_stats"]
    positions = [int(row["position"]) for row in rows]
    rates = [float(row["fork_fire_rate"]) for row in rows]
    selected = [int(row["selected_count"]) for row in rows]
    usable = [int(row["usable_count"]) for row in rows]

    total_selected = sum(selected)
    total_usable = sum(usable)
    usable_fraction = (total_usable / total_selected) if total_selected else 0.0
    mean_usable_position = (
        sum(pos * use for pos, use in zip(positions, usable)) / total_usable if total_usable else 0.0
    )

    spread_num = 0.0
    spread_den = 0
    for row in rows:
        if "mean_reward_spread" in row:
            spread_num += float(row["mean_reward_spread"]) * int(row["usable_count"])
            spread_den += int(row["usable_count"])
    mean_reward_spread = (spread_num / spread_den) if spread_den else None

    return {
        "label": label,
        "path": path,
        "positions": positions,
        "rates": rates,
        "selected": selected,
        "usable": usable,
        "usable_fraction": usable_fraction,
        "mean_usable_position": mean_usable_position,
        "mean_reward_spread": mean_reward_spread,
    }


def filtered_rows(series: dict, min_selected: int, max_position: int | None) -> list[dict]:
    rows = []
    for pos, rate, sel, use in zip(series["positions"], series["rates"], series["selected"], series["usable"]):
        if sel < min_selected:
            continue
        if max_position is not None and pos > max_position:
            continue
        rows.append({"position": pos, "rate": rate, "selected": sel, "usable": use})
    return rows


def x_coord(position: int, x_min: int, x_max: int, left: float, width: float) -> float:
    span = max(x_max - x_min, 1)
    return left + ((position - x_min) / span) * width


def y_coord(rate: float, top: float, height: float) -> float:
    return top + height - (rate * height)


def render_png(
    series_list: list[dict],
    title: str,
    subtitle: str,
    output_png: Path,
    min_selected: int,
    max_position: int | None,
):
    width = 1600
    height = 980
    plot_left = 120
    plot_top = 150
    plot_width = width - 190
    plot_height = 480
    support_top = 710
    support_height = 120

    title_font = load_font(36, bold=True)
    subtitle_font = load_font(19)
    legend_font = load_font(20)
    axis_font = load_font(22, bold=True)
    tick_font = load_font(18)
    note_font = load_font(16)

    prepared = []
    for series in series_list:
        rows = filtered_rows(series, min_selected, max_position)
        prepared.append({**series, "rows": rows})

    x_values = [row["position"] for series in prepared for row in series["rows"]]
    if not x_values:
        raise SystemExit("No positions remain after filtering; lower --min_selected or increase --max_position.")
    x_min = min(x_values)
    x_max = max(x_values)
    max_selected = max(row["selected"] for series in prepared for row in series["rows"])

    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((28, 24, width - 28, height - 24), radius=28, outline=GRID, width=2, fill=PANEL)

    draw.text((plot_left, 42), title, font=title_font, fill=FG)
    draw.text((plot_left, 86), subtitle, font=subtitle_font, fill=MUTED)

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y_coord(frac, plot_top, plot_height)
        draw.line((plot_left, y, plot_left + plot_width, y), fill=GRID, width=1)
        label = f"{frac:.2f}".rstrip("0").rstrip(".")
        tw, th = _text_size(draw, label, tick_font)
        draw.text((plot_left - tw - 12, y - th / 2), label, font=tick_font, fill=MUTED)

    major_ticks = sorted(set([x for x in range((x_min // 5) * 5, x_max + 1, 5)] + [x_min, x_max]))
    for tick in major_ticks:
        if tick < x_min or tick > x_max:
            continue
        x = x_coord(tick, x_min, x_max, plot_left, plot_width)
        draw.line((x, plot_top, x, support_top + support_height), fill=GRID, width=1)
        label = str(tick)
        tw, _ = _text_size(draw, label, tick_font)
        draw.text((x - tw / 2, plot_top + plot_height + 12), label, font=tick_font, fill=MUTED)

    draw.line((plot_left, plot_top, plot_left, plot_top + plot_height), fill=AXIS, width=3)
    draw.line((plot_left, plot_top + plot_height, plot_left + plot_width, plot_top + plot_height), fill=AXIS, width=3)
    draw.line((plot_left, support_top, plot_left, support_top + support_height), fill=AXIS, width=2)
    draw.line((plot_left, support_top + support_height, plot_left + plot_width, support_top + support_height), fill=AXIS, width=2)

    for idx, series in enumerate(prepared):
        color = PALETTE[idx % len(PALETTE)]
        rows = series["rows"]
        points = [(x_coord(row["position"], x_min, x_max, plot_left, plot_width), y_coord(row["rate"], plot_top, plot_height)) for row in rows]
        if len(points) >= 2:
            draw.line(points, fill=color, width=5, joint="curve")
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=color)

        bar_offset = -6 if idx == 0 else 6
        for row in rows:
            x = x_coord(row["position"], x_min, x_max, plot_left, plot_width) + bar_offset
            bar_h = (row["selected"] / max_selected) * (support_height - 8)
            y0 = support_top + support_height - bar_h
            draw.rectangle((x - 4, y0, x + 4, support_top + support_height), fill=color, outline=color)

    xlabel = "Token position"
    ylabel = "Fork-fire rate"
    tw, _ = _text_size(draw, xlabel, axis_font)
    draw.text((plot_left + plot_width / 2 - tw / 2, height - 62), xlabel, font=axis_font, fill=FG)
    draw.text((32, plot_top - 10), ylabel, font=axis_font, fill=FG)
    draw.text((32, support_top - 10), "Selected\nsupport", font=note_font, fill=MUTED, spacing=2)

    legend_x = plot_left + 20
    legend_y = 112
    for idx, series in enumerate(prepared):
        color = PALETTE[idx % len(PALETTE)]
        y = legend_y + idx * 44
        draw.line((legend_x, y + 10, legend_x + 28, y + 10), fill=color, width=5)
        draw.ellipse((legend_x + 10, y + 5, legend_x + 20, y + 15), fill=color, outline=color)
        spread_text = (
            f", spread={series['mean_reward_spread']:.03f}" if series["mean_reward_spread"] is not None else ""
        )
        label = (
            f"{series['label']}  usable={series['usable_fraction']:.3f}, "
            f"mean usable pos={series['mean_usable_position']:.1f}{spread_text}"
        )
        draw.text((legend_x + 40, y), label, font=legend_font, fill=FG)

    note = "Support bars are normalized within this figure; sparse tail positions are filtered to reduce visual noise."
    draw.text((plot_left, height - 96), note, font=note_font, fill=MUTED)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png)


def render_svg(
    series_list: list[dict],
    title: str,
    subtitle: str,
    output_svg: Path,
    min_selected: int,
    max_position: int | None,
):
    width = 1600
    height = 980
    plot_left = 120
    plot_top = 150
    plot_width = width - 190
    plot_height = 480
    support_top = 710
    support_height = 120

    prepared = []
    for series in series_list:
        rows = filtered_rows(series, min_selected, max_position)
        prepared.append({**series, "rows": rows})

    x_values = [row["position"] for series in prepared for row in series["rows"]]
    if not x_values:
        raise SystemExit("No positions remain after filtering; lower --min_selected or increase --max_position.")
    x_min = min(x_values)
    x_max = max(x_values)
    max_selected = max(row["selected"] for series in prepared for row in series["rows"])

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="100%" height="100%" fill="{BG}"/>',
        f'<rect x="28" y="24" width="{width - 56}" height="{height - 48}" rx="28" ry="28" fill="{PANEL}" stroke="{GRID}" stroke-width="2"/>',
        f'<text x="{plot_left}" y="72" font-family="DejaVu Sans, Arial, sans-serif" font-size="36" font-weight="700" fill="{FG}">{html.escape(title)}</text>',
    ]

    lines.append(
        f'<text x="{plot_left}" y="102" font-family="DejaVu Sans, Arial, sans-serif" font-size="19" fill="{MUTED}">{html.escape(subtitle)}</text>'
    )

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y_coord(frac, plot_top, plot_height)
        label = f"{frac:.2f}".rstrip("0").rstrip(".")
        lines.append(f'<line x1="{plot_left}" y1="{y}" x2="{plot_left + plot_width}" y2="{y}" stroke="{GRID}" stroke-width="1"/>')
        lines.append(
            f'<text x="{plot_left - 14}" y="{y + 6}" text-anchor="end" font-family="DejaVu Sans, Arial, sans-serif" font-size="18" fill="{MUTED}">{label}</text>'
        )

    major_ticks = sorted(set([x for x in range((x_min // 5) * 5, x_max + 1, 5)] + [x_min, x_max]))
    for tick in major_ticks:
        if tick < x_min or tick > x_max:
            continue
        x = x_coord(tick, x_min, x_max, plot_left, plot_width)
        lines.append(f'<line x1="{x}" y1="{plot_top}" x2="{x}" y2="{support_top + support_height}" stroke="{GRID}" stroke-width="1"/>')
        lines.append(
            f'<text x="{x}" y="{plot_top + plot_height + 30}" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="18" fill="{MUTED}">{tick}</text>'
        )

    lines.append(f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_height}" stroke="{AXIS}" stroke-width="3"/>')
    lines.append(f'<line x1="{plot_left}" y1="{plot_top + plot_height}" x2="{plot_left + plot_width}" y2="{plot_top + plot_height}" stroke="{AXIS}" stroke-width="3"/>')
    lines.append(f'<line x1="{plot_left}" y1="{support_top}" x2="{plot_left}" y2="{support_top + support_height}" stroke="{AXIS}" stroke-width="2"/>')
    lines.append(f'<line x1="{plot_left}" y1="{support_top + support_height}" x2="{plot_left + plot_width}" y2="{support_top + support_height}" stroke="{AXIS}" stroke-width="2"/>')

    for idx, series in enumerate(prepared):
        color = PALETTE[idx % len(PALETTE)]
        rows = series["rows"]
        points = [(x_coord(row["position"], x_min, x_max, plot_left, plot_width), y_coord(row["rate"], plot_top, plot_height)) for row in rows]
        if points:
            point_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            lines.append(
                f'<polyline points="{point_str}" fill="none" stroke="{color}" stroke-width="5" stroke-linejoin="round" stroke-linecap="round"/>'
            )
            for x, y in points:
                lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}"/>')

        bar_offset = -6 if idx == 0 else 6
        for row in rows:
            x = x_coord(row["position"], x_min, x_max, plot_left, plot_width) + bar_offset
            bar_h = (row["selected"] / max_selected) * (support_height - 8)
            y0 = support_top + support_height - bar_h
            lines.append(
                f'<rect x="{x - 4:.2f}" y="{y0:.2f}" width="8" height="{bar_h:.2f}" fill="{color}" opacity="0.95"/>'
            )

    lines.append(
        f'<text x="{plot_left + plot_width / 2}" y="{height - 42}" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="22" font-weight="700" fill="{FG}">Token position</text>'
    )
    lines.append(
        f'<text x="32" y="{plot_top - 8}" font-family="DejaVu Sans, Arial, sans-serif" font-size="22" font-weight="700" fill="{FG}">Fork-fire rate</text>'
    )
    lines.append(
        f'<text x="32" y="{support_top - 8}" font-family="DejaVu Sans, Arial, sans-serif" font-size="16" fill="{MUTED}">Selected support</text>'
    )

    legend_x = plot_left + 20
    legend_y = 126
    for idx, series in enumerate(prepared):
        color = PALETTE[idx % len(PALETTE)]
        y = legend_y + idx * 44
        spread_text = (
            f", spread={series['mean_reward_spread']:.03f}" if series["mean_reward_spread"] is not None else ""
        )
        label = (
            f"{series['label']}  usable={series['usable_fraction']:.3f}, "
            f"mean usable pos={series['mean_usable_position']:.1f}{spread_text}"
        )
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="5" stroke-linecap="round"/>')
        lines.append(f'<circle cx="{legend_x + 14}" cy="{y}" r="5" fill="{color}"/>')
        lines.append(
            f'<text x="{legend_x + 40}" y="{y + 6}" font-family="DejaVu Sans, Arial, sans-serif" font-size="20" fill="{FG}">{html.escape(label)}</text>'
        )

    note = "Support bars are normalized within this figure; sparse tail positions are filtered to reduce visual noise."
    lines.append(
        f'<text x="{plot_left}" y="{height - 76}" font-family="DejaVu Sans, Arial, sans-serif" font-size="16" fill="{MUTED}">{html.escape(note)}</text>'
    )
    lines.append("</svg>")

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_svg.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", action="append", required=True, help="Repeated argument of the form label=/path/to/stats.json")
    parser.add_argument("--output_png", type=str, required=True)
    parser.add_argument("--output_svg", type=str, default="")
    parser.add_argument("--title", type=str, default="Fork-Fire Rate Vs Token Position")
    parser.add_argument("--subtitle", type=str, default="")
    parser.add_argument("--min_selected", type=int, default=3)
    parser.add_argument("--max_position", type=int, default=0, help="If > 0, clamp the x-axis to this position.")
    args = parser.parse_args()

    series_list = [load_series(spec) for spec in args.series]
    max_position = args.max_position if args.max_position > 0 else None
    subtitle = args.subtitle or (
        f"Checkpoint-based fork-fire rate. Showing positions with at least {args.min_selected} selected forks"
        + (", full observed x-range." if max_position is None else f", up to token {max_position}")
        + " Bottom bars show position support."
    )

    output_png = Path(args.output_png)
    render_png(series_list, args.title, subtitle, output_png, args.min_selected, max_position)
    if args.output_svg:
        render_svg(series_list, args.title, subtitle, Path(args.output_svg), args.min_selected, max_position)


if __name__ == "__main__":
    main()
