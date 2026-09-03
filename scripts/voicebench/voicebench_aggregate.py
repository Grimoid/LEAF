#!/usr/bin/env python
"""Aggregate <method>__<subset>.scored.jsonl files into a LEAF-vs-GRPO markdown table (mean 1-5 score).

    python scripts/voicebench/voicebench_aggregate.py --dir out/voicebench --out out/voicebench/RESULTS.md
"""
import argparse
import glob
import json
import os


def stats(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    sc = [r["score"] for r in rows if r.get("score") is not None]
    return (sum(sc) / len(sc) if sc else None, len(sc), len(rows) - len(sc))


def cell(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory holding <method>__<subset>.scored.jsonl")
    ap.add_argument("--out", default="", help="markdown output (default: <dir>/VOICEBENCH_RESULTS.md)")
    ap.add_argument("--method_a", default="leaf")
    ap.add_argument("--method_b", default="grpo")
    args = ap.parse_args()
    out = args.out or os.path.join(args.dir, "VOICEBENCH_RESULTS.md")

    data = {}  # subset -> method -> (mean, n, fail)
    for f in glob.glob(os.path.join(args.dir, "*__*.scored.jsonl")):
        base = os.path.basename(f)[: -len(".scored.jsonl")]
        method, subset = base.split("__", 1)
        data.setdefault(subset, {})[method] = stats(f)

    a, b = args.method_a, args.method_b
    lines = [
        "# VoiceBench open-ended QA — LEAF vs GRPO",
        "",
        "Judge: M-Prometheus-14B with the VoiceBench open-QA prompt, 1–5 (higher = better).",
        "",
        f"| subset | {a.upper()} | {b.upper()} | Δ ({a.upper()}−{b.upper()}) | n | parse-fails |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for subset in sorted(data):
        m = data[subset]
        am = m.get(a, (None, 0, 0))
        bm = m.get(b, (None, 0, 0))
        delta = (am[0] - bm[0]) if isinstance(am[0], (int, float)) and isinstance(bm[0], (int, float)) else None
        dstr = f"{delta:+.3f}" if isinstance(delta, (int, float)) else "—"
        lines.append(f"| {subset} | {cell(am[0])} | {cell(bm[0])} | {dstr} | {max(am[1], bm[1])} | {am[2] + bm[2]} |")
    others = sorted({mth for d in data.values() for mth in d if mth not in (a, b)})
    if others:
        lines += ["", f"(other methods present: {', '.join(others)} — see per-file scored jsonls)"]
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
