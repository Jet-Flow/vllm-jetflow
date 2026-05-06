#!/usr/bin/env python3
"""Compile per-dataset metrics_report.txt from a DFlash tree-mode sweep.

Reads the per-dataset metrics_report.txt files written by dflash_profiling.py
under <sweep_dir>/<dataset>/profile/{ar,dflash}/tp1/bs1/metrics_report.txt
(matching the layout produced by dflash_profiling_qwen3moe_causal_tree_sweep.sh)
and prints a single comparison table with AR vs tree throughput, speedup,
acceptance length, and acceptance rate per dataset.

Usage:
    python examples/offline_inference/summarize_tree_sweep.py <sweep_dir>

If <sweep_dir> is omitted, picks the most recently modified sweep dir under
./logs/qwen3moe_tree_sweep-* (the default LOG_DIR pattern in the sweep script).
"""

import sys
from pathlib import Path


def latest_sweep() -> Path:
    cands = sorted(
        Path("./logs").glob("qwen3moe_tree_sweep-*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        sys.exit(
            "no sweep dir found in ./logs/qwen3moe_tree_sweep-*; pass one explicitly"
        )
    return cands[0]


def load_metrics(report: Path) -> dict:
    if not report.exists():
        return {}
    out = {}
    for line in report.read_text().splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def fnum(d, k, default=float("nan")):
    try:
        return float(d.get(k, default))
    except (TypeError, ValueError):
        return default


def main():
    sweep = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_sweep()
    print(f"sweep dir: {sweep}\n")

    rows = []
    for ds_dir in sorted(p for p in sweep.iterdir() if p.is_dir()):
        ds = ds_dir.name
        prof = ds_dir / "profile"
        ar = load_metrics(prof / "ar" / "tp1" / "bs1" / "metrics_report.txt")
        df = load_metrics(prof / "dflash" / "tp1" / "bs1" / "metrics_report.txt")
        if not df:
            continue

        ar_tps = fnum(ar, "e2e_throughput_tok_s")
        tree_tps = fnum(df, "e2e_throughput_tok_s")
        speedup = tree_tps / ar_tps if ar_tps and ar_tps == ar_tps else float("nan")
        rows.append({
            "dataset": ds,
            "n": df.get("num_samples", "?"),
            "ar_tps": ar_tps,
            "tree_tps": tree_tps,
            "speedup": speedup,
            "acc_len": fnum(df, "acceptance_length"),
            "acc_rate": fnum(df, "acceptance_rate"),
        })

    print(
        f"{'dataset':<14} {'N':>4}  {'AR_tps':>8} {'Tree_tps':>9} {'speedup':>8}  "
        f"{'acc_len':>7} {'acc_rate':>8}"
    )
    print("-" * 68)
    for r in rows:
        print(
            f"{r['dataset']:<14} {str(r['n']):>4}  "
            f"{r['ar_tps']:>8.1f} {r['tree_tps']:>9.1f} {r['speedup']:>7.2f}x  "
            f"{r['acc_len']:>7.2f} {r['acc_rate']:>8.4f}"
        )


if __name__ == "__main__":
    main()
