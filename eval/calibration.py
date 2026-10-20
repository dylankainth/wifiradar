"""Is per-wall confidence meaningful? Runs the full pipeline (with bootstrap ensemble) on simulated houses and
bins estimated walls by bootstrap support, reporting the fraction of wall length that lies within 0.5 m of a true wall.

    python -m eval.calibration --seeds 101 102 103 104 105 106 107 108 --out eval/out/calibration
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.sweep import run_cases


def auroc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    if label.all() or (~label).all():
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    for v in np.unique(score):
        m = score == v
        ranks[m] = ranks[m].mean()
    n1, n0 = label.sum(), (~label).sum()
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(101, 109)))
    ap.add_argument("--out", default="eval/out/calibration")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = run_cases([{"seed": s, "wall_detail": True} for s in a.seeds], a.procs)
    (out / "rows.json").write_text(json.dumps(rows, indent=1, default=float))
    det = [d for r in rows if r.get("ok") for d in r.get("walls_detail", [])]
    sup = np.array([d["support"] for d in det]); fc = np.array([d["frac_correct"] for d in det])
    ln = np.array([d["length"] for d in det]); dl = np.array([d["dL"] for d in det])
    good = fc >= 0.7
    lines = [f"{len(det)} estimated walls from {len(rows)} runs; {good.mean():.0%} are 'correct' (>= 70 % of their length within 0.5 m of a true wall)",
             f"AUROC of bootstrap support for 'correct': {auroc(sup, good):.2f};  of leave-one-out evidence dL: {auroc(dl, good):.2f}", "",
             "| support bin | walls | length [m] | mean fraction correct (length-weighted) |", "|---|---|---|---|"]
    bins = [(0, 0.34), (0.34, 0.67), (0.67, 1.01)]
    xs, ys = [], []
    for lo, hi in bins:
        m = (sup >= lo) & (sup < hi)
        if m.any():
            wc = float((fc[m] * ln[m]).sum() / ln[m].sum())
            lines.append(f"| {lo:.2f}-{min(hi, 1):.2f} | {int(m.sum())} | {ln[m].sum():.1f} | {wc:.2f} |")
            xs.append(float(sup[m].mean())); ys.append(wc)
    (out / "calibration.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    ax.scatter(sup + np.random.default_rng(0).normal(0, 0.01, len(sup)), fc, s=10 + 4 * ln, alpha=.5)
    ax.plot(xs, ys, "ro-", label="binned (length-weighted)")
    ax.set_xlabel("bootstrap support"); ax.set_ylabel("fraction of wall within 0.5 m of a true wall")
    ax.legend(fontsize=7); ax.grid(alpha=.3); ax.set_title("wall confidence vs correctness (simulated)", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "calibration.png", dpi=130)


if __name__ == "__main__":
    main()
