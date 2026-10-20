"""Overlay a saved result on a ground-truth plan (tape-measured ground_truth.yaml or sim truth.json) and score it.

    python -m eval.overlay --result eval/out/<run>/result.json --truth ground_truth.yaml [--out overlay.png]

Needs result.json from `solver run` (it stores solver-frame walls, APs and the estimated marker positions); the
solver frame is aligned to your frame by a rigid fit through the markers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.metrics import align_from_markers, ap_errors, apply, observable_mask, wall_metrics
from eval.truth import load_truth


def score_saved(res: dict, gt):
    ev: dict[int, list] = {}
    for m in res["markers_est"]:
        ev.setdefault(m["marker"], []).append(np.array(m["xy"]))
    R, t = align_from_markers(ev, gt.markers)
    M = {}
    errs = [np.linalg.norm(apply(R, t, p[None])[0] - gt.markers[m]) for m, v in ev.items() if m in gt.markers for p in v]
    M["marker_rmse_m"] = float(np.sqrt(np.mean(np.square(errs)))) if errs else None
    W = np.array([w["xy"] for w in res["walls"]], float).reshape(-1, 4)
    Wt = np.hstack([apply(R, t, W[:, :2]), apply(R, t, W[:, 2:])]) if len(W) else W
    nodes = apply(R, t, np.array(res["nodes_xy"]))
    om = observable_mask(gt.walls, nodes, np.array([a.xy for a in gt.aps])) if len(gt.aps) else None
    for tol in (0.5, 1.0):
        M[f"walls@{tol}"] = {k: round(v, 3) for k, v in wall_metrics(Wt, gt.walls, tol, observable=om).items()}
    for tag, key in (("multilat", "aps_multilat"), ("joint", "aps")):
        e = ap_errors(res[key], gt, R, t)
        M[f"ap_err_{tag}_m"] = {k: round(v, 2) for k, v in e.items()}
    return M, (R, t), Wt, om


def overlay(res: dict, gt, out_png: Path):
    M, (R, t), Wt, om = score_saved(res, gt)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for k, w in enumerate(gt.walls):
        faint = om is not None and not om[k]
        ax.plot([w[0], w[2]], [w[1], w[3]], color="#888", lw=6, alpha=0.2 if faint else 0.5, solid_capstyle="butt",
                label="ground truth wall (faint: not observable by RSS)" if k == 0 else None)
    cm = plt.get_cmap("viridis")
    for k, (w, wd) in enumerate(zip(Wt, res["walls"])):
        c = wd["confidence"]
        ax.plot([w[0], w[2]], [w[1], w[3]], color=cm(c), lw=2 + 3 * c, solid_capstyle="butt", label="estimated wall (colour = confidence)" if k == 0 else None)
    for a in gt.aps:
        ax.plot(*a.xy, "k^", ms=9, mfc="none", label="true AP" if a is gt.aps[0] else None)
    est = apply(R, t, np.array([a["xy"] for a in res["aps"]]))
    ax.plot(est[:, 0], est[:, 1], "o", color="#d62728", ms=4, label="estimated AP")
    for p in gt.markers.values():
        ax.plot(*p, "ks", ms=4)
    ax.plot(*apply(R, t, np.array(res["nodes_xy"])).T, ".", color="#aaa", ms=2, label="estimated positions")
    if gt.bounds:
        b = gt.bounds
        ax.set_xlim(b[0] - 3, b[2] + 3); ax.set_ylim(b[1] - 3, b[3] + 3)
    w5 = M["walls@0.5"]
    ax.set_title(f"estimate vs ground truth  (marker RMSE {M['marker_rmse_m']:.2f} m; walls @0.5 m: P {w5['precision']:.2f} "
                 f"R(obs) {w5['recall']:.2f} F1(obs) {w5['f1']:.2f}, F1(all) {w5.get('f1_all', w5['f1']):.2f})", fontsize=8)
    ax.set_aspect("equal"); ax.grid(alpha=.2); ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)
    return M


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    res = json.loads(Path(a.result).read_text())
    out = Path(a.out) if a.out else Path(a.result).with_name("overlay.png")
    M = overlay(res, load_truth(a.truth), out)
    print(json.dumps(M, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
