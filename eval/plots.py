"""Intermediate-result plots for eval/out/: trajectory, AP guesses, walls (with confidence), theta scan."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.evaluate import Evaluation, _marker_positions
from eval.metrics import apply
from eval.truth import GroundTruth
from solver.pipeline import Result

GT_COLOR = "#999999"


def _limits(ax, gt, pad=3.0):
    """Zoom on the building: far-away neighbour APs would otherwise squash the plan."""
    if gt is not None and gt.bounds is not None:
        b = gt.bounds
        ax.set_xlim(b[0] - pad, b[2] + pad)
        ax.set_ylim(b[1] - pad, b[3] + pad)


def _gt_walls(ax, gt: GroundTruth | None, obs=None):
    if gt is None:
        return
    for k, w in enumerate(gt.walls):
        faint = obs is not None and not obs[k]
        ax.plot([w[0], w[2]], [w[1], w[3]], color=GT_COLOR, lw=5, alpha=0.18 if faint else 0.45, solid_capstyle="butt",
                label="truth wall (faint = not observable)" if k == 0 else None)


def plot_all(res: Result, ev: Evaluation | None, gt: GroundTruth | None, out_dir: str | Path, prefix: str = ""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    R, t = (ev.R, ev.t) if ev else (np.eye(2), np.zeros(2))
    Rp, tp = (ev.R_pdr, ev.t_pdr) if ev else (np.eye(2), np.zeros(2))
    A = lambda xy: apply(R, t, xy)
    paths = {}

    # 1. trajectory
    fig, ax = plt.subplots(figsize=(7.5, 6))
    _gt_walls(ax, gt)
    if gt is not None and gt.traj is not None:
        ax.plot(*gt.traj[1].T, "-", color="k", lw=1, alpha=.5, label="truth")
    ax.plot(*apply(Rp, tp, res.pg.nodes.pdr_xy).T, ".-", color="#d95f02", ms=2, lw=.7, alpha=.8, label="PDR only")
    ax.plot(*A(res.pg.P).T, ".-", color="#1b9e77", ms=2, lw=.9, label="pose graph + loops")
    for a, b, _ in res.pg.loops_kept:
        p, q = A(res.pg.P[[a, b]])
        ax.plot([p[0], q[0]], [p[1], q[1]], "-", color="#7570b3", lw=.3, alpha=.5)
    if gt is not None:
        for m, p in gt.markers.items():
            ax.plot(*p, "ks", ms=4)
    ax.set_aspect("equal"); ax.legend(fontsize=7, loc="best"); ax.grid(alpha=.2)
    ttl = "trajectory"
    if ev:
        ttl += f"  (marker RMSE: PDR {ev.metrics['marker_rmse_pdr_only_m']:.2f} m -> graph {ev.metrics['marker_rmse_m']:.2f} m; purple = fingerprint loop closures)"
    ax.set_title(ttl, fontsize=8)
    paths["trajectory"] = out_dir / f"{prefix}trajectory.png"
    fig.tight_layout(); fig.savefig(paths["trajectory"], dpi=130); plt.close(fig)

    # 2. APs
    fig, ax = plt.subplots(figsize=(7.5, 6))
    _gt_walls(ax, gt)
    ax.plot(*A(res.ds.node_xy).T, ".", color="#bbb", ms=3)
    if gt is not None:
        for a in gt.aps:
            ax.plot(*a.xy, "^", color="k", ms=8, mfc="none")
            ax.annotate(a.name, a.xy, fontsize=6, xytext=(4, 4), textcoords="offset points")
    ax.plot(*A(res.aps0.xy).T, "x", color="#d95f02", ms=7, label="multilateration (walls ignored)")
    if res.fit is not None:
        ax.plot(*A(res.fit.ap_xy).T, "o", color="#1b9e77", ms=5, label="joint refinement")
    ax.set_aspect("equal"); ax.legend(fontsize=7); ax.grid(alpha=.2); _limits(ax, gt, 6.0)
    if ev:
        ax.set_title(f"AP positions (inside-house mean error: {ev.metrics['ap_err_multilat_inside_m']:.2f} m -> "
                     f"{ev.metrics['ap_err_joint_inside_m']:.2f} m)", fontsize=8)
    paths["aps"] = out_dir / f"{prefix}aps.png"
    fig.tight_layout(); fig.savefig(paths["aps"], dpi=130); plt.close(fig)

    # 3. walls
    fig, ax = plt.subplots(figsize=(7.5, 6))
    _gt_walls(ax, gt, ev.obs_mask if ev else None)
    ax.plot(*A(res.pg.P).T, ".", color="#bbb", ms=3)
    if res.fit is not None:
        cm = plt.get_cmap("viridis")
        for k, w in enumerate(res.fit.walls):
            p, q = A(w[:2][None])[0], A(w[2:][None])[0]
            c = float(res.support[k])
            ax.plot([p[0], q[0]], [p[1], q[1]], "-", color=cm(c), lw=2 + 3 * c, solid_capstyle="butt")
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 1))
        fig.colorbar(sm, ax=ax, label="wall confidence (bootstrap support)", shrink=.7)
    ax.plot(*A(res.fit.ap_xy).T if res.fit is not None else A(res.aps0.xy).T, "^", color="#d62728", ms=6)
    ax.set_aspect("equal"); ax.grid(alpha=.2); _limits(ax, gt)
    if ev and "walls@0.5" in ev.metrics:
        m = ev.metrics["walls@0.5"]
        ax.set_title(f"estimated walls  (@0.5 m: precision {m['precision']:.2f}, recall(obs) {m['recall']:.2f}, "
                     f"F1(obs) {m['f1']:.2f}, F1(all) {m.get('f1_all', m['f1']):.2f})", fontsize=8)
    paths["walls"] = out_dir / f"{prefix}walls.png"
    fig.tight_layout(); fig.savefig(paths["walls"], dpi=130); plt.close(fig)

    # 4. theta scan
    fig, ax = plt.subplots(figsize=(6, 3))
    th, ob = zip(*res.theta_scan)
    ax.plot(th, np.array(ob) - min(ob), "o-", ms=3)
    for c in res.candidates:
        ax.axvline(c[0], color="#999", lw=.7)
        ax.annotate(f"{c[2]}w\n{c[1]:.0f}", (c[0], 0), fontsize=6, xytext=(2, 20), textcoords="offset points")
    if ev and "theta_err_deg" in ev.metrics and res.fit is not None:
        ax.axvline((np.rad2deg(res.fit.theta)) % 90, color="r", lw=1.5, label=f"chosen (err {ev.metrics['theta_err_deg']:.1f} deg)")
        ax.legend(fontsize=7)
    ax.set_xlabel("Manhattan angle [deg, solver frame]"); ax.set_ylabel("greedy-selection objective - min")
    ax.set_title("Manhattan-angle search (grey: candidates refined by the joint fit)", fontsize=8)
    paths["theta"] = out_dir / f"{prefix}theta_scan.png"
    fig.tight_layout(); fig.savefig(paths["theta"], dpi=130); plt.close(fig)
    return paths
