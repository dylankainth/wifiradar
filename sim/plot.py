from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MAT_COLOR = {"brick": "#7a3b2e", "drywall": "#8f9aa3", "concrete": "#333", "glass": "#5bc0de"}


def draw_plan(ax, plan, lw_scale=1.0, alpha=1.0, color=None, label=None):
    for i, w in enumerate(plan["walls"]):
        ax.plot([w["x1"], w["x2"]], [w["y1"], w["y2"]], color=color or MAT_COLOR.get(w["material"], "k"),
                lw=(3.0 if w["material"] != "drywall" else 2.0) * lw_scale, alpha=alpha, solid_capstyle="butt",
                label=label if i == 0 else None)


def plot_truth(sim_dir: str | Path, out_png: str | Path):
    T = json.loads((Path(sim_dir) / "truth.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 6.5))
    draw_plan(ax, T["plan"])
    xy = np.array(T["traj"]["xy"])
    ax.plot(xy[:, 0], xy[:, 1], "-", color="#1f77b4", lw=1, alpha=.7, label="walk")
    for a in T["aps"]:
        ax.scatter(*a["xy"], marker="^" if a["mine"] else "v", s=70, color="#d62728" if a["mine"] else "#999", zorder=5)
        ax.annotate(a["name"], a["xy"], xytext=(4, 4), textcoords="offset points", fontsize=7)
    for k, p in T["markers"].items():
        ax.scatter(*p, marker="s", s=25, color="k", zorder=4)
        ax.annotate(k, p, xytext=(3, -9), textcoords="offset points", fontsize=7)
    b = T["plan"]["bounds"]
    ax.set_xlim(b[0] - 3, b[2] + 3)
    ax.set_ylim(b[1] - 3, b[3] + 3)
    ax.set_aspect("equal")
    ax.set_title(f"simulated survey (seed {T['config']['seed']}): plan, APs, walk, markers")
    ax.set_xlabel("x [m] (plan frame)")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
