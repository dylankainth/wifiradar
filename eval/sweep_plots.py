from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {"rss_noise_db": "per-scan RSS noise sigma [dB]", "shadow_sigma_db": "spatially-correlated shadowing sigma [dB]",
          "scan_period_s": "WiFi scan period [s]", "heading_drift_scale": "heading drift (x default)",
          "n_visits": "number of marker stops in the walk"}


def _agg(rows, sweep, key):
    vals = sorted({r["value"] for r in rows if r["sweep"] == sweep})
    mean, sd = [], []
    for v in vals:
        x = [r[key] for r in rows if r["sweep"] == sweep and r["value"] == v and r.get("ok") and r.get(key) is not None]
        mean.append(np.mean(x) if x else np.nan)
        sd.append(np.std(x) if x else np.nan)
    return vals, np.array(mean), np.array(sd)


def plot_sweeps(rows, out_dir: Path):
    out_dir = Path(out_dir)
    sweeps = [s for s in LABELS if any(r["sweep"] == s for r in rows)]
    fig, axes = plt.subplots(3, len(sweeps), figsize=(3.3 * len(sweeps), 8), squeeze=False)
    series = [(("marker_rmse", "trajectory error at markers [m]"), ("marker_rmse_pdr", "PDR only")),
              (("ap_in_joint", "AP error, inside APs [m]"), ("ap_in_multilat", "multilateration only")),
              (("f1", "wall F1 (observable walls, 0.5 m)"), ("precision", "precision"), ("recall", "recall (obs.)"))]
    for j, s in enumerate(sweeps):
        for i, grp in enumerate(series):
            ax = axes[i][j]
            for k, (key, lab) in enumerate(grp):
                v, m, sd = _agg(rows, s, key)
                ax.errorbar(v, m, sd, marker="o", ms=3, capsize=2, label=lab if k else lab, lw=1.5 if k == 0 else 0.8,
                            ls="-" if k == 0 else "--")
            ax.set_xlabel(LABELS[s], fontsize=7)
            if j == 0:
                ax.set_ylabel(grp[0][1], fontsize=7)
            if grp[0][0] == "f1":
                ax.set_ylim(0, 1)
            if s in ("scan_period_s",):
                pass
            ax.tick_params(labelsize=7); ax.grid(alpha=.3)
            if j == len(sweeps) - 1:
                ax.legend(fontsize=6)
    fig.suptitle("Sensitivity sweeps on simulated data (mean +/- sd over seeds; defaults: noise 1.5 dB, shadowing 2.5 dB, scan period 4 s)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "sweeps.png", dpi=130)
    plt.close(fig)
    # markdown table
    lines = ["| sweep | value | n ok | marker RMSE [m] | AP err inside [m] | precision | recall (obs) | F1 (obs) | F1 (all) |", "|---|---|---|---|---|---|---|---|---|"]
    for s in sweeps:
        for v in sorted({r["value"] for r in rows if r["sweep"] == s}):
            sel = [r for r in rows if r["sweep"] == s and r["value"] == v]
            ok = [r for r in sel if r.get("ok")]
            def f(k):
                x = [r[k] for r in ok if r.get(k) is not None]
                return f"{np.mean(x):.2f}" if x else "-"
            lines.append(f"| {s} | {v} | {len(ok)}/{len(sel)} | {f('marker_rmse')} | {f('ap_in_joint')} | {f('precision')} | {f('recall')} | {f('f1')} | {f('f1_all')} |")
    (out_dir / "sweep_table.md").write_text("\n".join(lines) + "\n")
