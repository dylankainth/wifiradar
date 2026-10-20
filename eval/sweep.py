"""Benchmark harness: simulate -> (real loaders) -> pipeline -> evaluate, in parallel; plus the noise/rate sweeps."""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import tempfile
import time
from dataclasses import fields, is_dataclass
from pathlib import Path


def _set(obj, dotted, value):
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def run_case(case: dict) -> dict:
    """case: {seed, sim: {dotted: value}, pipe: {dotted: value}, oracle_theta: bool}"""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    import numpy as np
    from common.session import load_session
    from eval.evaluate import evaluate
    from eval.truth import load_truth
    from sim.generate import SimConfig, generate
    from solver.pipeline import PipelineConfig, run_pipeline

    sc = SimConfig(seed=case["seed"])
    for k, v in case.get("sim", {}).items():
        _set(sc, k, v)
    pc = PipelineConfig()
    for k, v in case.get("pipe", {}).items():
        _set(pc, k, v)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        p = generate(sc, Path(d) / "s")
        ses, gt = load_session(p), load_truth(p)
        if case.get("anchors"):   # labelled ablation: marker coordinates are given to the pose graph as anchors
            pc.marker_xy = gt.markers
        if case.get("oracle_theta"):
            T = json.loads((p / "truth.json").read_text())
            pc.theta_override_rad = T["world_rot"] % (np.pi / 2)
        row = {"seed": case["seed"], **{f"sim.{k}": v for k, v in case.get("sim", {}).items()},
               **{f"pipe.{k}": v for k, v in case.get("pipe", {}).items()}, "n_scans": len(ses.scans)}
        try:
            res = run_pipeline(ses, pc)
            ev = evaluate(res, gt)
            m = ev.metrics
            w = m.get("walls@0.5", {})
            row.update({"ok": True, "marker_rmse": m["marker_rmse_m"], "marker_rmse_pdr": m["marker_rmse_pdr_only_m"],
                        "ate": m.get("ate_rmse_m"), "ate_pdr": m.get("ate_pdr_only_m"), "drift_pct": m.get("pdr_drift_pct"),
                        "ap_in_multilat": m["ap_err_multilat_inside_m"], "ap_in_joint": m["ap_err_joint_inside_m"],
                        "precision": w.get("precision"), "recall": w.get("recall"), "f1": w.get("f1"),
                        "f1_all": w.get("f1_all"), "chamfer": w.get("chamfer_m"), "theta_err": m.get("theta_err_deg"),
                        "n_walls": m.get("walls_est"), "grouping": m["grouping_exact"],
                        "f1_1m": m.get("walls@1.0", {}).get("f1")})
            if case.get("wall_detail") and res.fit is not None and len(res.fit.walls):
                from common.geometry import sample_walls
                from eval.metrics import _dist_to_walls, apply
                W = np.hstack([apply(ev.R, ev.t, res.fit.walls[:, :2]), apply(ev.R, ev.t, res.fit.walls[:, 2:])])
                det = []
                for k, w in enumerate(W):
                    pts, _ = sample_walls(w[None], 0.05)
                    det.append({"support": float(res.support[k]), "dL": float(res.fit.dL[k]), "atten": float(res.fit.atten[k]),
                                "length": float(np.hypot(w[2] - w[0], w[3] - w[1])),
                                "frac_correct": float((_dist_to_walls(pts, gt.walls) <= 0.5).mean())})
                row["walls_detail"] = det
        except Exception as e:  # keep sweeps going; failures are reported, not hidden
            import traceback
            row.update({"ok": False, "error": repr(e), "tb": traceback.format_exc()[-600:]})
    row["runtime_s"] = round(time.time() - t0, 1)
    return row


def run_cases(cases, procs=4):
    ctx = mp.get_context("spawn")
    with ctx.Pool(procs) as pool:
        return pool.map(run_case, cases, chunksize=1)


def summarize(rows, keys=("marker_rmse", "ap_in_joint", "precision", "recall", "f1", "f1_all", "chamfer", "theta_err")):
    import numpy as np
    ok = [r for r in rows if r.get("ok")]
    out = {"n": len(rows), "failed": len(rows) - len(ok)}
    for k in keys:
        v = [abs(r[k]) if k == "theta_err" else r[k] for r in ok if r.get(k) is not None]
        out[k] = (float(np.mean(v)), float(np.std(v))) if v else (None, None)
    return out


def sweeps(out_dir: Path, seeds=range(1, 5), procs=4, quick=False):
    """Noise / sampling-rate sweeps. Each point = mean over seeds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    grids = {
        "rss_noise_db": ("radio.meas_sigma_db", [0.5, 1.5, 3.0, 5.0]),
        "shadow_sigma_db": ("radio.shadow_sigma_db", [0.0, 2.5, 4.0, 6.0]),
        "scan_period_s": ("scan_period_s", [2.0, 4.0, 8.0, 15.0]),
        "heading_drift_scale": ("heading_drift_scale", [0.0, 1.0, 2.0, 4.0]),
        "n_visits": ("walk.n_visits", [15, 30, 60]),
    }
    if quick:
        grids = {k: (a, v[:2]) for k, (a, v) in grids.items()}
    cases, tags = [], []
    for name, (key, vals) in grids.items():
        for v in vals:
            for s in seeds:
                cases.append({"seed": s, "sim": {key: v}, "pipe": {"n_boot": 0}})
                tags.append((name, v))
    rows = run_cases(cases, procs)
    for r, (n, v) in zip(rows, tags):
        r["sweep"], r["value"] = n, v
    (out_dir / "sweep_rows.json").write_text(json.dumps(rows, indent=1, default=float))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/out/sweep")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    from eval.sweep_plots import plot_sweeps
    rows = sweeps(Path(a.out), range(1, a.seeds + 1), a.procs, a.quick)
    plot_sweeps(rows, Path(a.out))


if __name__ == "__main__":
    main()
