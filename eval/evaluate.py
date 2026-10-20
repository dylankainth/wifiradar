"""Evaluate a pipeline Result against ground truth (simulated truth.json or hand-measured ground_truth.yaml)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.geometry import rot2, umeyama_rigid, wrap_angle
from eval.metrics import (align_from_markers, ap_errors, apply, ate_rmse, observable_mask, wall_metrics)
from eval.truth import GroundTruth
from solver.pipeline import Result

MARKER_DELAY_S = 1.5   # the button is pressed on arrival; we are standing at the marker a moment later


def _marker_positions(res: Result, fn):
    ev: dict[int, list] = {}
    for w in res.session.waypoints:
        ev.setdefault(int(w["marker"]), []).append(fn(w["t_s"] + MARKER_DELAY_S))
    return ev


@dataclass
class Evaluation:
    metrics: dict
    R: np.ndarray            # solver frame -> truth frame
    t: np.ndarray
    R_pdr: np.ndarray        # same, but fitted to the PDR-only trajectory
    t_pdr: np.ndarray
    obs_mask: np.ndarray | None


def evaluate(res: Result, gt: GroundTruth) -> Evaluation:
    M: dict = {}
    ses = res.session
    pg = res.pg
    # --- alignment (markers only: this is the information a real survey has) ---
    est_m = _marker_positions(res, lambda t: pg.at(t)[0])
    R, t = align_from_markers(est_m, gt.markers)
    pdr_m = _marker_positions(res, lambda t: res.pdr.pos(t))
    R_p, t_p = align_from_markers(pdr_m, gt.markers)
    # --- trajectory ---
    errs = [np.linalg.norm(apply(R, t, p[None])[0] - gt.markers[m]) for m, v in est_m.items() if m in gt.markers for p in v]
    errs_p = [np.linalg.norm(apply(R_p, t_p, p[None])[0] - gt.markers[m]) for m, v in pdr_m.items() if m in gt.markers for p in v]
    M["marker_rmse_m"] = float(np.sqrt(np.mean(np.square(errs)))) if errs else None
    M["marker_rmse_pdr_only_m"] = float(np.sqrt(np.mean(np.square(errs_p)))) if errs_p else None
    if gt.traj is not None:
        tt, txy = gt.traj
        node_t = pg.nodes.t
        tru = np.stack([np.interp(node_t, tt, txy[:, 0]), np.interp(node_t, tt, txy[:, 1])], 1)
        M["ate_rmse_m"] = ate_rmse(apply(R, t, pg.P), tru)
        M["ate_pdr_only_m"] = ate_rmse(apply(R_p, t_p, pg.nodes.pdr_xy), tru)
        path = np.linalg.norm(np.diff(txy, axis=0), axis=1).sum()
        M["path_length_m"] = float(path)
        M["pdr_drift_pct"] = float(100 * np.linalg.norm(apply(R_p, t_p, pg.nodes.pdr_xy[-1:])[0] - tru[-1]) / path)
    # --- grouping (needs BSSIDs in truth) ---
    seen = set(ses.bssids())
    groups = {frozenset(g) for g in res.groups}
    ok = tot = 0
    for ap in gt.aps:
        want = frozenset(b for b in ap.bssids if b in seen)
        if want:
            tot += 1
            ok += want in groups
    M["grouping_exact"] = f"{ok}/{tot}"
    # --- APs ---
    e0 = ap_errors([{"bssids": g, "xy": res.aps0.xy[j]} for j, g in enumerate(res.ds.groups)], gt, R, t)
    e1 = ap_errors(res.ap_list, gt, R, t)
    if gt.bounds is not None:
        x0, y0, x1, y1 = gt.bounds
        inside = {a.name for a in gt.aps if x0 <= a.xy[0] <= x1 and y0 <= a.xy[1] <= y1}
    else:
        inside = {a.name for a in gt.aps}
    for tag, e in (("multilat", e0), ("joint", e1)):
        vi = [v for k, v in e.items() if k in inside]
        M[f"ap_err_{tag}_inside_m"] = float(np.mean(vi)) if vi else None
        M[f"ap_err_{tag}_all"] = {k: round(v, 2) for k, v in e.items()}
    # --- walls ---
    om = None
    if res.fit is not None and len(gt.walls):
        W = np.hstack([apply(R, t, res.fit.walls[:, :2]), apply(R, t, res.fit.walls[:, 2:])]) if len(res.fit.walls) else np.zeros((0, 4))
        om = observable_mask(gt.walls, apply(R, t, res.ds.node_xy), np.array([a.xy for a in gt.aps]))
        M["walls_est"] = int(len(W))
        for tol in (0.5, 1.0):
            m = wall_metrics(W, gt.walls, tol, observable=om)
            M[f"walls@{tol}"] = {k: round(v, 3) for k, v in m.items()}
        # Manhattan angle error (truth frame axes are at 0 deg)
        th_truth_frame = res.fit.theta + np.arctan2(R[1, 0], R[0, 0])
        M["theta_err_deg"] = float(np.rad2deg(((th_truth_frame + np.pi / 4) % (np.pi / 2)) - np.pi / 4))
    return Evaluation(M, R, t, R_p, t_p, om)
