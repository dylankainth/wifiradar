"""Metrics: trajectory ATE, AP error, wall chamfer + length-weighted F1. All after marker-based rigid alignment."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from common.geometry import sample_walls, umeyama_rigid
from eval.truth import GroundTruth


def align_from_markers(est_at_markers: dict[int, list[np.ndarray]], truth_markers: dict[int, np.ndarray]):
    """Rigid transform (R, t) mapping solver frame -> truth frame from marker visits (every visit is a pair)."""
    src, dst = [], []
    for m, visits in est_at_markers.items():
        if m in truth_markers:
            for p in visits:
                src.append(p); dst.append(truth_markers[m])
    if len(src) < 2:
        return np.eye(2), np.zeros(2)
    return umeyama_rigid(np.array(src), np.array(dst))


def apply(R, t, xy):
    return np.asarray(xy) @ R.T + t


def ate_rmse(est_xy: np.ndarray, true_xy: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((est_xy - true_xy) ** 2, 1))))


def ap_errors(est_aps: list[dict], gt: GroundTruth, R, t):
    """Match estimated AP groups to true APs by shared BSSIDs. est_aps: dicts with 'bssids' and 'xy'."""
    errs = {}
    for ta in gt.aps:
        best = None
        for ea in est_aps:
            if set(ta.bssids) & set(ea["bssids"]):
                e = float(np.linalg.norm(apply(R, t, np.array(ea["xy"])[None])[0] - ta.xy))
                if best is None or e < best:
                    best = e
        if best is not None:
            errs[ta.name] = best
    return errs


def _dist_to_walls(points: np.ndarray, walls: np.ndarray, step=0.05):
    if len(walls) == 0 or len(points) == 0:
        return np.full(len(points), np.inf)
    pts, _ = sample_walls(walls, step)
    return cKDTree(pts).query(points)[0]


def wall_metrics(est_walls: np.ndarray, gt_walls: np.ndarray, tol: float = 0.5, region: np.ndarray | None = None,
                 observable: np.ndarray | None = None):
    """Chamfer (symmetric mean nearest distance, m) and length-weighted precision/recall/F1 at tolerance tol.

    region: optional (x0,y0,x1,y1); only GT wall points inside it count for recall (walls the walker could not
    have observed are not held against the solver). observable: optional bool mask over gt_walls (see
    observable_mask); if given, recall/F1 are computed over observable GT walls only and the plain versions are
    returned as *_all."""
    ep, _ = sample_walls(est_walls, 0.05)
    gp, _ = sample_walls(gt_walls, 0.05)
    if region is not None:
        x0, y0, x1, y1 = region
        gp = gp[(gp[:, 0] >= x0) & (gp[:, 0] <= x1) & (gp[:, 1] >= y0) & (gp[:, 1] <= y1)]
    if len(ep) == 0 or len(gp) == 0:
        return {"chamfer_m": float("inf"), "precision": 0.0, "recall": 0.0, "f1": 0.0}
    d_e = _dist_to_walls(ep, gt_walls)     # est -> gt (precision uses the full GT)
    d_g = _dist_to_walls(gp, est_walls)    # gt -> est
    prec, rec = float((d_e <= tol).mean()), float((d_g <= tol).mean())
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    out = {"chamfer_m": float(0.5 * (d_e.mean() + d_g.mean())), "precision": prec, "recall": rec, "f1": f1}
    if observable is not None and observable.any():
        go, _ = sample_walls(gt_walls[observable], 0.05)
        rec_o = float((_dist_to_walls(go, est_walls) <= tol).mean())
        f1_o = 0.0 if prec + rec_o == 0 else 2 * prec * rec_o / (prec + rec_o)
        out.update({"recall_all": rec, "f1_all": f1, "recall": rec_o, "f1": f1_o,
                    "observable_frac": float(gt_walls[observable].shape[0] and
                                             np.hypot(*(gt_walls[observable][:, 2:] - gt_walls[observable][:, :2]).T).sum() /
                                             np.hypot(*(gt_walls[:, 2:] - gt_walls[:, :2]).T).sum())})
    return out


def observable_mask(gt_walls: np.ndarray, node_xy: np.ndarray, ap_xy: np.ndarray, min_each: int = 5) -> np.ndarray:
    """Which GT walls can RSS identify at all? A wall is observable if, for at least one AP, at least `min_each`
    phone positions have the wall between them and the AP and at least `min_each` do not. Otherwise its effect
    is a constant that the per-BSSID transmit-power offset absorbs (e.g. the outer shell seen from outside APs,
    or any wall nobody crosses a ray through). node_xy / ap_xy must be in the truth frame."""
    from common.geometry import segment_crossings
    obs = np.zeros(len(gt_walls), bool)
    for ap in ap_xy:
        c = segment_crossings(node_xy, np.tile(ap, (len(node_xy), 1)), gt_walls)   # (N,K)
        n_c = c.sum(0)
        obs |= (n_c >= min_each) & (len(node_xy) - n_c >= min_each)
    return obs
