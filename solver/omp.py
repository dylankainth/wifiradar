"""Greedy (orthogonal matching pursuit) wall selection on the atlas lattice.

Candidates are runs of 1..m consecutive pieces on one lattice line, so a candidate is a whole wall segment. At each
step the candidate that best explains the remaining excess loss is added, all coefficients are refit with
non-negative least squares, and we stop when the gain no longer pays a BIC-like penalty. Unlike LASSO there is no
shrinkage bias and nearly collinear neighbouring lines do not all light up.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls
from scipy.sparse import coo_matrix, csr_matrix

from common.geometry import rot2
from solver.atlas import AtlasConfig, crossing_matrix, _lattice
from solver.dataset import Dataset


@dataclass
class OMPConfig:
    atlas: AtlasConfig = None
    run_lengths: tuple = (1, 2, 3, 4, 6, 8, 12, 20)
    sigma: float = 3.0
    penalty: float = 10.0      # required drop in 0.5*chi2 per added segment
    max_atoms: int = 40
    n_grid: tuple = (2.0, 2.4, 2.8, 3.3)
    min_a: float = 0.8

    def __post_init__(self):
        self.atlas = self.atlas or AtlasConfig()


def _run_matrix(n_pieces, line_starts, nper, run_lengths):
    """Group matrix G (n_pieces, n_runs) and run descriptors (first piece, length) for every line."""
    rows, cols, desc = [], [], []
    rid = 0
    for ls, npl in zip(line_starts, nper):
        for m in run_lengths:
            for k in range(0, npl - m + 1):
                rows.extend(range(ls + k, ls + k + m)); cols.extend([rid] * m)
                desc.append((ls + k, m)); rid += 1
    return coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_pieces, rid)).tocsr(), np.array(desc)


def omp_segments(ds: Dataset, ap_xy: np.ndarray, theta: float, cfg: OMPConfig, node_mask=None, obs_weight=None):
    """Returns (segments (K,4) in rotated frame, atten (K,), objective, n, info)."""
    ac = cfg.atlas
    Rm = rot2(-theta)
    nodes, aps = ds.node_xy @ Rm.T, ap_xy @ Rm.T
    lo, hi = nodes.min(0) - ac.margin, nodes.max(0) + ac.margin
    o_ap = ds.bssid_ap[ds.o_bssid]
    sel = np.ones(len(ds.o_node), bool) if node_mask is None else node_mask[ds.o_node]
    on, ob, oa, orr = ds.o_node[sel], ds.o_bssid[sel], o_ap[sel], ds.o_rssi[sel]
    key = on.astype(np.int64) * 10_000 + oa
    uk, inv = np.unique(key, return_inverse=True)
    pn, pa = uk // 10_000, uk % 10_000
    A, pieces, horiz = crossing_matrix(nodes[pn], aps[pa], lo, hi, ac)
    xs, ys, nxp, nyp = _lattice(lo, hi, ac)
    starts = [i * nxp for i in range(len(ys))] + [len(ys) * nxp + i * nyp for i in range(len(xs))]
    nper = [nxp] * len(ys) + [nyp] * len(xs)
    G, desc = _run_matrix(A.shape[1], starts, nper, cfg.run_lengths)
    S_runs = (A @ G).tocsr()
    S = S_runs[inv].tocsr()                         # per-observation rows
    d = np.maximum(np.linalg.norm(nodes[on] - aps[oa], axis=1), 0.5)
    B = len(ds.bssids)
    cnt = np.bincount(ob, minlength=B).astype(float); cnt[cnt == 0] = 1
    P = lambda v: v - (np.bincount(ob, weights=v, minlength=B) / cnt)[ob]
    w = np.ones(len(orr)) if obs_weight is None else obs_weight[sel]
    sw = np.sqrt(w)
    colnorm = np.sqrt(np.asarray(S.multiply(S).sum(0)).ravel()) + 1e-9
    best = None
    for n in cfg.n_grid:
        y = -P(sw * (orr + 10 * n * np.log10(d)))   # = P S a  (excess attenuation)
        support: list[int] = []
        r = y.copy()
        a = np.zeros(0)
        obj_prev = 0.5 * float(y @ y) / cfg.sigma ** 2
        while len(support) < cfg.max_atoms:
            g = (S.T @ (sw * r)) / colnorm
            if support:
                g[support] = -np.inf
            j = int(np.argmax(g))
            if g[j] <= 0:
                break
            sup2 = support + [j]
            X = np.stack([P(sw * np.asarray(S[:, k].todense()).ravel()) for k in sup2], 1)
            a2, _ = nnls(X, y)
            r2 = y - X @ a2
            obj = 0.5 * float(r2 @ r2) / cfg.sigma ** 2
            if obj_prev - obj < cfg.penalty:
                break
            support, a, r, obj_prev = sup2, a2, r2, obj
        total = obj_prev + cfg.penalty * len(support)
        if best is None or total < best[0]:
            best = (total, n, list(support), a.copy())
    total, n, support, a = best
    segs, att = [], []
    for j, aj in zip(support, a):
        if aj < cfg.min_a:
            continue
        first, m = desc[j]
        seg = np.array([pieces[first][0], pieces[first][1], pieces[first + m - 1][2], pieces[first + m - 1][3]])
        segs.append(seg); att.append(aj)
    return np.array(segs, float).reshape(-1, 4), np.array(att), total, n, {"n_atoms": len(support)}
