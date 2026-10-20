"""Sparse wall-atlas tomography.

Candidate walls are all short axis-aligned pieces on a fine lattice of horizontal/vertical lines in a frame
rotated by theta. A ray (phone node -> AP) crosses a piece or not (exact test), so the model is linear:
    excess_loss(ray) = sum_pieces crossing(ray, piece) * a_piece,   a >= 0, sparse.
Unlike a pixel image there is no blur along the ray. The fit objective as a function of theta is also what
tells us the building's Manhattan orientation (aligned lattice -> sparser, better explanation).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from common.geometry import rot2
from solver.dataset import Dataset


@dataclass
class AtlasConfig:
    line_step: float = 0.2
    piece_len: float = 0.5
    margin: float = 0.8
    lam: float = 8.0
    iters: int = 300
    n_grid: tuple = (2.0, 2.4, 2.8, 3.3)
    min_len: float = 1.0
    thresh_db: float = 0.8       # pieces below this are noise
    gap_pieces: int = 1


@dataclass
class Atlas:
    theta: float
    n: float
    obj: float
    a: np.ndarray                  # (n_pieces,)
    pieces: np.ndarray             # (n_pieces, 4) in the rotated frame
    horiz: np.ndarray              # bool per piece
    pair_nodes: np.ndarray
    pair_aps: np.ndarray


def _lattice(lo, hi, cfg: AtlasConfig):
    ys = np.arange(lo[1], hi[1] + 1e-9, cfg.line_step)
    xs = np.arange(lo[0], hi[0] + 1e-9, cfg.line_step)
    nxp = int(np.ceil((hi[0] - lo[0]) / cfg.piece_len))
    nyp = int(np.ceil((hi[1] - lo[1]) / cfg.piece_len))
    return xs, ys, nxp, nyp


def crossing_matrix(P, Q, lo, hi, cfg: AtlasConfig):
    """Sparse (R, n_pieces) 0/1 matrix. Piece ordering: horizontal lines (line-major) then vertical lines."""
    xs, ys, nxp, nyp = _lattice(lo, hi, cfg)
    R = len(P)
    rows, cols = [], []
    # horizontal lines y = ys[i], pieces along x
    py, qy = P[:, 1][:, None], Q[:, 1][:, None]
    straddle = (py - ys[None]) * (qy - ys[None]) < 0
    t = (ys[None] - py) / np.where(np.abs(qy - py) < 1e-12, 1e-12, qy - py)
    xc = P[:, 0][:, None] + t * (Q[:, 0] - P[:, 0])[:, None]
    pi = np.floor((xc - lo[0]) / cfg.piece_len).astype(int)
    ok = straddle & (pi >= 0) & (pi < nxp)
    r_idx, l_idx = np.nonzero(ok)
    rows.append(r_idx); cols.append(l_idx * nxp + pi[r_idx, l_idx])
    off = len(ys) * nxp
    px, qx = P[:, 0][:, None], Q[:, 0][:, None]
    straddle = (px - xs[None]) * (qx - xs[None]) < 0
    t = (xs[None] - px) / np.where(np.abs(qx - px) < 1e-12, 1e-12, qx - px)
    yc = P[:, 1][:, None] + t * (Q[:, 1] - P[:, 1])[:, None]
    pj = np.floor((yc - lo[1]) / cfg.piece_len).astype(int)
    ok = straddle & (pj >= 0) & (pj < nyp)
    r_idx, l_idx = np.nonzero(ok)
    rows.append(r_idx); cols.append(off + l_idx * nyp + pj[r_idx, l_idx])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    n_pieces = off + len(xs) * nyp
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(R, n_pieces)).tocsr()
    # piece geometry
    pc = []
    for i, y in enumerate(ys):
        for k in range(nxp):
            pc.append([lo[0] + k * cfg.piece_len, y, min(lo[0] + (k + 1) * cfg.piece_len, hi[0]), y])
    for i, x in enumerate(xs):
        for k in range(nyp):
            pc.append([x, lo[1] + k * cfg.piece_len, x, min(lo[1] + (k + 1) * cfg.piece_len, hi[1])])
    horiz = np.r_[np.ones(off, bool), np.zeros(n_pieces - off, bool)]
    return A, np.array(pc), horiz


def fit_atlas(ds: Dataset, ap_xy: np.ndarray, theta: float, cfg: AtlasConfig, node_mask=None,
              obs_weight=None) -> Atlas:
    Rm = rot2(-theta)
    nodes, aps = ds.node_xy @ Rm.T, ap_xy @ Rm.T
    lo, hi = nodes.min(0) - cfg.margin, nodes.max(0) + cfg.margin
    o_ap = ds.bssid_ap[ds.o_bssid]
    sel = np.ones(len(ds.o_node), bool) if node_mask is None else node_mask[ds.o_node]
    on, ob, oa, orr = ds.o_node[sel], ds.o_bssid[sel], o_ap[sel], ds.o_rssi[sel]
    key = on.astype(np.int64) * 10_000 + oa
    uk, inv = np.unique(key, return_inverse=True)
    pn, pa = uk // 10_000, uk % 10_000
    A, pieces, horiz = crossing_matrix(nodes[pn], aps[pa], lo, hi, cfg)
    S = A[inv].tocsr()
    d = np.maximum(np.linalg.norm(nodes[on] - aps[oa], axis=1), 0.5)
    B = len(ds.bssids)
    cnt = np.bincount(ob, minlength=B).astype(float)
    cnt[cnt == 0] = 1
    P = lambda v: v - (np.bincount(ob, weights=v, minlength=B) / cnt)[ob]
    w = np.ones(len(orr)) if obs_weight is None else obs_weight[sel]
    sw = np.sqrt(w)
    x = np.random.default_rng(0).normal(size=S.shape[1])
    for _ in range(25):
        x = S.T @ (sw * P(sw * (S @ x)))
        x /= np.linalg.norm(x) + 1e-12
    Lip = float(np.linalg.norm(S.T @ (sw * P(sw * (S @ x)))) + 1e-9)
    best = None
    for n in cfg.n_grid:
        z = orr + 10 * n * np.log10(d)
        a = np.zeros(S.shape[1]); y = a.copy(); tk = 1.0
        for _ in range(cfg.iters):
            grad = S.T @ (sw * P(sw * (z + S @ y)))
            a_new = np.maximum(0.0, y - (grad + cfg.lam) / Lip)
            tk1 = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
            y = a_new + ((tk - 1) / tk1) * (a_new - a)
            a, tk = a_new, tk1
        r = sw * P(sw * (z + S @ a))
        obj = 0.5 * float(r @ r) + cfg.lam * a.sum()
        if best is None or obj < best[0]:
            best = (obj, n, a)
    obj, n, a = best
    return Atlas(theta, n, obj, a, pieces, horiz, pn, pa)


def atlas_segments(at: Atlas, cfg: AtlasConfig):
    """Merge active pieces into segments (rotated frame) with a dB estimate. Adjacent parallel lines are merged
    by keeping the strongest line per 'band' (non-max suppression across the normal direction)."""
    out, att = [], []
    act = at.a >= cfg.thresh_db
    if not act.any():
        return np.zeros((0, 4)), np.zeros(0)
    pcs, a = at.pieces, at.a
    for horizontal in (True, False):
        sel = np.where(act & (at.horiz == horizontal))[0]
        if len(sel) == 0:
            continue
        # group by line coordinate then by along-coordinate
        line = pcs[sel, 1] if horizontal else pcs[sel, 0]
        along0 = pcs[sel, 0] if horizontal else pcs[sel, 1]
        along1 = pcs[sel, 2] if horizontal else pcs[sel, 3]
        order = np.lexsort((along0, line))
        sel, line, along0, along1 = sel[order], line[order], along0[order], along1[order]
        i = 0
        while i < len(sel):
            j = i
            while (j + 1 < len(sel) and abs(line[j + 1] - line[i]) < 1e-9 and
                   along0[j + 1] - along1[j] <= cfg.gap_pieces * cfg.piece_len + 1e-9):
                j += 1
            length = along1[j] - along0[i]
            if length >= cfg.min_len:
                dB = float(a[sel[i:j + 1]].mean())
                c = line[i]
                seg = [along0[i], c, along1[j], c] if horizontal else [c, along0[i], c, along1[j]]
                out.append(seg); att.append(dB)
            i = j + 1
    return np.array(out, float).reshape(-1, 4), np.array(att)
