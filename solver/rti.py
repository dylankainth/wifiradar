"""Radio tomographic imaging used to initialise wall candidates.

Every (node, AP) ray loses `c_b - 10 n log10 d - RSS` dB beyond free space; to first order that is the line
integral of an attenuation density over the ray. We solve a sparse, non-negative linear inverse problem on a
grid aligned with the Manhattan axes (FISTA, per-BSSID offsets projected out) and pull axis-aligned ridges out of
the image as wall candidates. The image is blurry and ambiguous - it only has to be good enough to start from.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix

from common.geometry import rot2
from solver.dataset import Dataset


@dataclass
class RTIConfig:
    cell: float = 0.3
    margin: float = 1.0
    lam: float = 1.0            # L1 weight (dB-ish per cell)
    iters: int = 400
    n_grid: tuple = (2.0, 2.4, 2.8, 3.3)
    ridge_rel: float = 0.25     # keep ridge cells above this fraction of the image maximum ...
    ridge_abs: float = 1.2      # ... and above this many dB
    min_len: float = 0.9
    gap_cells: int = 2


@dataclass
class RTIResult:
    D: np.ndarray               # (Gy,Gx) attenuation density, dB per metre
    origin: np.ndarray
    cell: float
    n: float
    obj: float
    theta: float
    candidates: np.ndarray      # (K,4) in the rotated (Manhattan) frame
    cand_atten: np.ndarray      # (K,) dB


def _ray_matrix(P, Q, origin, shape, cell, step=0.05):
    """Sparse (R, Gy*Gx) matrix of path lengths of rays P->Q through grid cells (sampled)."""
    Gy, Gx = shape
    rows, cols, vals = [], [], []
    for r, (p, q) in enumerate(zip(P, Q)):
        L = np.linalg.norm(q - p)
        n = max(2, int(L / step))
        s = (np.arange(n) + 0.5) / n
        pts = p[None] + s[:, None] * (q - p)[None]
        ix = np.floor((pts[:, 0] - origin[0]) / cell).astype(int)
        iy = np.floor((pts[:, 1] - origin[1]) / cell).astype(int)
        ok = (ix >= 0) & (ix < Gx) & (iy >= 0) & (iy < Gy)
        if ok.any():
            cid = iy[ok] * Gx + ix[ok]
            u, cnt = np.unique(cid, return_counts=True)
            rows.append(np.full(len(u), r)); cols.append(u); vals.append(cnt * (L / n))
    if not rows:
        return csr_matrix((len(P), Gy * Gx))
    return csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(P), Gy * Gx))


def solve_image(ds: Dataset, ap_xy: np.ndarray, theta: float, cfg: RTIConfig, n_path: float,
                node_mask: np.ndarray | None = None, obs_weight: np.ndarray | None = None):
    R = rot2(-theta)
    nodes = ds.node_xy @ R.T
    aps = ap_xy @ R.T
    lo = nodes.min(0) - cfg.margin
    hi = nodes.max(0) + cfg.margin
    origin = lo
    Gx, Gy = int(np.ceil((hi[0] - lo[0]) / cfg.cell)), int(np.ceil((hi[1] - lo[1]) / cfg.cell))
    o_ap = ds.bssid_ap[ds.o_bssid]
    sel = np.ones(len(ds.o_node), bool) if node_mask is None else node_mask[ds.o_node]
    on, ob, oa, orr = ds.o_node[sel], ds.o_bssid[sel], o_ap[sel], ds.o_rssi[sel]
    key = on.astype(np.int64) * 10_000 + oa
    uk, inv = np.unique(key, return_inverse=True)
    pn, pa = uk // 10_000, uk % 10_000
    A = _ray_matrix(nodes[pn], aps[pa], origin, (Gy, Gx), cfg.cell)
    S = A[inv]                                         # per-observation rows
    d = np.maximum(np.linalg.norm(nodes[on] - aps[oa], axis=1), 0.5)
    # per-BSSID mean-removal projector
    B = len(ds.bssids)
    cnt = np.bincount(ob, minlength=B).astype(float)
    cnt[cnt == 0] = 1
    def P(v):
        return v - (np.bincount(ob, weights=v, minlength=B) / cnt)[ob]
    w = np.ones(len(orr)) if obs_weight is None else obs_weight[sel]
    sw = np.sqrt(w)
    # L via power iteration
    x = np.random.default_rng(0).normal(size=S.shape[1])
    for _ in range(25):
        x = S.T @ (sw * P(sw * (S @ x)))
        x /= np.linalg.norm(x) + 1e-12
    Lip = float(np.linalg.norm(S.T @ (sw * P(sw * (S @ x)))) + 1e-9)
    best = None
    for n in cfg.n_grid:
        z = orr + 10 * n * np.log10(d)                 # = c_b - S a
        a = np.zeros(S.shape[1]); y = a.copy(); tk = 1.0
        for _ in range(cfg.iters):
            grad = S.T @ (sw * P(sw * (z + S @ y)))
            a_new = np.maximum(0.0, y - grad / Lip - cfg.lam / Lip)
            tk1 = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
            y = a_new + ((tk - 1) / tk1) * (a_new - a)
            a, tk = a_new, tk1
        r = sw * P(sw * (z + S @ a))
        obj = 0.5 * float(r @ r) + cfg.lam * a.sum()
        if best is None or obj < best[0]:
            best = (obj, n, a)
    obj, n, a = best
    return a.reshape(Gy, Gx), origin, n, obj


def orientation_from_image(D: np.ndarray) -> float:
    """Dominant axis angle mod 90deg from the structure tensor of the density image."""
    g = gaussian_filter(D, 1.0)
    gy, gx = np.gradient(g)
    phi = np.arctan2(gy, gx)
    w = gx ** 2 + gy ** 2
    return float(np.angle((w * np.exp(4j * phi)).sum()) / 4.0)


def _runs(mask_row, gap):
    idx = np.where(mask_row)[0]
    if len(idx) == 0:
        return []
    runs, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > gap + 1:
            runs.append((s, p)); s = i
        p = i
    runs.append((s, p))
    return runs


def extract_candidates(D, origin, cell, cfg: RTIConfig):
    """Axis-aligned ridges -> segments (x1,y1,x2,y2) in the grid frame, with a dB estimate."""
    out, att = [], []
    thr = max(cfg.ridge_abs / cell * 0 + cfg.ridge_abs, cfg.ridge_rel * 0) if False else None
    for horizontal in (True, False):
        img = D if horizontal else D.T                 # rows run along the wall direction
        prof = gaussian_filter(img, (0.0, 0.8))        # along-wall smoothing
        # integrate across the wall (3 rows) to get dB per crossing
        integ = (np.roll(prof, 1, 0) + prof + np.roll(prof, -1, 0)) * cell
        integ[0] = integ[-1] = 0
        up, dn = np.roll(integ, 1, 0), np.roll(integ, -1, 0)
        ridge = (integ > up) & (integ >= dn)
        tau = max(cfg.ridge_abs, cfg.ridge_rel * integ.max())
        m = ridge & (integ >= tau)
        for iy in range(m.shape[0]):
            for (a, b) in _runs(m[iy], cfg.gap_cells):
                length = (b - a + 1) * cell
                if length < cfg.min_len:
                    continue
                w = prof[iy - 1:iy + 2, a:b + 1].sum(1)
                yc = (np.arange(iy - 1, iy + 2) + 0.5) @ w / max(w.sum(), 1e-9)
                dB = float(integ[iy, a:b + 1].mean())
                if horizontal:
                    seg = [origin[0] + a * cell, origin[1] + yc * cell, origin[0] + (b + 1) * cell, origin[1] + yc * cell]
                else:
                    seg = [origin[0] + yc * cell, origin[1] + a * cell, origin[0] + yc * cell, origin[1] + (b + 1) * cell]
                out.append(seg); att.append(dB)
    return np.array(out, float).reshape(-1, 4), np.array(att, float)


def run_rti(ds: Dataset, ap_xy: np.ndarray, cfg: RTIConfig | None = None, theta: float | None = None,
            node_mask=None, n_path: float = 2.5) -> RTIResult:
    cfg = cfg or RTIConfig()
    if theta is None:
        D0, o0, n0, _ = solve_image(ds, ap_xy, 0.0, cfg, n_path, node_mask)
        theta = orientation_from_image(D0)
    D, origin, n, obj = solve_image(ds, ap_xy, theta, cfg, n_path, node_mask)
    cand, att = extract_candidates(D, origin, cfg.cell, cfg)
    return RTIResult(D, origin, cfg.cell, n, obj, theta, cand, att)
