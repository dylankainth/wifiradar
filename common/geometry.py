"""Small 2D geometry helpers shared by sim, solver and eval."""
from __future__ import annotations

import numpy as np


def segment_crossings(p: np.ndarray, q: np.ndarray, walls: np.ndarray) -> np.ndarray:
    """Hard intersection test. p,q: (M,2); walls: (K,4) x1,y1,x2,y2. Returns bool (M,K)."""
    p = np.asarray(p, float)[:, None, :]
    q = np.asarray(q, float)[:, None, :]
    a = walls[None, :, 0:2]
    b = walls[None, :, 2:4]

    def orient(u, v, w):
        return (v[..., 0] - u[..., 0]) * (w[..., 1] - u[..., 1]) - (v[..., 1] - u[..., 1]) * (w[..., 0] - u[..., 0])

    o1, o2 = orient(a, b, p), orient(a, b, q)
    o3, o4 = orient(p, q, a), orient(p, q, b)
    return (o1 * o2 < 0) & (o3 * o4 < 0)


def sample_walls(walls: np.ndarray, step: float = 0.1) -> np.ndarray:
    """Points sampled along segments (K,4) -> (N,2), plus a per-point segment index and the segment lengths."""
    pts, idx = [], []
    for k, (x1, y1, x2, y2) in enumerate(walls):
        L = float(np.hypot(x2 - x1, y2 - y1))
        n = max(2, int(np.ceil(L / step)) + 1)
        t = np.linspace(0, 1, n)
        pts.append(np.stack([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], 1))
        idx.append(np.full(n, k))
    if not pts:
        return np.zeros((0, 2)), np.zeros(0, int)
    return np.concatenate(pts), np.concatenate(idx)


def rot2(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def umeyama_rigid(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None):
    """Least-squares rigid (rotation+translation, no scale/reflection) mapping src->dst. Returns R(2,2), t(2)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    w = np.ones(len(src)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    mu_s, mu_d = (w[:, None] * src).sum(0), (w[:, None] * dst).sum(0)
    S = (w[:, None] * (src - mu_s)).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(S)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, d]) @ U.T
    return R, mu_d - R @ mu_s


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi
