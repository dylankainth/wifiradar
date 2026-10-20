"""Stage 3: pose graph with WiFi-fingerprint loop closures (scipy least_squares; GTSAM not needed for 2D).

Variables: per-scan node pose (x, y, theta) plus one global odometry scale s.
Factors: PDR relative-pose odometry between consecutive nodes, weak compass prior on theta, an anchor on node 0,
         and loop closures between time-separated nodes whose RSSI fingerprints look alike (robust loss).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from common.session import Session
from solver.grouping import rssi_matrix
from solver.pdr import PDRResult


@dataclass
class PGConfig:
    sig_t0: float = 0.08           # odometry translation sigma: sig_t0 + sig_t1 * distance
    sig_t1: float = 0.06
    sig_th0: float = 0.02          # odometry heading sigma: sig_th0 + sig_th1 * sqrt(dt)
    sig_th1: float = 0.01
    sig_compass: float = 0.35      # weak absolute heading prior (rad)
    sig_scale: float = 0.015      # odometry scale: NOT observable from loop closures (they bias it low); trust the step length
    # loop closures from fingerprints
    use_loops: bool = True
    min_dt_s: float = 25.0
    fp_floor_dbm: float = -85.0
    fp_thresh_scale: float = 1.7   # accept if distance < scale * (stationary-pair baseline)
    fp_thresh_min_db: float = 3.0
    max_loops_per_node: int = 2
    sig_loop: float = 1.5
    loop_radius: float = 0.0       # optional dead zone (m) around 'same place'; 0 = plain equality (tested: no better)
    robust_scale: float = 2.5
    # labelled ablation: marker ids give exact loop closures (+ anchors if coordinates are supplied)
    use_labels: bool = False
    sig_label: float = 0.25


@dataclass
class Nodes:
    t: np.ndarray
    scan_idx: np.ndarray
    pdr_xy: np.ndarray
    pdr_h: np.ndarray


@dataclass
class PoseGraphResult:
    nodes: Nodes
    P: np.ndarray                  # (N,2) corrected positions (solver frame)
    th: np.ndarray                 # (N,) corrected headings
    scale: float
    loops: list[tuple[int, int, float]]
    loops_kept: list[tuple[int, int, float]]
    info: dict = field(default_factory=dict)

    def at(self, t):
        """Corrected position at arbitrary times (apply node-local correction to PDR motion, blend neighbours)."""
        t = np.atleast_1d(np.asarray(t, float))
        nd = self.nodes
        out = np.zeros((len(t), 2))
        for k, tk in enumerate(t):
            i = int(np.clip(np.searchsorted(nd.t, tk) - 1, 0, len(nd.t) - 1))
            j = min(i + 1, len(nd.t) - 1)
            cands = []
            for n in {i, j}:
                R = _rot(self.th[n] - nd.pdr_h[n])
                cands.append((n, self.P[n] + self.scale * R @ (self._pdr(tk) - nd.pdr_xy[n])))
            if i == j or len(cands) == 1:
                out[k] = cands[0][1]
            else:
                lam = np.clip((tk - nd.t[i]) / max(nd.t[j] - nd.t[i], 1e-6), 0, 1)
                pi = dict(cands)
                out[k] = (1 - lam) * pi[i] + lam * pi[j]
        return out

    _pdr: object = None


def _rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def build_nodes(ses: Session, pdr: PDRResult) -> Nodes:
    idx = np.arange(len(ses.scans))
    t = np.array([s.t_s for s in ses.scans])
    ok = (t > pdr._node_t[0])
    return Nodes(t[ok], idx[ok], pdr.pos(t[ok]), pdr.heading(t[ok]))


def fingerprint_distances(F: np.ndarray, floor: float):
    """Pairwise RMS RSSI difference over BSSIDs where either value is above floor. NaN -> floor-5."""
    N = len(F)
    X = np.where(np.isnan(F), floor - 5.0, F)
    D = np.full((N, N), np.inf)
    strong = X > floor
    for i in range(N):
        use = strong[i][None, :] | strong
        n = use.sum(1)
        d2 = (np.where(use, (X[i][None, :] - X) ** 2, 0)).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            D[i] = np.where(n >= 3, np.sqrt(d2 / np.maximum(n, 1)), np.inf)
    return D


def find_loops(nodes: Nodes, F: np.ndarray, cfg: PGConfig):
    D = fingerprint_distances(F, cfg.fp_floor_dbm)
    N = len(nodes.t)
    # baseline: fingerprint distance between consecutive scans taken while standing still
    still = [D[i, i + 1] for i in range(N - 1)
             if np.linalg.norm(nodes.pdr_xy[i + 1] - nodes.pdr_xy[i]) < 0.15 and np.isfinite(D[i, i + 1])]
    d0 = float(np.median(still)) if len(still) >= 5 else 2.2
    thr = max(cfg.fp_thresh_min_db, cfg.fp_thresh_scale * d0)
    loops = []
    for i in range(N):
        order = np.argsort(D[i])
        n = 0
        for j in order:
            if not np.isfinite(D[i, j]) or D[i, j] > thr:
                break
            if abs(nodes.t[j] - nodes.t[i]) < cfg.min_dt_s:
                continue
            a, b = (i, j) if i < j else (j, i)
            loops.append((a, b, float(D[i, j])))
            n += 1
            if n >= cfg.max_loops_per_node:
                break
    seen, uniq = set(), []
    for a, b, d in loops:
        if (a, b) not in seen:
            seen.add((a, b)); uniq.append((a, b, d))
    return uniq, d0, thr


def optimise(nodes: Nodes, loops, cfg: PGConfig, label_pairs=(), anchors=()):
    """label_pairs: [(i,j)] exact revisits; anchors: [(i, xy)] known positions (labelled ablation only)."""
    N = len(nodes.t)
    h, pxy = nodes.pdr_h, nodes.pdr_xy
    dt = np.diff(nodes.t)
    dp_loc = np.array([_rot(-h[i]).dot(pxy[i + 1] - pxy[i]) for i in range(N - 1)])
    dth = np.diff(h)
    dist = np.linalg.norm(dp_loc, axis=1)
    sig_t = cfg.sig_t0 + cfg.sig_t1 * dist
    sig_th = cfg.sig_th0 + cfg.sig_th1 * np.sqrt(np.maximum(dt, 0))
    L = list(loops)
    nres = 2 * (N - 1) + (N - 1) + N + 3 + 1 + 2 * len(L) + 2 * len(label_pairs) + 2 * len(anchors)

    def unpack(x):
        return x[:N], x[N:2 * N], x[2 * N:3 * N], x[3 * N]

    def res(x):
        X, Y, TH, s = unpack(x)
        out = []
        c, sn = np.cos(TH[:-1]), np.sin(TH[:-1])
        dx, dy = X[1:] - X[:-1], Y[1:] - Y[:-1]
        lx, ly = c * dx + sn * dy, -sn * dx + c * dy
        out += [(lx - s * dp_loc[:, 0]) / sig_t, (ly - s * dp_loc[:, 1]) / sig_t, (TH[1:] - TH[:-1] - dth) / sig_th]
        out.append((TH - h) / cfg.sig_compass)
        g = 1e3 if len(anchors) >= 2 else 0.01     # anchors fix the gauge themselves; otherwise pin node 0
        out.append(np.array([(X[0] - pxy[0, 0]) / g, (Y[0] - pxy[0, 1]) / g, 0.0]))
        out.append(np.array([(s - 1) / cfg.sig_scale]))
        if L:
            a = np.array([l[0] for l in L]); b = np.array([l[1] for l in L])
            dxl, dyl = X[a] - X[b], Y[a] - Y[b]
            dd = np.hypot(dxl, dyl) + 1e-9
            f = np.maximum(0.0, 1.0 - cfg.loop_radius / dd)       # zero penalty inside the matching radius
            out += [dxl * f / cfg.sig_loop, dyl * f / cfg.sig_loop]
        if label_pairs:
            a = np.array([l[0] for l in label_pairs]); b = np.array([l[1] for l in label_pairs])
            out += [(X[a] - X[b]) / cfg.sig_label, (Y[a] - Y[b]) / cfg.sig_label]
        if anchors:
            ia = np.array([a[0] for a in anchors]); xy = np.array([a[1] for a in anchors])
            out += [(X[ia] - xy[:, 0]) / cfg.sig_label, (Y[ia] - xy[:, 1]) / cfg.sig_label]
        return np.concatenate(out)

    x0 = np.concatenate([pxy[:, 0], pxy[:, 1], h, [1.0]])
    r0 = res(x0)
    # sparsity pattern
    S = lil_matrix((len(r0), 3 * N + 1), dtype=int)
    r = 0
    for k in range(N - 1):
        for rr in range(2):
            for v in (k, k + 1, N + k, N + k + 1, 2 * N + k, 3 * N):
                S[r + rr * (N - 1) + k, v] = 1
    r += 2 * (N - 1)
    for k in range(N - 1):
        for v in (2 * N + k, 2 * N + k + 1):
            S[r + k, v] = 1
    r += N - 1
    for k in range(N):
        S[r + k, 2 * N + k] = 1
    r += N
    S[r, 0] = S[r + 1, N] = 1
    r += 3
    S[r, 3 * N] = 1
    r += 1
    def pair_block(pairs, r):
        m = len(pairs)
        for k, (a, b, *_) in enumerate(pairs):
            for v in (a, b):
                S[r + k, v] = 1
                S[r + m + k, N + v] = 1
        return r + 2 * m
    r = pair_block(L, r)
    r = pair_block(label_pairs, r)
    for k, (i, _) in enumerate(anchors):
        S[r + k, i] = 1
        S[r + len(anchors) + k, N + i] = 1
    if label_pairs or anchors:
        # user-supplied marker information is trusted: a plain least-squares pass first, so that large initial
        # anchor residuals are not discounted as outliers by the robust loss (which then leaves the graph in the wrong basin)
        x0 = least_squares(res, x0, jac_sparsity=S, loss="linear", method="trf", x_scale="jac", max_nfev=200).x
    sol = least_squares(res, x0, jac_sparsity=S, loss="soft_l1", f_scale=cfg.robust_scale, method="trf",
                        x_scale="jac", max_nfev=400)
    X, Y, TH, s = unpack(sol.x)
    return np.stack([X, Y], 1), TH, float(s), sol


def run_posegraph(ses: Session, pdr: PDRResult, groups: list[list[str]], cfg: PGConfig | None = None,
                  marker_xy: dict[int, np.ndarray] | None = None) -> PoseGraphResult:
    cfg = cfg or PGConfig()
    nodes = build_nodes(ses, pdr)
    _, M = rssi_matrix(ses)
    F = M[nodes.scan_idx]
    loops, d0, thr = ([], 0.0, 0.0)
    if cfg.use_loops:
        loops, d0, thr = find_loops(nodes, F, cfg)
    label_pairs, anchors = [], []
    pdr_pos = pdr.pos
    if cfg.use_labels and ses.waypoints and marker_xy:
        # labelled ablation with known marker coordinates: express the PDR frame in the marker frame first
        from common.geometry import umeyama_rigid
        src = [pdr.pos(w["t_s"] + 1.5) for w in ses.waypoints if w["marker"] in marker_xy]
        dst = [marker_xy[w["marker"]] for w in ses.waypoints if w["marker"] in marker_xy]
        if len(src) >= 2:
            R0, t0 = umeyama_rigid(np.array(src), np.array(dst))
            ang = float(np.arctan2(R0[1, 0], R0[0, 0]))
            nodes.pdr_xy = nodes.pdr_xy @ R0.T + t0
            nodes.pdr_h = nodes.pdr_h + ang
            pdr_pos = lambda t, _p=pdr.pos: _p(t) @ R0.T + t0
    if cfg.use_labels and ses.waypoints:
        last: dict[int, int] = {}
        for w in ses.waypoints:
            # the button is pressed on arrival; the scans taken while standing there are within a few seconds
            cand = np.where((nodes.t >= w["t_s"] - 0.3) & (nodes.t <= w["t_s"] + 3.5))[0]
            # only scans taken while still standing at the marker: no detected step since the button press
            cand = np.array([c for c in cand if pdr.steps_between(w["t_s"] + 0.3, nodes.t[c]) == 0], dtype=int)
            if len(cand) == 0:
                continue
            i = int(cand[np.argmin(np.abs(nodes.t[cand] - (w["t_s"] + 1.5)))])
            m = w["marker"]
            if m in last and abs(nodes.t[i] - nodes.t[last[m]]) > 5:
                label_pairs.append((last[m], i))
            last[m] = i
            if marker_xy and m in marker_xy:
                anchors.append((i, marker_xy[m]))
    P, th, s, sol = optimise(nodes, loops, cfg, label_pairs, anchors)
    # drop closures that disagree after the first solve, then re-solve
    keep = [(a, b, d) for a, b, d in loops if np.linalg.norm(P[a] - P[b]) < cfg.loop_radius + 3.0 * cfg.sig_loop]
    if len(keep) < len(loops):
        P, th, s, sol = optimise(nodes, keep, cfg, label_pairs, anchors)
    res = PoseGraphResult(nodes, P, th, s, loops, keep, {"d0": d0, "thr": thr, "cost": float(sol.cost),
                                                        "labels": len(label_pairs), "anchors": len(anchors)})
    res._pdr = pdr_pos
    return res
