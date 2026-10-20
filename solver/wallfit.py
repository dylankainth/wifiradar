"""Stage 5: differentiable wall fitting (JAX).

Forward model for a phone position p and AP x:
    RSS  = c_b - 10 n log10 |p - x|  - w_band * sum_k a_k * cross_k(p, x)
    RTT  = |p - x|                   + alpha  * sum_k a_k * cross_k(p, x)
with soft crossing indicators cross_k in [0,1] (sigmoids of signed distances, annealed temperature tau), so the
loss is differentiable in the wall endpoints and in the AP positions. Priors: walls close to the Manhattan axes,
no tiny walls, mild sparsity. Optimised with L-BFGS-B on (jit'd) JAX gradients, then pruned by leave-one-out
evidence (a BIC-like threshold).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

from common.geometry import rot2
from solver.aploc import APModel
from solver.dataset import Dataset

jax.config.update("jax_enable_x64", True)


@dataclass
class WallFitConfig:
    sigma_rss: float = 3.0
    huber: float = 2.0
    sigma_rtt_min: float = 0.4
    lam_orth: float = 25.0
    lam_a: float = 0.3
    lam_minlen: float = 30.0
    min_len: float = 0.6
    lam_ap: float = 0.02           # weak pull of AP positions towards their multilateration estimate
    taus: tuple = (0.7, 0.35, 0.15)
    iters: int = 60
    prune_thr: float = 12.0        # leave-one-out loss increase needed to keep a wall
    prune_refit_iters: int = 25
    a_init_min: float = 2.0
    max_walls: int = 48
    box_margin: float = 3.0
    merge_dist: float = 0.35


@dataclass
class WallFit:
    walls: np.ndarray            # (K,4) solver frame, alive walls only
    atten: np.ndarray            # (K,) dB
    dL: np.ndarray               # (K,) leave-one-out loss increase (evidence)
    ap_xy: np.ndarray
    c: np.ndarray
    n: float
    w5: float
    alpha: float
    loss: float
    theta: float                 # final Manhattan angle (initial theta + learned offset)
    history: list = field(default_factory=list)


def _sig(x):
    return jax.nn.sigmoid(x)


def soft_cross(p, q, A, B, tau):
    """p,q: (M,2); A,B: (K,2) -> (M,K) soft indicator that segment pq crosses segment AB."""
    ab = B - A
    L = jnp.sqrt((ab ** 2).sum(-1) + 1e-9)
    n = jnp.stack([-ab[:, 1], ab[:, 0]], -1) / L[:, None]
    dp = jnp.einsum("mkc,kc->mk", p[:, None, :] - A[None], n)
    dq = jnp.einsum("mkc,kc->mk", q[:, None, :] - A[None], n)
    s1 = _sig(dp / tau) * _sig(-dq / tau) + _sig(-dp / tau) * _sig(dq / tau)
    pq = q - p
    Lpq = jnp.sqrt((pq ** 2).sum(-1) + 1e-9)
    nq = jnp.stack([-pq[:, 1], pq[:, 0]], -1) / Lpq[:, None]
    ea = jnp.einsum("mkc,mc->mk", A[None] - p[:, None, :], nq)
    eb = jnp.einsum("mkc,mc->mk", B[None] - p[:, None, :], nq)
    s2 = _sig(ea / tau) * _sig(-eb / tau) + _sig(-ea / tau) * _sig(eb / tau)
    return s1 * s2


def _ph(z, d):
    return d * d * (jnp.sqrt(1 + (z / d) ** 2) - 1)


class Problem:
    """Holds the (rotated-frame) data as jnp arrays and the jit'd loss."""

    def __init__(self, ds: Dataset, ap0: np.ndarray, theta: float, cfg: WallFitConfig, node_mask=None, obs_weight=None):
        self.cfg, self.theta, self.ds = cfg, theta, ds
        R = rot2(-theta)
        nodes = ds.node_xy @ R.T
        o_ap = ds.bssid_ap[ds.o_bssid]
        sel = np.ones(len(ds.o_node), bool) if node_mask is None else node_mask[ds.o_node]
        on, ob, oa, orr = ds.o_node[sel], ds.o_bssid[sel], o_ap[sel], ds.o_rssi[sel]
        key = on.astype(np.int64) * 10_000 + oa
        uk, inv = np.unique(key, return_inverse=True)
        pn, pa = uk // 10_000, uk % 10_000
        w = np.ones(len(orr)) if obs_weight is None else obs_weight[sel]
        self.data = dict(
            pair_p=jnp.asarray(nodes[pn]), pair_ap=jnp.asarray(pa), inv=jnp.asarray(inv),
            ob=jnp.asarray(ob), rssi=jnp.asarray(orr), w=jnp.asarray(w),
            band5=jnp.asarray((ds.bssid_freq > 4000).astype(float)),
            r_p=jnp.asarray(ds.r_xy @ R.T), r_ap=jnp.asarray(ds.r_ap), r_d=jnp.asarray(ds.r_dist),
            r_sig=jnp.asarray(np.maximum(ds.r_std, cfg.sigma_rtt_min)),
            ap0=jnp.asarray(ap0 @ R.T),
        )
        self.nodes = nodes
        self.lo = nodes.min(0) - cfg.box_margin
        self.hi = nodes.max(0) + cfg.box_margin
        self.has_rtt = len(ds.r_dist) > 0
        self._loss = jax.jit(self._loss_fn)
        self._vg = jax.jit(jax.value_and_grad(self._loss_fn))
        self._parts = jax.jit(self._loss_parts)

    # -- model ---------------------------------------------------------------------------------------
    def _loss_parts(self, params, alive, tau):
        d, cfg = self.data, self.cfg
        W = params["walls"]
        A, B = W[:, :2], W[:, 2:]
        a = jax.nn.softplus(params["a"]) * alive
        X = params["ap"]
        n = params["n"]
        Q = X[d["pair_ap"]]
        cr = soft_cross(d["pair_p"], Q, A, B, tau)
        wl = cr @ a
        dist = jnp.maximum(jnp.linalg.norm(d["pair_p"] - Q, axis=1), 0.5)
        wband = 1.0 + (params["w5"] - 1.0) * d["band5"][d["ob"]]
        pred = params["c"][d["ob"]] - 10 * n * jnp.log10(dist[d["inv"]]) - wband * wl[d["inv"]]
        z = (d["rssi"] - pred) / cfg.sigma_rss
        rss = (d["w"] * _ph(z, cfg.huber)).sum()
        if self.has_rtt:
            Qr = X[d["r_ap"]]
            crr = soft_cross(d["r_p"], Qr, A, B, tau)
            pr = jnp.linalg.norm(d["r_p"] - Qr, axis=1) + params["alpha"] * (crr @ a)
            rtt = _ph((d["r_d"] - pr) / d["r_sig"], 1.5).sum()
        else:
            rtt = 0.0
        ab = B - A
        L2 = (ab ** 2).sum(-1)
        L = jnp.sqrt(L2 + 1e-9)
        # Manhattan prior in a frame rotated by the (learned) small offset dth from the initial axes
        cs, sn = jnp.cos(params["dth"]), jnp.sin(params["dth"])
        abx, aby = cs * ab[:, 0] + sn * ab[:, 1], -sn * ab[:, 0] + cs * ab[:, 1]
        orth = cfg.lam_orth * (alive * (abx * aby) ** 2 / (L2 + 0.25)).sum()
        short = cfg.lam_minlen * (alive * jax.nn.relu(cfg.min_len - L) ** 2).sum()
        sparse = cfg.lam_a * (a * L).sum()
        app = cfg.lam_ap * ((X - d["ap0"]) ** 2).sum()
        return jnp.stack([rss, rtt, orth, short, sparse, app])

    def _loss_fn(self, params, alive, tau):
        return self._loss_parts(params, alive, tau).sum()

    # -- optimisation --------------------------------------------------------------------------------
    def _bounds(self, params, K):
        lo, hi = self.lo, self.hi
        b = {"walls": (np.tile(np.r_[lo, lo], (K, 1)), np.tile(np.r_[hi, hi], (K, 1))),
             "a": (np.full(K, -6.0), np.full(K, 30.0)),   # raw; a = softplus(raw) dB, so up to ~30 dB
             "ap": (np.tile(lo - 15, (params["ap"].shape[0], 1)), np.tile(hi + 15, (params["ap"].shape[0], 1))),
             "c": (np.full(params["c"].shape, -90.0), np.full(params["c"].shape, -10.0)),
             "n": (1.5, 4.5), "w5": (0.7, 1.8), "alpha": (0.0, 0.2), "dth": (-0.25, 0.25)}
        flat_lo, flat_hi = [], []
        for k in sorted(params):
            flat_lo.append(np.asarray(b[k][0], float).ravel()); flat_hi.append(np.asarray(b[k][1], float).ravel())
        return list(zip(np.concatenate(flat_lo), np.concatenate(flat_hi)))

    def optimise(self, params, alive, tau, iters):
        params = {k: jnp.asarray(v, float) for k, v in params.items()}
        flat, unravel = ravel_pytree(params)
        alive = jnp.asarray(alive, float)

        def fg(x):
            v, g = self._vg(unravel(jnp.asarray(x)), alive, tau)
            return float(v), np.asarray(ravel_pytree(g)[0], float)

        bounds = self._bounds(params, params["walls"].shape[0])
        r = minimize(fg, np.asarray(flat), jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": iters})
        return {k: np.asarray(v) for k, v in unravel(jnp.asarray(r.x)).items()}, float(r.fun)

    def loo(self, params, alive, tau):
        """Loss increase when each alive wall is removed (no refit)."""
        base = float(self._loss(params, jnp.asarray(alive, float), tau))
        out = np.zeros(len(alive))
        for k in np.where(alive > 0)[0]:
            al = alive.copy(); al[k] = 0
            out[k] = float(self._loss(params, jnp.asarray(al, float), tau)) - base
        return out, base


def merge_parallel(walls, a, alive, dist_tol):
    """Merge near-duplicate (same orientation, close, overlapping) segments into one; returns new alive mask."""
    alive = alive.copy()
    horiz = np.abs(walls[:, 2] - walls[:, 0]) >= np.abs(walls[:, 3] - walls[:, 1])
    changed = True
    while changed:
        changed = False
        idx = np.where(alive > 0)[0]
        for i in idx:
            for j in idx:
                if j <= i or alive[i] == 0 or alive[j] == 0 or horiz[i] != horiz[j]:
                    continue
                if horiz[i]:
                    ci, cj = (walls[i, 1] + walls[i, 3]) / 2, (walls[j, 1] + walls[j, 3]) / 2
                    ai, bi = sorted((walls[i, 0], walls[i, 2])); aj, bj = sorted((walls[j, 0], walls[j, 2]))
                else:
                    ci, cj = (walls[i, 0] + walls[i, 2]) / 2, (walls[j, 0] + walls[j, 2]) / 2
                    ai, bi = sorted((walls[i, 1], walls[i, 3])); aj, bj = sorted((walls[j, 1], walls[j, 3]))
                if abs(ci - cj) < dist_tol and min(bi, bj) - max(ai, aj) > -0.3:
                    lo_, hi_ = min(ai, aj), max(bi, bj)
                    wi, wj = (bi - ai), (bj - aj)
                    c = (ci * wi + cj * wj) / (wi + wj)
                    walls[i] = [lo_, c, hi_, c] if horiz[i] else [c, lo_, c, hi_]
                    a[i] = max(a[i], a[j])
                    alive[j] = 0
                    changed = True
    return alive


def fit_walls(ds: Dataset, aps: APModel, theta: float, init_walls: np.ndarray, init_atten: np.ndarray,
              cfg: WallFitConfig | None = None, node_mask=None, obs_weight=None, verbose=False) -> WallFit:
    """init_walls/atten are given in the Manhattan frame (rotated by -theta from the solver frame)."""
    cfg = cfg or WallFitConfig()
    prob = Problem(ds, aps.xy, theta, cfg, node_mask, obs_weight)
    K0 = min(len(init_walls), cfg.max_walls)
    order = np.argsort(-init_atten * (np.hypot(init_walls[:, 2] - init_walls[:, 0], init_walls[:, 3] - init_walls[:, 1])))[:K0]
    walls0 = init_walls[order].copy()
    K = int(np.ceil(max(K0, 1) / 8) * 8)
    W = np.zeros((K, 4)); W[:K0] = walls0
    W[K0:] = np.r_[prob.lo, prob.lo]
    att = np.full(K, 1.0); att[:K0] = np.maximum(init_atten[order], cfg.a_init_min)
    alive = np.zeros(K); alive[:K0] = 1
    inv_softplus = lambda y: np.log(np.expm1(np.maximum(y, 1e-3)))
    R = rot2(-theta)
    params = {"walls": W, "a": inv_softplus(att), "ap": aps.xy @ R.T, "c": aps.c.copy(), "n": np.array(aps.n),
              "w5": np.array(1.15), "alpha": np.array(0.05), "dth": np.array(0.0)}
    hist = []
    alive = merge_parallel(W, att, alive, cfg.merge_dist)
    params["walls"] = W
    for tau in cfg.taus:
        params, f = prob.optimise(params, alive, tau, cfg.iters)
        hist.append(("opt", tau, f, int(alive.sum())))
    tau = cfg.taus[-1]
    # prune by leave-one-out evidence
    while alive.sum() > 0:
        dl, base = prob.loo(params, alive, tau)
        dl_alive = np.where(alive > 0, dl, np.inf)
        k = int(np.argmin(dl_alive))
        if dl_alive[k] >= cfg.prune_thr:
            break
        alive[k] = 0
        params, f = prob.optimise(params, alive, tau, cfg.prune_refit_iters)
        hist.append(("prune", k, f, int(alive.sum())))
        if verbose:
            print("pruned", k, "dL", dl_alive[k], "left", int(alive.sum()))
    # merge duplicates that drifted together and finish
    Wn = np.array(params["walls"]); an = np.array(jax.nn.softplus(jnp.asarray(params["a"])))
    alive = merge_parallel(Wn, an, alive, cfg.merge_dist)
    params["walls"] = Wn
    params, f = prob.optimise(params, alive, tau, cfg.iters)
    dl, base = prob.loo(params, alive, tau)
    keep = np.where(alive > 0)[0]
    Wf = np.array(params["walls"])[keep]
    af = np.array(jax.nn.softplus(jnp.asarray(params["a"])))[keep]
    Rb = rot2(theta)
    walls_sol = np.hstack([Wf[:, :2] @ Rb.T, Wf[:, 2:] @ Rb.T])
    return WallFit(walls_sol, af, dl[keep], np.asarray(params["ap"]) @ Rb.T, np.asarray(params["c"]), float(params["n"]),
                   float(params["w5"]), float(params["alpha"]), base, theta + float(params["dth"]), hist)
