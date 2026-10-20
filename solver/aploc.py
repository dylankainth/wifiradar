"""Stage 4: AP localisation by log-distance multilateration (+ RTT ranges), ignoring walls.

Walls bias this (they inflate the apparent distance, hence also the fitted path-loss exponent), so it only
provides the initial guess; the joint wall fit refines AP positions, offsets and exponent together.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from solver.dataset import Dataset


@dataclass
class APModel:
    xy: np.ndarray        # (J,2)
    c: np.ndarray         # (B,) RSS at 1 m per BSSID
    n: float
    cost: float = 0.0


def localise_aps(ds: Dataset, n_init: float = 2.3, rss_min: float = -85.0, f_scale: float = 4.0,
                 fit_n: bool = True, n_bounds=(1.6, 4.0)) -> APModel:
    J, B = ds.n_ap, len(ds.bssids)
    keep = ds.o_rssi >= rss_min
    on, ob, orr = ds.o_node[keep], ds.o_bssid[keep], ds.o_rssi[keep]
    oa = ds.bssid_ap[ob]
    xy0 = np.zeros((J, 2))
    for j in range(J):
        m = oa == j
        if not m.any():
            xy0[j] = ds.node_xy.mean(0)
            continue
        w = 10 ** ((orr[m] - orr[m].max()) / 10.0)
        top = w > 0.1
        xy0[j] = (ds.node_xy[on[m]][top] * w[top, None]).sum(0) / w[top].sum()
    c0 = np.array([np.mean(orr[ob == b]) + 10 * n_init * np.log10(3.0) if (ob == b).any() else -45.0 for b in range(B)])
    use_rtt = len(ds.r_dist) > 0

    def unpack(x):
        return x[:2 * J].reshape(J, 2), x[2 * J:2 * J + B], (x[-1] if fit_n else n_init)

    def res(x):
        xy, c, n = unpack(x)
        d = np.maximum(np.linalg.norm(ds.node_xy[on] - xy[oa], axis=1), 0.3)
        r = [(orr - (c[ob] - 10 * n * np.log10(d))) / 1.0]
        if use_rtt:
            dr = np.linalg.norm(ds.r_xy - xy[ds.r_ap], axis=1)
            r.append((dr - ds.r_dist) / ds.r_std * f_scale / 1.0)   # RTT residual scaled to the same robust scale
        return np.concatenate(r)

    x0 = np.concatenate([xy0.ravel(), c0] + ([[n_init]] if fit_n else []))
    lb = np.concatenate([np.full(2 * J, -60), np.full(B, -80), [n_bounds[0]] if fit_n else []])
    ub = np.concatenate([np.full(2 * J, 60), np.full(B, -10), [n_bounds[1]] if fit_n else []])
    best = None
    rng = np.random.default_rng(0)
    for trial in range(4):
        x = x0.copy()
        if trial:
            x[:2 * J] += rng.normal(0, 2.0, 2 * J)
        s = least_squares(res, x, bounds=(lb, ub), loss="soft_l1", f_scale=f_scale)
        if best is None or s.cost < best.cost:
            best = s
    xy, c, n = unpack(best.x)
    return APModel(xy, c, float(n), float(best.cost))
