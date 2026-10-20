"""Ground-truth radio models used by the simulator: multi-wall RSS and RTT with NLOS excess + outliers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.geometry import segment_crossings


@dataclass
class RadioConfig:
    path_loss_n: float = 2.3
    p5g_offset_db: float = -7.0       # extra free-space loss at 5 GHz relative to 2.4
    wall_scale_5g: float = 1.2        # walls attenuate more at 5 GHz
    shadow_sigma_db: float = 2.5      # spatially correlated unmodelled shadowing
    shadow_corr_m: float = 2.0
    meas_sigma_db: float = 1.5        # per-scan noise
    detect_mid_dbm: float = -88.0
    # RTT
    rtt_sigma_m: float = 0.45
    rtt_excess_per_db_m: float = 0.05
    rtt_nlos_exp_m: float = 0.35
    rtt_outlier_p: float = 0.06
    rtt_fail_p: float = 0.1


class ShadowField:
    """Smooth random field via random Fourier features (Gaussian kernel with correlation length ell)."""

    def __init__(self, rng, sigma, ell, m=24):
        self.w = rng.normal(0, 1 / ell, (m, 2))
        self.ph = rng.uniform(0, 2 * np.pi, m)
        self.sigma = sigma
        self.m = m

    def __call__(self, xy):
        return self.sigma * np.sqrt(2 / self.m) * np.cos(np.atleast_2d(xy) @ self.w.T + self.ph).sum(1)


def wall_loss(p, ap_xy, walls, atten, scale=1.0):
    """Sum of attenuation (dB) for LOS from each p (M,2) to ap_xy; also number of crossings and the nominal dB sum."""
    if len(walls) == 0:
        z = np.zeros(len(p))
        return z, z.astype(int)
    q = np.broadcast_to(np.asarray(ap_xy, float), p.shape)
    c = segment_crossings(p, q, walls)
    return (c * atten[None, :]).sum(1) * scale, c.sum(1)


def rss_mean(p, ap_xy, p0_2g, freq_mhz, walls, atten, cfg: RadioConfig):
    p = np.atleast_2d(p)
    d = np.maximum(np.linalg.norm(p - np.asarray(ap_xy), axis=1), 0.5)
    is5 = freq_mhz > 4000
    p0 = p0_2g + (cfg.p5g_offset_db if is5 else 0.0)
    wl, _ = wall_loss(p, ap_xy, walls, atten, cfg.wall_scale_5g if is5 else 1.0)
    return p0 - 10 * cfg.path_loss_n * np.log10(d) - wl


def rtt_range(rng, p, ap_xy, walls, atten, cfg: RadioConfig):
    """One RTT range result: (dist_m or None, std_m). Excess delay from walls + exponential NLOS tail + outliers."""
    p = np.atleast_2d(p)
    d = np.linalg.norm(p - np.asarray(ap_xy), axis=1)
    wl, nw = wall_loss(p, ap_xy, walls, atten)
    out = []
    for i in range(len(p)):
        pf = cfg.rtt_fail_p * (1 + nw[i])
        if rng.random() < min(pf, 0.8):
            out.append((None, None))
            continue
        excess = cfg.rtt_excess_per_db_m * wl[i] + (rng.exponential(cfg.rtt_nlos_exp_m) if nw[i] > 0 else 0.0)
        if rng.random() < cfg.rtt_outlier_p:
            excess += rng.uniform(1.0, 8.0)
        meas = d[i] + excess + rng.normal(0, cfg.rtt_sigma_m)
        out.append((max(meas, 0.0), cfg.rtt_sigma_m * rng.uniform(0.5, 1.5)))
    return out
