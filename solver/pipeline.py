"""End-to-end pipeline: PDR -> BSSID grouping -> pose graph -> AP localisation -> wall fitting -> confidence."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from common.geometry import rot2, sample_walls
from common.session import Session
from solver.aploc import APModel, localise_aps
from solver.dataset import Dataset, build_dataset
from solver.grouping import GroupingConfig, group_bssids
from solver.omp import OMPConfig, omp_segments
from solver.pdr import PDRConfig, PDRResult, run_pdr
from solver.posegraph import PGConfig, PoseGraphResult, run_posegraph
from solver.wallfit import WallFit, WallFitConfig, fit_walls


@dataclass
class PipelineConfig:
    pdr: PDRConfig = field(default_factory=PDRConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    pg: PGConfig = field(default_factory=PGConfig)
    omp: OMPConfig = field(default_factory=OMPConfig)
    wall: WallFitConfig = field(default_factory=lambda: WallFitConfig(lam_a=0.02))
    theta_step_deg: float = 5.0
    n_theta_cands: int = 3
    wall_bic: float = 6.0          # extra loss charged per wall when comparing Manhattan-angle candidates
    n_boot: int = 6
    boot_block: int = 6
    boot_frac: float = 0.7
    support_tol_m: float = 0.5
    rssi_floor: float = -88.0      # drop scan results weaker than this (censored near the sensitivity limit)
    seed: int = 0
    theta_override_rad: float | None = None   # diagnostics only: skip the angle search (oracle experiments)
    marker_xy: dict | None = None  # only used with pg.use_labels (ablation)
    verbose: bool = False


@dataclass
class Result:
    session: Session
    pdr: PDRResult
    groups: list[list[str]]
    pg: PoseGraphResult
    ds: Dataset
    aps0: APModel                  # multilateration-only estimate
    fit: WallFit | None
    support: np.ndarray            # (K,) bootstrap support in [0,1]
    theta_scan: list               # [(theta_deg, objective)]
    candidates: list               # [(theta_deg, total_loss, n_walls)]
    timings: dict = field(default_factory=dict)

    @property
    def ap_list(self):
        """[{'bssids': [...], 'xy': [x,y]}] using the jointly refined positions when available."""
        xy = self.fit.ap_xy if self.fit is not None else self.aps0.xy
        return [{"bssids": g, "xy": xy[j].tolist()} for j, g in enumerate(self.ds.groups)]


def _local_minima(theta_deg, objs, k, min_sep=10.0):
    n = len(objs)
    cand = [i for i in range(n) if objs[i] <= objs[(i - 1) % n] and objs[i] <= objs[(i + 1) % n]]
    cand.sort(key=lambda i: objs[i])
    out = []
    for i in cand:
        if all(min(abs(theta_deg[i] - theta_deg[j]), 90 - abs(theta_deg[i] - theta_deg[j])) >= min_sep for j in out):
            out.append(i)
        if len(out) == k:
            break
    return out


def wall_support(walls: np.ndarray, others: list[np.ndarray], tol: float) -> np.ndarray:
    """For each wall: mean over other runs of the fraction of its length within tol of a wall of that run."""
    from scipy.spatial import cKDTree
    if len(walls) == 0 or not others:
        return np.zeros(len(walls))
    pts, idx = sample_walls(walls, 0.05)
    sup = np.zeros(len(walls))
    for o in others:
        if len(o) == 0:
            continue
        op, _ = sample_walls(o, 0.05)
        d = cKDTree(op).query(pts)[0]
        for k in range(len(walls)):
            sup[k] += (d[idx == k] <= tol).mean()
    return sup / len(others)


def _fit_at(ds, aps0, theta, cfg: PipelineConfig, node_mask=None):
    seg, att, obj, n, _ = omp_segments(ds, aps0.xy, theta, cfg.omp, node_mask)
    fit = fit_walls(ds, aps0, theta, seg, att, cfg.wall, node_mask)
    return fit


def run_pipeline(ses: Session, cfg: PipelineConfig | None = None) -> Result:
    cfg = cfg or PipelineConfig()
    problems = []
    if len(ses.scans) < 15:
        problems.append(f"only {len(ses.scans)} WiFi scans (need >= 15; is scan throttling still on?)")
    for k in ("acc", "rotvec"):
        if k not in ses.imu or len(ses.imu[k][0]) < 100:
            problems.append(f"missing/short IMU stream '{k}' (needed for dead reckoning)")
    if problems:
        raise ValueError("session unusable: " + "; ".join(problems))
    tm = {}
    t = time.time()
    pdr = run_pdr(ses, cfg.pdr)
    if len(pdr.step_t) < 10:
        raise ValueError(f"only {len(pdr.step_t)} steps detected; was the phone held flat while walking? "
                         f"(PDRConfig.peak_height={cfg.pdr.peak_height} m/s^2)")
    groups = group_bssids(ses, cfg.grouping)
    tm["pdr+grouping"] = time.time() - t; t = time.time()
    pg = run_posegraph(ses, pdr, groups, cfg.pg, cfg.marker_xy)
    tm["posegraph"] = time.time() - t; t = time.time()
    ds = build_dataset(ses, pg, groups, rssi_floor=cfg.rssi_floor)
    aps0 = localise_aps(ds)
    tm["aploc"] = time.time() - t; t = time.time()
    # Manhattan angle: coarse sweep of the greedy-selection objective, then joint refinement of the best few
    if cfg.theta_override_rad is not None:
        thetas, objs = np.array([np.rad2deg(cfg.theta_override_rad) % 90.0]), np.zeros(1)
        picks = [0]
    else:
        thetas = np.arange(0.0, 90.0, cfg.theta_step_deg)
        objs = np.array([omp_segments(ds, aps0.xy, np.deg2rad(th), cfg.omp)[2] for th in thetas])
        picks = _local_minima(thetas, objs, cfg.n_theta_cands)
    scan = list(zip(thetas.tolist(), objs.tolist()))
    tm["theta_scan"] = time.time() - t; t = time.time()
    best, cands = None, []
    for i in picks:
        fit = _fit_at(ds, aps0, np.deg2rad(thetas[i]), cfg)
        total = fit.loss + cfg.wall_bic * len(fit.walls)
        cands.append((float(thetas[i]), float(total), len(fit.walls)))
        if cfg.verbose:
            print(f"theta {thetas[i]:5.1f}: loss {fit.loss:8.1f} walls {len(fit.walls)} total {total:.1f}")
        if best is None or total < best[0]:
            best = (total, fit)
    fit = best[1]
    tm["wall_fit"] = time.time() - t; t = time.time()
    # bootstrap ensemble (blocks of consecutive nodes) -> per-wall support
    rng = np.random.default_rng(cfg.seed)
    N = len(ds.node_t)
    boots = []
    for b in range(cfg.n_boot):
        mask = np.zeros(N, bool)
        while mask.mean() < cfg.boot_frac:
            s = int(rng.integers(0, N))
            mask[s:s + cfg.boot_block] = True
        try:
            fb = _fit_at(ds, aps0, fit.theta, cfg, mask)
            boots.append(fb.walls)
        except Exception as e:  # a degenerate resample must not kill the run
            if cfg.verbose:
                print("bootstrap failed:", e)
    support = wall_support(fit.walls, boots, cfg.support_tol_m)
    tm["bootstrap"] = time.time() - t
    return Result(ses, pdr, groups, pg, ds, aps0, fit, support, scan, cands, tm)
