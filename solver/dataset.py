"""Assemble the arrays used by AP localisation and wall fitting from session + groups + corrected trajectory."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.session import Session
from solver.posegraph import PoseGraphResult


@dataclass
class Dataset:
    bssids: list[str]
    bssid_ap: np.ndarray      # (B,) physical AP index per BSSID
    bssid_freq: np.ndarray    # (B,) MHz
    groups: list[list[str]]   # per AP index
    node_xy: np.ndarray       # (N,2)
    node_t: np.ndarray
    # RSS observations
    o_node: np.ndarray
    o_bssid: np.ndarray
    o_rssi: np.ndarray
    # RTT observations
    r_xy: np.ndarray          # (R,2) phone position at range time
    r_ap: np.ndarray
    r_bssid: np.ndarray
    r_dist: np.ndarray
    r_std: np.ndarray

    @property
    def n_ap(self):
        return len(self.groups)


def build_dataset(ses: Session, pg: PoseGraphResult, groups: list[list[str]], min_obs: int = 8,
                  min_max_rssi: float = -90.0) -> Dataset:
    node_of_scan = {int(s): k for k, s in enumerate(pg.nodes.scan_idx)}
    counts: dict[str, int] = {}
    for k in node_of_scan:
        for o in ses.scans[k].obs:
            counts[o.bssid] = counts.get(o.bssid, 0) + 1
    keep_groups = [[b for b in g if counts.get(b, 0) >= min_obs] for g in groups]
    keep_groups = [g for g in keep_groups if g]
    bssids = [b for g in keep_groups for b in g]
    bidx = {b: i for i, b in enumerate(bssids)}
    ap_of = {b: j for j, g in enumerate(keep_groups) for b in g}
    freq = np.zeros(len(bssids))
    on, ob, orr = [], [], []
    for k, node in node_of_scan.items():
        for o in ses.scans[k].obs:
            if o.bssid in bidx:
                on.append(node); ob.append(bidx[o.bssid]); orr.append(o.rssi)
                freq[bidx[o.bssid]] = o.freq_mhz
    rx, ra, rb, rd, rs = [], [], [], [], []
    ts = [r.t_s for r in ses.rtts if r.dist_m is not None and r.bssid in bidx]
    pos = pg.at(ts) if ts else np.zeros((0, 2))
    k = 0
    for r in ses.rtts:
        if r.dist_m is None or r.bssid not in bidx:
            continue
        rx.append(pos[k]); k += 1
        ra.append(ap_of[r.bssid]); rb.append(bidx[r.bssid]); rd.append(r.dist_m)
        rs.append(max(r.std_m or 0.5, 0.3))
    return Dataset(bssids, np.array([ap_of[b] for b in bssids]), freq, keep_groups, pg.P.copy(), pg.nodes.t.copy(),
                   np.array(on), np.array(ob), np.array(orr, float),
                   np.array(rx).reshape(-1, 2), np.array(ra, int), np.array(rb, int), np.array(rd, float), np.array(rs, float))
