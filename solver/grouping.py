"""Stage 2: cluster BSSIDs into physical APs.

A physical box usually broadcasts several BSSIDs (2.4/5 GHz, guest SSIDs) whose MACs differ by a few counts, with
the locally-administered bit sometimes flipped. MAC closeness only nominates candidates; a merge additionally
needs the RSSI traces to move together (high correlation, small residual spread after removing a constant offset),
because mesh extenders from the same vendor have MACs right next to the router's but sit somewhere else.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.session import Session


@dataclass
class GroupingConfig:
    max_mac_dist: int = 32          # |last 24 bits| difference after clearing the LA bit (same first 3 bytes)
    min_common_scans: int = 6
    min_corr: float = 0.75
    max_resid_sd_db: float = 4.2
    max_mean_offset_db: float = 22.0
    low_var_db: float = 4.0         # weak/far APs barely vary along the walk, so correlation is uninformative:
                                    # there the (small) residual spread alone decides
    sparse_mac_dist: int = 3        # too few common scans: merge only if MACs are this close...
    sparse_max_offset_db: float = 14.0   # ...and mean RSSI differs plausibly


def mac_int(mac: str) -> int:
    return int(mac.replace(":", ""), 16)


def mac_key(mac: str) -> tuple[int, int]:
    """(OUI with LA bit cleared, low 24 bits)."""
    v = mac_int(mac) & ~(0x02 << 40)
    return v >> 24, v & 0xFFFFFF


def mac_distance(a: str, b: str) -> float:
    ka, kb = mac_key(a), mac_key(b)
    return abs(ka[1] - kb[1]) if ka[0] == kb[0] else np.inf


def rssi_matrix(ses: Session, bssids: list[str] | None = None):
    bssids = bssids or ses.bssids()
    idx = {b: i for i, b in enumerate(bssids)}
    M = np.full((len(ses.scans), len(bssids)), np.nan)
    for si, sc in enumerate(ses.scans):
        for o in sc.obs:
            if o.bssid in idx:
                M[si, idx[o.bssid]] = o.rssi
    return bssids, M


def pair_stats(x: np.ndarray, y: np.ndarray):
    ok = ~np.isnan(x) & ~np.isnan(y)
    n = int(ok.sum())
    if n < 3:
        return n, np.nan, np.nan, np.nan, np.nan
    d = x[ok] - y[ok]
    corr = np.corrcoef(x[ok], y[ok])[0, 1] if x[ok].std() > 1e-6 and y[ok].std() > 1e-6 else 0.0
    return n, float(corr), float(d.std()), float(d.mean()), float(max(x[ok].std(), y[ok].std()))


def group_bssids(ses: Session, cfg: GroupingConfig | None = None) -> list[list[str]]:
    cfg = cfg or GroupingConfig()
    bssids, M = rssi_matrix(ses)
    n = len(bssids)
    cand = []
    for i in range(n):
        for j in range(i + 1, n):
            md = mac_distance(bssids[i], bssids[j])
            if md > cfg.max_mac_dist:
                continue
            nc, corr, sd, mean, spread = pair_stats(M[:, i], M[:, j])
            if nc >= cfg.min_common_scans:
                good = (corr >= cfg.min_corr or spread < cfg.low_var_db) and sd <= cfg.max_resid_sd_db and abs(mean) <= cfg.max_mean_offset_db
                score = sd
            else:
                good = md <= cfg.sparse_mac_dist and (nc < 3 or abs(mean) <= cfg.sparse_max_offset_db)
                score = 99.0
            if good:
                cand.append((score, i, j))
    cand.sort()
    groups = {i: {i} for i in range(n)}
    ok = {(i, j): True for _, i, j in cand}
    for _, i, j in cand:
        gi, gj = groups[i], groups[j]
        if gi is gj:
            continue
        # complete linkage: every cross pair must itself be a candidate (or never co-observed)
        if all(ok.get((min(a, b), max(a, b)), False) for a in gi for b in gj):
            merged = gi | gj
            for k in merged:
                groups[k] = merged
    seen, out = set(), []
    for i in range(n):
        g = groups[i]
        if id(g) not in seen:
            seen.add(id(g))
            out.append(sorted(bssids[k] for k in g))
    return out
