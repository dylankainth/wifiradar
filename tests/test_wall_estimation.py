"""Wall estimation on a tiny synthetic scene with known geometry (no simulator, no PDR)."""
import numpy as np
import pytest

from common.geometry import segment_crossings
from solver.aploc import APModel
from solver.dataset import Dataset
from solver.omp import OMPConfig, omp_segments
from solver.wallfit import WallFitConfig, fit_walls

WALL = np.array([[6.0, 2.0, 6.0, 8.0]])
ATTEN = 8.0


def scene(seed=0, with_rtt=False):
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(1.5, 11, 1.0), np.arange(0.5, 10, 1.0))
    nodes = np.stack([gx.ravel(), gy.ravel()], 1)
    aps = np.array([[0.0, 5.0], [11.0, 1.0], [3.0, 10.5], [9.0, 9.0]])
    on, ob, orr = [], [], []
    for j, a in enumerate(aps):
        d = np.maximum(np.linalg.norm(nodes - a, axis=1), 0.5)
        cr = segment_crossings(nodes, np.tile(a, (len(nodes), 1)), WALL)[:, 0]
        rssi = -40 - 23 * np.log10(d) - ATTEN * cr + rng.normal(0, 0.7, len(nodes))
        on += list(range(len(nodes))); ob += [j] * len(nodes); orr += list(rssi)
    ds = Dataset(
        bssids=[f"aa:00:00:00:00:0{j}" for j in range(4)], bssid_ap=np.arange(4), bssid_freq=np.full(4, 2437.0),
        groups=[[f"aa:00:00:00:00:0{j}"] for j in range(4)], node_xy=nodes, node_t=np.arange(len(nodes), dtype=float),
        o_node=np.array(on), o_bssid=np.array(ob), o_rssi=np.array(orr),
        r_xy=np.zeros((0, 2)), r_ap=np.zeros(0, int), r_bssid=np.zeros(0, int), r_dist=np.zeros(0), r_std=np.zeros(0))
    return ds, aps


def test_omp_finds_the_wall():
    ds, aps = scene()
    seg, att, obj, n, info = omp_segments(ds, aps, 0.0, OMPConfig())
    assert len(seg) >= 1
    # a vertical segment near x = 6 overlapping y in [2, 8] carries most of the attenuation
    vert = [(s, a) for s, a in zip(seg, att) if abs(s[0] - s[2]) < 1e-6 and abs(s[0] - 6.0) < 0.5]
    assert vert, seg
    s, a = max(vert, key=lambda sa: sa[0][3] - sa[0][1])
    assert s[3] - s[1] > 3.0 and 4.0 < a < 12.0


def test_omp_with_no_wall_returns_little():
    ds, aps = scene()
    ds.o_rssi = np.array([-40 - 23 * np.log10(max(np.linalg.norm(ds.node_xy[n] - aps[b]), .5)) for n, b in zip(ds.o_node, ds.o_bssid)])
    ds.o_rssi += np.random.default_rng(1).normal(0, 0.7, len(ds.o_rssi))
    seg, att, *_ = omp_segments(ds, aps, 0.0, OMPConfig())
    assert len(seg) <= 1 or att.max() < 2.5


def test_joint_fit_keeps_and_refines_wall_and_recovers_ap_offsets():
    ds, aps = scene()
    seg, att, *_ = omp_segments(ds, aps, 0.0, OMPConfig())
    start = APModel(aps + np.random.default_rng(3).normal(0, 0.4, aps.shape), np.full(4, -40.0), 2.3)
    fit = fit_walls(ds, start, 0.0, seg, att, WallFitConfig(lam_a=0.02))
    assert len(fit.walls) >= 1
    best = fit.walls[np.argmax(fit.dL)]
    assert abs((best[0] + best[2]) / 2 - 6.0) < 0.35 and abs(best[0] - best[2]) < 0.5   # vertical, at x ~ 6
    assert 4.0 < fit.atten[np.argmax(fit.dL)] < 12.0
    assert np.abs(fit.ap_xy - aps).max() < 1.0
    assert fit.dL.max() > 12.0     # evidence beats the pruning threshold
    assert abs(fit.n - 2.3) < 0.5
