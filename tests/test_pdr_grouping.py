import json

import numpy as np
import pytest

from common.session import load_session
from eval.truth import load_truth
from sim.generate import SimConfig, generate
from solver.grouping import group_bssids, mac_distance
from solver.pdr import PDRConfig, run_pdr


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    p = generate(SimConfig(seed=3), tmp_path_factory.mktemp("s") / "s")
    return p, load_session(p), json.loads((p / "truth.json").read_text())


def test_step_detection_matches_truth(sim):
    _, ses, T = sim
    res = run_pdr(ses)
    true_steps = np.array(T["step_t"])
    assert abs(len(res.step_t) - len(true_steps)) <= 0.06 * len(true_steps) + 2
    # most detected steps are within 0.25 s of a true step
    d = np.abs(res.step_t[:, None] - true_steps[None, :]).min(1)
    assert (d < 0.25).mean() > 0.9


def test_no_steps_when_stationary(sim):
    _, ses, T = sim
    res = run_pdr(ses)
    first_pause_end = T["visits"][0]["t_s"] + 2.0
    assert (res.step_t < first_pause_end).sum() == 0


def test_pdr_path_length(sim):
    _, ses, T = sim
    res = run_pdr(ses)
    true_xy = np.array(T["traj"]["xy"])
    true_len = np.linalg.norm(np.diff(true_xy, axis=0), axis=1).sum()
    est_len = np.linalg.norm(np.diff(res.step_xy, axis=0), axis=1).sum()
    assert abs(est_len - true_len) / true_len < 0.1


def test_heading_modes_all_run(sim):
    _, ses, _ = sim
    for mode in ("rotvec", "gyro", "fused"):
        r = run_pdr(ses, PDRConfig(heading_mode=mode))
        assert np.isfinite(r.step_xy).all()


def test_mac_distance_clears_la_bit():
    assert mac_distance("c4:e9:84:5a:10:10", "c6:e9:84:5a:10:12") == 2
    assert mac_distance("c4:e9:84:5a:10:10", "00:e9:84:5a:10:12") == np.inf


def test_grouping_recovers_physical_aps(sim):
    p, ses, _ = sim
    gt = load_truth(p)
    groups = {frozenset(g) for g in group_bssids(ses)}
    seen = set(ses.bssids())
    for ap in gt.aps:
        want = frozenset(b for b in ap.bssids if b in seen)
        if want:
            assert want in groups, (ap.name, want, groups)


def test_grouping_does_not_merge_extender_with_router(sim):
    _, ses, _ = sim
    groups = group_bssids(ses)
    router, ext = "c4:e9:84:5a:10:10", "c4:e9:84:5a:10:18"
    assert mac_distance(router, ext) == 8     # MAC-wise they look like the same radio...
    for g in groups:
        assert not (router in g and ext in g)  # ...but RSSI traces say otherwise
