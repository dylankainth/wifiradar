import json

import jax.numpy as jnp
import numpy as np
import pytest

from common.geometry import rot2, segment_crossings
from common.session import load_session
from eval.metrics import wall_metrics
from sim.generate import SimConfig, generate
from solver.aploc import localise_aps
from solver.dataset import build_dataset
from solver.grouping import group_bssids
from solver.pdr import run_pdr
from solver.pipeline import wall_support
from solver.posegraph import PGConfig, fingerprint_distances, run_posegraph
from solver.wallfit import soft_cross


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    p = generate(SimConfig(seed=3), tmp_path_factory.mktemp("s") / "s")
    ses = load_session(p)
    pdr = run_pdr(ses)
    groups = group_bssids(ses)
    return p, ses, pdr, groups, json.loads((p / "truth.json").read_text())


def test_soft_cross_matches_hard_cross_at_small_tau():
    rng = np.random.default_rng(0)
    P, Q = rng.uniform(0, 10, (400, 2)), rng.uniform(0, 10, (400, 2))
    walls = rng.uniform(0, 10, (6, 4))
    hard = segment_crossings(P, Q, walls)
    soft = np.asarray(soft_cross(jnp.asarray(P), jnp.asarray(Q), jnp.asarray(walls[:, :2]), jnp.asarray(walls[:, 2:]), 0.005))
    assert (np.abs(soft - hard) > 0.5).mean() < 0.01


def test_soft_cross_is_differentiable_wrt_wall_endpoints():
    import jax
    p, q = jnp.array([[0.0, 0.0]]), jnp.array([[4.0, 0.0]])
    f = lambda w: soft_cross(p, q, w[None, :2], w[None, 2:], 0.3).sum()
    g = jax.grad(f)(jnp.array([2.0, -1.0, 2.0, 1.0]))
    assert np.isfinite(np.asarray(g)).all() and np.abs(np.asarray(g)).max() > 1e-3


def test_posegraph_improves_on_pdr(prepared):
    from eval.truth import load_truth
    from eval.metrics import align_from_markers, apply, ate_rmse
    p, ses, pdr, groups, T = prepared
    gt = load_truth(p)
    tt, txy = gt.traj

    def ate(res):
        tru = np.stack([np.interp(res.nodes.t, tt, txy[:, 0]), np.interp(res.nodes.t, tt, txy[:, 1])], 1)
        ev = {}
        for w in ses.waypoints:
            ev.setdefault(w["marker"], []).append(res.at(w["t_s"] + 1.5)[0])
        R, t = align_from_markers(ev, gt.markers)
        return ate_rmse(apply(R, t, res.P), tru)

    odo = run_posegraph(ses, pdr, groups, PGConfig(use_loops=False))
    loop = run_posegraph(ses, pdr, groups, PGConfig())
    assert len(loop.loops_kept) > 10
    assert ate(loop) < 0.8 * ate(odo)


def test_loop_closures_are_mostly_correct(prepared):
    p, ses, pdr, groups, T = prepared
    res = run_posegraph(ses, pdr, groups, PGConfig())
    tt = np.array(T["traj"]["t_s"]); txy = np.array(T["traj"]["xy"])
    tr = lambda t: np.array([np.interp(t, tt, txy[:, 0]), np.interp(t, tt, txy[:, 1])])
    good = [np.linalg.norm(tr(res.nodes.t[a]) - tr(res.nodes.t[b])) < 3.0 for a, b, _ in res.loops]
    assert np.mean(good) > 0.8


def test_fingerprint_distance_symmetry():
    F = np.array([[-50, -60, np.nan], [-52, -61, -80.0], [-80, -40, -70.0]])
    D = fingerprint_distances(F, -85)
    assert np.allclose(D, D.T, equal_nan=False) or np.isinf(D).any()
    assert D[0, 1] < D[0, 2]


def test_aploc_finds_interior_aps(prepared):
    from eval.truth import load_truth
    p, ses, pdr, groups, T = prepared
    gt = load_truth(p)
    pg = run_posegraph(ses, pdr, groups, PGConfig())
    ds = build_dataset(ses, pg, groups)
    m = localise_aps(ds)
    # solver frame -> truth frame via the marker alignment is tested elsewhere; here compare inter-AP distances
    est = {}
    for j, g in enumerate(ds.groups):
        for a in gt.aps:
            if a.mine and set(a.bssids) & set(g):
                est[a.name] = (m.xy[j], a.xy)
    names = [n for n in ("router", "extender", "tv_stick", "printer") if n in est]
    assert len(names) >= 3
    errs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            de = np.linalg.norm(est[names[i]][0] - est[names[j]][0])
            dt = np.linalg.norm(est[names[i]][1] - est[names[j]][1])
            errs.append(abs(de - dt))
    assert np.mean(errs) < 2.5


def test_wall_metrics_basics():
    gt = np.array([[0, 0, 10, 0], [0, 0, 0, 5.0]])
    assert wall_metrics(gt, gt)["f1"] > 0.99
    shifted = gt + np.array([0, 0.3, 0, 0.3])
    assert wall_metrics(shifted, gt, tol=0.5)["f1"] > 0.9
    assert wall_metrics(shifted, gt, tol=0.1)["f1"] < 0.5
    half = wall_metrics(gt[:1], gt)
    assert half["precision"] > 0.99 and 0.5 < half["recall"] < 0.8
    obs = np.array([True, False])
    assert wall_metrics(gt[:1], gt, observable=obs)["recall"] > 0.99


def test_wall_support():
    a = np.array([[0, 0, 4, 0.0], [0, 3, 4, 3.0]])
    other = [np.array([[0, 0.1, 4, 0.1]]), np.array([[0, 0.0, 2, 0.0]])]
    s = wall_support(a, other, 0.5)
    assert s[0] > 0.7 and s[1] < 0.01
