import hashlib
import json

import numpy as np
import pytest

from common.geometry import segment_crossings
from common.session import load_session
from sim.floorplan import home_plan, random_plan
from sim.generate import SimConfig, generate
from sim.radio import RadioConfig, rss_mean, rtt_range


def _digest(p):
    return hashlib.sha256((p / "events.jsonl").read_bytes() + (p / "truth.json").read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def sim_dir(tmp_path_factory):
    return generate(SimConfig(seed=3), tmp_path_factory.mktemp("sim") / "s")


def test_deterministic(tmp_path):
    a = generate(SimConfig(seed=5), tmp_path / "a")
    b = generate(SimConfig(seed=5), tmp_path / "b")
    c = generate(SimConfig(seed=6), tmp_path / "c")
    assert _digest(a) == _digest(b) != _digest(c)


def test_parses_with_real_loader(sim_dir):
    s = load_session(sim_dir)
    assert len(s.scans) > 40 and {"acc", "gyro", "rotvec", "mag"} <= set(s.imu)
    assert len(s.waypoints) >= 10 and not s.rtts
    assert s.imu["rotvec"][1].shape[1] == 4
    # 50 Hz IMU
    assert abs(np.median(np.diff(s.imu["acc"][0])) - 0.02) < 1e-3


def test_walk_never_crosses_walls(sim_dir):
    T = json.loads((sim_dir / "truth.json").read_text())
    walls = np.array([[w["x1"], w["y1"], w["x2"], w["y2"]] for w in T["plan"]["walls"]])
    xy = np.array(T["traj"]["xy"])
    assert not segment_crossings(xy[:-1], xy[1:], walls).any()


def test_route_loops_and_revisits(sim_dir):
    T = json.loads((sim_dir / "truth.json").read_text())
    ids = [v["marker"] for v in T["visits"]]
    assert ids[0] == ids[-1] and len(set(ids)) < len(ids)


def test_wall_attenuation_and_doors():
    plan = home_plan(0)
    W, A = plan.wall_array(), plan.atten_array()
    cfg = RadioConfig()
    ap = (2.0, 4.75)   # router in hall
    # similar distance (~3.0-3.4 m): hall point in LOS vs. through the y=4 brick wall into living room
    los = rss_mean(np.array([[5.9, 4.75]]), ap, -40, 2437, W, A, cfg)[0]
    nlos = rss_mean(np.array([[4.5, 1.75]]), ap, -40, 2437, W, A, cfg)[0]
    assert los - nlos > 6          # brick wall ~10 dB
    door = rss_mean(np.array([[1.95, 1.75 - 0.0]]), ap, -40, 2437, W, A, cfg)[0]   # straight through the door gap
    assert abs(door - (-40 - 23 * np.log10(3.0))) < 0.01
    # 5 GHz is weaker than 2.4
    assert rss_mean(np.array([[5.9, 4.75]]), ap, -40, 5180, W, A, cfg)[0] < los - 5


def test_rtt_nlos_positive_bias():
    plan = home_plan(0)
    W, A = plan.wall_array(), plan.atten_array()
    rng = np.random.default_rng(0)
    cfg = RadioConfig(rtt_outlier_p=0, rtt_fail_p=0)
    los = [r[0] for r in rtt_range(rng, np.tile([5.9, 4.75], (400, 1)), (2.0, 4.75), W, A, cfg)]
    nlos = [r[0] for r in rtt_range(rng, np.tile([4.5, 1.75], (400, 1)), (2.0, 4.75), W, A, cfg)]
    assert abs(np.mean(los) - 3.9) < 0.1
    assert np.mean(nlos) - np.hypot(2.5, 3.0) > 0.4


def test_rtt_session(tmp_path):
    p = generate(SimConfig(seed=2, rtt=True), tmp_path / "r")
    s = load_session(p)
    assert len(s.rtts) > 50 and s.meta["rtt"]["available"]
    assert any(o.rtt_responder for sc in s.scans for o in sc.obs)


def test_random_plan_connected():
    for seed in range(5):
        plan = random_plan(seed)
        names = {r.name for r in plan.rooms}
        seen, frontier = {plan.rooms[0].name}, [plan.rooms[0].name]
        while frontier:
            c = frontier.pop()
            for a, b, *_ in plan.doors:
                for u, v in ((a, b), (b, a)):
                    if u == c and v not in seen:
                        seen.add(v)
                        frontier.append(v)
        assert seen == names, seed


def test_random_plan_session(tmp_path):
    p = generate(SimConfig(seed=4, plan="random"), tmp_path / "x")
    assert len(load_session(p).scans) > 30
