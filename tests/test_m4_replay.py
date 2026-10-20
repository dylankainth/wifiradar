"""M4 acceptance: a simulator run, converted into a 'fake real' session, runs through the real loaders, the full
`solver run` CLI against a tape-measure style ground_truth.yaml, and the overlay script."""
import json

import numpy as np
import pytest

from common.session import load_session
from eval.truth import load_truth
from sim.fake_real import UPTIME_OFFSET_NS, make_fake_real
from sim.generate import SimConfig, generate
from solver.cli import main as solver_main


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    d = tmp_path_factory.mktemp("m4")
    sim = generate(SimConfig(seed=2), d / "sim")
    return sim, make_fake_real(sim, d / "fake_real")


def test_fake_real_has_no_truth_and_real_quirks(fake):
    _, fr = fake
    assert not (fr / "truth.json").exists() and (fr / "ground_truth.yaml").exists()
    raw = (fr / "events.jsonl").read_text().splitlines()
    assert not raw[-1].endswith("}")                          # truncated last line (app killed)
    ts = [json.loads(l)["t_ns"] for l in raw[:-1]]
    assert ts != sorted(ts)                                    # arrival order, not time order
    assert min(ts) > UPTIME_OFFSET_NS


def test_loader_dedupes_cached_redeliveries(fake):
    sim, fr = fake
    ses_sim, ses_fr = load_session(sim), load_session(fr)
    raw_scans = sum(1 for l in (fr / "events.jsonl").read_text().splitlines()[:-1] if '"type":"wifi_scan"' in l)
    assert raw_scans > len(ses_fr.scans)                       # duplicates existed ...
    assert abs(len(ses_fr.scans) - len(ses_sim.scans)) <= 1   # ... and were removed
    assert ses_fr.waypoints[0]["label"].startswith("marker")
    assert ses_fr.imu["acc"][0][0] >= 0                        # clock rebased with meta.start_elapsed_ns


def test_ground_truth_yaml_loads(fake):
    _, fr = fake
    gt = load_truth(fr / "ground_truth.yaml")
    assert gt.walls.shape[1] == 4 and len(gt.walls) > 10
    assert {a.name for a in gt.aps} >= {"router", "extender"} and len(gt.markers) >= 8


def test_solver_cli_and_overlay_on_fake_real(fake, tmp_path):
    _, fr = fake
    out = tmp_path / "run"
    solver_main(["run", "--session", str(fr), "--truth", str(fr / "ground_truth.yaml"), "--out", str(out), "--boots", "0"])
    for f in ("result.json", "walls.geojson", "walls.svg", "trajectory.png", "walls.png", "aps.png", "theta_scan.png"):
        assert (out / f).exists(), f
    res = json.loads((out / "result.json").read_text())
    m = res["metrics"]
    assert m["marker_rmse_m"] < 1.5 and m["marker_rmse_m"] < m["marker_rmse_pdr_only_m"]
    assert m["ap_err_joint_inside_m"] < 2.5
    gj = json.loads((out / "walls.geojson").read_text())
    assert {f["properties"]["kind"] for f in gj["features"]} >= {"wall", "ap", "trajectory"}
    # overlay from saved result only
    from eval.overlay import main as overlay_main
    overlay_main(["--result", str(out / "result.json"), "--truth", str(fr / "ground_truth.yaml"), "--out", str(tmp_path / "o.png")])
    assert (tmp_path / "o.png").exists()
