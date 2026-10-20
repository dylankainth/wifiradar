"""Run a full simulated survey and write it in the real session format (+ truth.json)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from common.session import SessionWriter
from sim.aps import APSpec, home_aps
from sim.floorplan import FloorPlan, home_plan, random_plan
from sim.radio import RadioConfig, ShadowField, rss_mean, rtt_range
from sim.walk import WalkConfig, plan_route, synth_imu, timeline

START_NS = 123_456_000_000


@dataclass
class SimConfig:
    seed: int = 1
    plan: str = "home"                 # "home" or "random"
    scan_period_s: float = 4.0         # achieved WiFi scan period (Android throttling-off ~ 4-6 s)
    scan_jitter_s: float = 1.0
    rtt: bool = False                  # simulate RTT-capable phone with ESP32 FTM anchors
    esp32_anchors: int = 2
    rtt_period_s: float = 2.0
    n_neighbours: int = 3
    heading_drift_scale: float = 1.0   # multiplies compass disturbance and gyro bias
    radio: RadioConfig = field(default_factory=RadioConfig)
    walk: WalkConfig = field(default_factory=WalkConfig)


def _interp_pos(tl, t):
    return np.array([np.interp(t, tl["t"], tl["pos"][:, 0]), np.interp(t, tl["t"], tl["pos"][:, 1])])


def generate(cfg: SimConfig, out: str | Path) -> Path:
    out = Path(out)
    ss = np.random.SeedSequence(cfg.seed)
    r_plan, r_ap, r_walk, r_imu, r_wifi, r_rtt, r_shadow = [np.random.default_rng(s) for s in ss.spawn(7)]
    plan: FloorPlan = home_plan(cfg.seed) if cfg.plan == "home" else random_plan(cfg.seed)
    walls, atten = plan.wall_array(), plan.atten_array()
    aps: list[APSpec] = home_aps(r_ap, cfg.n_neighbours, cfg.esp32_anchors if cfg.rtt else 0)
    if cfg.plan != "home":  # scatter APs into the random plan's rooms
        bx0, by0, bx1, by1 = plan.bounds
        for ap in aps:
            if ap.mine:
                r = plan.rooms[int(r_ap.integers(len(plan.rooms)))]
                ap.xy = (float(r_ap.uniform(r.x0 + .5, r.x1 - .5)), float(r_ap.uniform(r.y0 + .5, r.y1 - .5)))
    wc = cfg.walk
    wc.gyro_bias_dps *= cfg.heading_drift_scale
    wc.mag_yaw_sigma_deg *= cfg.heading_drift_scale
    markers, seq, verts, stops = plan_route(plan, r_walk, wc)
    tl = timeline(verts, stops, wc, r_walk)
    world_rot = float(r_imu.uniform(-np.pi, np.pi))
    imu = synth_imu(tl, wc, r_imu, world_rot)
    T = tl["T"]

    meta = {"session_name": out.name, "simulated": True, "app_version": "sim-0.1",
            "device": {"manufacturer": "sim", "model": "SIMULATED", "android_release": "0", "sdk_int": 0},
            "start_wall_time": "2026-01-01T00:00:00Z", "start_elapsed_ns": START_NS, "duration_s": T,
            "my_bssids": [b.mac for a in aps if a.mine for b in a.bssids], "anonymised": False,
            "notes": f"simulated, seed={cfg.seed}"}
    w = SessionWriter(out, meta)
    ns = lambda t: START_NS + int(round(t * 1e9))

    # IMU
    jit = lambda n: r_imu.normal(0, 5e-4, n)
    for key, hz_div in (("acc", 1), ("gyro", 1), ("rotvec", 1), ("mag", 2)):
        idx = np.arange(0, len(imu["t"]), hz_div)
        for i, tj in zip(idx, imu["t"][idx] + jit(len(idx))):
            w.add(ns(max(tj, 0)), key, v=np.round(imu[key][i], 5).tolist())
    for tj in np.arange(0, T, 0.2):
        w.add(ns(tj), "baro", hpa=round(float(1013.25 - 0.12 * 1.2 + 0.02 * np.sin(tj / 60) + r_imu.normal(0, .02)), 3))

    # WiFi scans
    shadow = {b.mac: ShadowField(r_shadow, cfg.radio.shadow_sigma_db, cfg.radio.shadow_corr_m)
              for a in aps for b in a.bssids}
    t = 2.0
    scan_id = 0
    scan_times = []
    while t < T - 1:
        p = _interp_pos(tl, t)
        res = []
        for a in aps:
            for b in a.bssids:
                m = float(rss_mean(p, a.xy, a.p0_2g, b.freq_mhz, walls, atten, cfg.radio)[0]) + float(shadow[b.mac](p)[0])
                rssi = m + r_wifi.normal(0, cfg.radio.meas_sigma_db)
                pdet = 1 / (1 + np.exp(-(rssi - cfg.radio.detect_mid_dbm) / 2.0))
                if rssi > -95 and r_wifi.random() < pdet:
                    res.append({"bssid": b.mac, "ssid": b.ssid, "rssi_dbm": int(round(rssi)), "freq_mhz": b.freq_mhz,
                                "channel_width_mhz": 80 if b.freq_mhz > 4000 else 20,
                                "is_rtt_responder": bool(a.rtt)})
        r_wifi.shuffle(res)
        w.add(ns(t), "wifi_scan", scan_id=scan_id, source="app", results=res)
        scan_times.append(t)
        scan_id += 1
        t += max(1.0, cfg.scan_period_s + r_wifi.uniform(-cfg.scan_jitter_s, cfg.scan_jitter_s))

    # RTT
    n_rtt = n_rtt_ok = 0
    if cfg.rtt:
        t = 3.0
        while t < T - 1:
            p = _interp_pos(tl, t)
            for a in aps:
                if not a.rtt:
                    continue
                d = np.linalg.norm(p - np.asarray(a.xy))
                if d > 25:
                    continue
                dist, sd = rtt_range(r_rtt, p, a.xy, walls, atten, cfg.radio)[0]
                n_rtt += 1
                ok = dist is not None
                n_rtt_ok += ok
                w.add(ns(t + 0.3), "rtt", bssid=a.bssids[0].mac,
                      distance_mm=None if not ok else int(dist * 1000),
                      distance_std_mm=None if not ok else int(sd * 1000),
                      rssi=int(round(float(rss_mean(p, a.xy, a.p0_2g, a.bssids[0].freq_mhz, walls, atten, cfg.radio)[0]))),
                      attempted=8, successful=int(r_rtt.integers(5, 9)) if ok else 0,
                      status="SUCCESS" if ok else "FAIL")
            t += cfg.rtt_period_s

    # markers (waypoint button pressed 0.6 s into each pause)
    visits = []
    marker_xy = {}
    pi = 0
    for (t0, t1, vi) in tl["pauses"]:
        pos = _interp_pos(tl, t0 + 0.1)
        # identify which marker id this pause belongs to
        mid = int(np.argmin(np.linalg.norm(markers - pos, axis=1)))
        marker_xy[mid] = markers[mid].tolist()
        w.add(ns(t0 + 0.6), "waypoint", marker=mid, label="")
        visits.append({"marker": mid, "t_s": t0 + 0.6})
    w.meta["scan"] = {"requests": scan_id, "accepted": scan_id, "batches": scan_id,
                      "median_period_s": float(np.median(np.diff(scan_times))) if len(scan_times) > 1 else None,
                      "throttling_disabled_hint": cfg.scan_period_s < 30}
    w.meta["rtt"] = {"feature_present": cfg.rtt, "available": cfg.rtt, "attempts": n_rtt, "successes": n_rtt_ok}
    w.meta["sensors"] = {k: {"requested_hz": wc.imu_hz, "achieved_hz": wc.imu_hz} for k in ("acc", "gyro", "rotvec")}
    w.close()

    sub = slice(0, len(tl["t"]), 5)
    truth = {
        "config": json.loads(json.dumps(asdict(cfg), default=float)),
        "plan": plan.to_dict(),
        "aps": [a.to_dict() for a in aps],
        "markers": {str(k): v for k, v in marker_xy.items()},
        "visits": visits,
        "world_rot": world_rot,   # ENU frame = plan frame rotated CCW by this angle
        "traj": {"t_s": tl["t"][sub].tolist(), "xy": np.round(tl["pos"][sub], 4).tolist(),
                 "heading": np.round(tl["heading"][sub], 4).tolist()},
        "step_t": imu["step_t"].tolist(),
        "gyro_bias": imu["gyro_bias"].tolist(),
        "scan_t": scan_times,
    }
    (out / "truth.json").write_text(json.dumps(truth))
    return out
