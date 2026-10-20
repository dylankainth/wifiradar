"""Waypoint-based walks (with loops/revisits) and synthetic IMU for them."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.transform import Rotation as Rot

from common.geometry import wrap_angle
from sim.floorplan import FloorPlan

G = 9.81


@dataclass
class WalkConfig:
    n_markers: int = 12          # physical marker locations
    n_visits: int = 30           # stops in the route (markers are revisited, route returns to the first)
    speed_mps: float = 1.2
    pause_s: float = 4.0
    step_len_m: float = 0.72     # true mean step length (solver assumes 0.70)
    step_len_sd: float = 0.04
    imu_hz: float = 50.0
    # sensor errors
    gyro_bias_dps: float = 0.35  # per-axis constant bias sigma
    gyro_noise: float = 0.01     # rad/s
    mag_yaw_sigma_deg: float = 8.0   # indoor compass disturbance (OU), applies to rotvec only
    mag_yaw_tau_s: float = 20.0
    acc_noise: float = 0.2


def place_markers(plan: FloorPlan, rng, n: int, margin: float = 0.6):
    pts, rooms = [], []
    per = max(1, n // len(plan.rooms))
    order = list(range(len(plan.rooms)))
    k = 0
    while len(pts) < n:
        r = plan.rooms[order[k % len(order)]]
        k += 1
        if r.x1 - r.x0 < 2 * margin or r.y1 - r.y0 < 2 * margin:
            continue
        p = (rng.uniform(r.x0 + margin, r.x1 - margin), rng.uniform(r.y0 + margin, r.y1 - margin))
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) > 1.5 for q in pts) or k > 200:
            pts.append(p)
            rooms.append(r.name)
    return np.array(pts), rooms


def _nav_graph(plan: FloorPlan, markers, marker_rooms):
    names = [r.name for r in plan.rooms]
    nodes = [(m[0], m[1], marker_rooms[i], None) for i, m in enumerate(markers)]
    nm = len(nodes)
    for a, b, x, y in plan.doors:
        nodes.append((x, y, a, b))
    n = len(nodes)
    A = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ri = {nodes[i][2], nodes[i][3]} - {None}
            rj = {nodes[j][2], nodes[j][3]} - {None}
            if ri & rj:
                A[i, j] = np.hypot(nodes[i][0] - nodes[j][0], nodes[i][1] - nodes[j][1])
    A[np.isinf(A)] = 0  # csgraph treats 0 as no edge for dense input
    return nodes, nm, A


def plan_route(plan: FloorPlan, rng, cfg: WalkConfig):
    """Returns marker coords, marker visit sequence, and the dense polyline vertices (with stop flags)."""
    markers, mrooms = place_markers(plan, rng, cfg.n_markers)
    nodes, nm, A = _nav_graph(plan, markers, mrooms)
    dist, pred = dijkstra(A, return_predecessors=True)
    seq = [int(rng.integers(0, nm))]
    while len(seq) < cfg.n_visits - 1:
        nxt = int(rng.integers(0, nm))
        if nxt != seq[-1] and np.isfinite(dist[seq[-1], nxt]):
            seq.append(nxt)
    seq.append(seq[0])  # loop closure back to the start
    verts, stops = [tuple(markers[seq[0]])], [0]
    for a, b in zip(seq[:-1], seq[1:]):
        path, cur = [], b
        while cur != a:
            path.append(cur)
            cur = pred[a, cur]
        for node in reversed(path):
            verts.append((nodes[node][0], nodes[node][1]))
            stops.append(1 if node == b else 0)
    return markers, seq, np.array(verts), np.array(stops, bool)


def timeline(verts, stops, cfg: WalkConfig, rng):
    """Piecewise-linear motion + pauses at stop vertices. Returns dict of arrays sampled at imu_hz."""
    dt = 1.0 / cfg.imu_hz
    t_end = 0.0
    segs = []   # (t0, t1, p0, p1)
    pauses = []  # (t0, t1, vertex index)
    t = 0.0
    pauses.append((t, t + cfg.pause_s, 0)); t += cfg.pause_s
    for i in range(1, len(verts)):
        L = float(np.linalg.norm(verts[i] - verts[i - 1]))
        if L < 1e-6:
            continue
        v = max(0.7, rng.normal(cfg.speed_mps, 0.08))
        segs.append((t, t + L / v, verts[i - 1], verts[i]))
        t += L / v
        if stops[i]:
            pa = max(2.0, rng.normal(cfg.pause_s, 0.8))
            pauses.append((t, t + pa, i)); t += pa
    T = t
    ts = np.arange(0, T, dt)
    pos = np.zeros((len(ts), 2))
    head_t = np.zeros(len(ts))
    moving = np.zeros(len(ts), bool)
    cur_head = None
    for t0, t1, p0, p1 in segs:
        m = (ts >= t0) & (ts < t1)
        f = (ts[m] - t0) / (t1 - t0)
        pos[m] = p0 + f[:, None] * (p1 - p0)
        d = p1 - p0
        head_t[m] = np.arctan2(d[1], d[0])
        moving[m] = True
    for t0, t1, i in pauses:
        m = (ts >= t0) & (ts < t1)
        pos[m] = verts[i]
    # fill gaps (pauses at start/end): hold previous heading
    first = np.argmax(moving)
    head_t[:first] = head_t[first]
    for k in range(first, len(ts)):
        if not moving[k]:
            head_t[k] = head_t[k - 1]
    # smooth heading (first-order, tau 0.35 s) on the unwrapped signal
    un = np.unwrap(head_t)
    a = dt / (0.35 + dt)
    h = np.empty_like(un)
    h[0] = un[0]
    for k in range(1, len(un)):
        h[k] = h[k - 1] + a * (un[k] - h[k - 1])
    return {"t": ts, "pos": pos, "heading": h, "moving": moving, "T": T, "pauses": pauses}


def _ou(rng, n, dt, sigma, tau):
    a = np.exp(-dt / tau)
    x = np.zeros(n)
    s = sigma * np.sqrt(1 - a * a)
    x[0] = rng.normal(0, sigma)
    w = rng.normal(0, s, n)
    for k in range(1, n):
        x[k] = a * x[k - 1] + w[k]
    return x


def synth_imu(tl, cfg: WalkConfig, rng, world_rot: float):
    """Synthesize acc/gyro/mag/rotvec at imu_hz for the timeline. Frames: see docs/data-format.md.

    Returns arrays plus true step times (for PDR tests)."""
    t, pos, h, moving = tl["t"], tl["pos"], tl["heading"], tl["moving"]
    n, dt = len(t), 1.0 / cfg.imu_hz
    # distance walked -> step phase
    ds = np.r_[0, np.linalg.norm(np.diff(pos, axis=0), axis=1)]
    s = np.cumsum(ds)
    n_steps = int(s[-1] / (cfg.step_len_m * 0.8)) + 5
    L = np.clip(rng.normal(cfg.step_len_m, cfg.step_len_sd, n_steps), 0.4, None)
    bounds = np.r_[0, np.cumsum(L)]
    k = np.clip(np.searchsorted(bounds, s, side="right") - 1, 0, n_steps - 1)
    phase = k + (s - bounds[k]) / L[k]
    amp = rng.normal(2.6, 0.4, n_steps)[k]
    walking = moving.astype(float)
    a_up = walking * amp * np.sin(2 * np.pi * phase)
    a_fw = walking * 0.35 * amp * np.sin(2 * np.pi * phase + 1.0)
    fwd = np.stack([np.cos(h), np.sin(h)], 1)
    a_w = np.zeros((n, 3))
    a_w[:, :2] = a_fw[:, None] * fwd
    a_w[:, 2] = a_up
    # true orientation, device y = walking direction
    psi_plan = h - np.pi / 2
    pitch = -0.45 + _ou(rng, n, dt, 0.08, 5.0) + 0.04 * np.sin(2 * np.pi * phase) * walking
    roll = _ou(rng, n, dt, 0.06, 5.0)
    psi_enu = psi_plan + world_rot
    R_true = Rot.from_euler("ZXY", np.stack([psi_enu, pitch, roll], 1))
    # step times: where integer phase crossing happens
    step_idx = np.where(np.diff(np.floor(phase)) > 0)[0] + 1
    step_idx = step_idx[moving[step_idx]]
    # sensors
    acc_dev = R_true.inv().apply(a_w + np.array([0, 0, G]))
    acc_dev += rng.normal(0, cfg.acc_noise, acc_dev.shape) + rng.normal(0, 0.03, 3)
    q = R_true.as_quat()
    rel = (R_true[:-1].inv() * R_true[1:]).as_rotvec() / dt
    gyro = np.vstack([rel, rel[-1:]])
    bias = np.deg2rad(rng.normal(0, cfg.gyro_bias_dps, 3))
    gyro_m = gyro + bias + rng.normal(0, cfg.gyro_noise, gyro.shape)
    # rotvec: gravity-aligned tilt exact, yaw corrupted by indoor magnetic disturbance (OU)
    yaw_err = np.deg2rad(_ou(rng, n, dt, cfg.mag_yaw_sigma_deg, cfg.mag_yaw_tau_s))
    R_meas = Rot.from_euler("Z", yaw_err[:, None]) * R_true
    mag_dev = R_true.inv().apply(np.array([0.0, 22.0, -40.0])) + rng.normal(0, 0.8, (n, 3))
    return {"t": t, "acc": acc_dev, "gyro": gyro_m, "mag": mag_dev, "rotvec": R_meas.as_quat(),
            "step_t": t[step_idx], "gyro_bias": bias, "yaw_err": yaw_err}
