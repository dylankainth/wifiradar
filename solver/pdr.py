"""Stage 1: pedestrian dead reckoning from phone IMU.

Steps come from low-passed |acc| peaks (orientation independent). Heading is the horizontal direction of the
device +y axis (phone held roughly flat, top edge pointing where you walk), taken from the rotation vector
(compass, absolute but disturbed indoors), the integrated gyro (smooth but drifts) or a complementary fusion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.spatial.transform import Rotation as Rot, Slerp

from common.geometry import wrap_angle
from common.session import Session


@dataclass
class PDRConfig:
    step_len_m: float = 0.70
    lp_hz: float = 3.0
    peak_height: float = 0.8        # m/s^2 above gravity after low-pass
    min_step_interval_s: float = 0.33
    heading_mode: str = "fused"     # "rotvec" | "gyro" | "fused"
    fuse_tau_s: float = 20.0        # complementary filter: below 1/tau rely on the compass, above on the gyro


@dataclass
class PDRResult:
    step_t: np.ndarray
    step_xy: np.ndarray          # position after each step, start at (0,0)
    step_h: np.ndarray
    h_t: np.ndarray              # heading time series
    h: np.ndarray
    _node_t: np.ndarray = None
    _node_xy: np.ndarray = None

    def __post_init__(self):
        # a step moves us during the ~0.6 s before it fires; no motion between walking bouts
        t, p = [0.0], [np.zeros(2)]
        prev_t, prev_p = 0.0, np.zeros(2)
        for tk, pk in zip(self.step_t, self.step_xy):
            dt = min(tk - prev_t, 0.6)
            t0 = max(tk - dt, t[-1])
            if t0 > t[-1]:
                t.append(t0); p.append(prev_p)
            t.append(tk); p.append(pk)
            prev_t, prev_p = tk, pk
        self._node_t, self._node_xy = np.array(t), np.array(p)

    def pos(self, t):
        t = np.asarray(t, float)
        return np.stack([np.interp(t, self._node_t, self._node_xy[:, 0]),
                         np.interp(t, self._node_t, self._node_xy[:, 1])], -1)

    def heading(self, t):
        return np.interp(t, self.h_t, np.unwrap(self.h))

    def steps_between(self, t0, t1):
        return int(((self.step_t > t0) & (self.step_t <= t1)).sum())


def detect_steps(t, acc, cfg: PDRConfig) -> np.ndarray:
    fs = 1.0 / np.median(np.diff(t))
    tg = np.arange(t[0], t[-1], 1 / fs)
    mag = np.interp(tg, t, np.linalg.norm(acc, axis=1))
    b, a = butter(2, cfg.lp_hz / (fs / 2))
    x = filtfilt(b, a, mag - np.median(mag))
    pk, _ = find_peaks(x, height=cfg.peak_height, distance=max(1, int(cfg.min_step_interval_s * fs)))
    return tg[pk]


def heading_from_rotvec(q):
    """Horizontal direction (world angle) of the device +y axis."""
    y = Rot.from_quat(q).apply([0.0, 1.0, 0.0])
    return np.arctan2(y[:, 1], y[:, 0])


def heading_series(ses: Session, cfg: PDRConfig):
    tq, q = ses.imu["rotvec"]
    h_rot = np.unwrap(heading_from_rotvec(q))
    if cfg.heading_mode == "rotvec" or "gyro" not in ses.imu:
        return tq, h_rot
    tg, w = ses.imu["gyro"]
    m = (tg >= tq[0]) & (tg <= tq[-1])
    tg, w = tg[m], w[m]
    R = Slerp(tq, Rot.from_quat(q))(tg)
    wz = R.apply(w)[:, 2]           # yaw rate about world z
    h_rot_g = np.interp(tg, tq, h_rot)
    h = np.empty(len(tg))
    h[0] = h_rot_g[0]
    for k in range(1, len(tg)):
        dt = tg[k] - tg[k - 1]
        pred = h[k - 1] + wz[k] * dt
        if cfg.heading_mode == "gyro":
            h[k] = pred
        else:
            h[k] = pred + (dt / cfg.fuse_tau_s) * (h_rot_g[k] - pred)
    return tg, h


def run_pdr(ses: Session, cfg: PDRConfig | None = None) -> PDRResult:
    cfg = cfg or PDRConfig()
    ta, acc = ses.imu["acc"]
    st = detect_steps(ta, acc, cfg)
    th, h = heading_series(ses, cfg)
    hs = np.interp(st, th, h)
    d = cfg.step_len_m * np.stack([np.cos(hs), np.sin(hs)], 1)
    xy = np.cumsum(d, 0)
    return PDRResult(st, xy, hs, th, h)
