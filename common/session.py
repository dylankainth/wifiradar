"""Read/write sessions in the wallmap data format (docs/data-format.md)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1


@dataclass
class ScanObs:
    """One BSSID observation in one scan."""
    bssid: str
    rssi: float
    freq_mhz: int = 0
    ssid: str = ""
    rtt_responder: bool | None = None


@dataclass
class Scan:
    t_s: float
    obs: list[ScanObs]


@dataclass
class RttObs:
    t_s: float
    bssid: str
    dist_m: float | None
    std_m: float | None
    rssi: float | None
    attempted: int
    successful: int


@dataclass
class Session:
    path: Path | None
    meta: dict
    t0_ns: int
    scans: list[Scan] = field(default_factory=list)
    rtts: list[RttObs] = field(default_factory=list)
    imu: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # type -> (t_s, values)
    baro: tuple[np.ndarray, np.ndarray] = (np.zeros(0), np.zeros(0))
    gps: list[dict] = field(default_factory=list)
    waypoints: list[dict] = field(default_factory=list)  # {t_s, marker, label}

    @property
    def duration_s(self) -> float:
        ts = [self.scans[-1].t_s] if self.scans else []
        for t, _ in self.imu.values():
            if len(t):
                ts.append(t[-1])
        return max(ts) if ts else 0.0

    def bssids(self) -> list[str]:
        return sorted({o.bssid for s in self.scans for o in s.obs})


IMU_TYPES = ("acc", "gyro", "mag", "rotvec")


def load_session(path: str | Path, dedupe_cached: bool = True) -> Session:
    """Parse a session folder. Tolerates unknown event types and blank/corrupt lines."""
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    events = []
    with open(path / "events.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # truncated last line after a crash
    events = [e for e in events if "t_ns" in e and "type" in e]
    events.sort(key=lambda e: e["t_ns"])
    t0 = int(meta.get("start_elapsed_ns", events[0]["t_ns"] if events else 0))

    def ts(e):
        return (e["t_ns"] - t0) * 1e-9

    ses = Session(path=path, meta=meta, t0_ns=t0)
    buf: dict[str, list] = {k: [] for k in IMU_TYPES}
    baro = []
    seen_beacon: dict[str, int] = {}
    for e in events:
        ty = e["type"]
        if ty == "wifi_scan":
            obs = []
            for r in e.get("results", []):
                b = r["bssid"].lower()
                bts = r.get("ts_ns")
                if dedupe_cached and bts is not None:
                    if seen_beacon.get(b) == bts:
                        continue  # re-delivered cached result
                    seen_beacon[b] = bts
                obs.append(ScanObs(b, float(r["rssi_dbm"]), int(r.get("freq_mhz", 0) or 0),
                                   r.get("ssid", "") or "", r.get("is_rtt_responder")))
            if obs:
                ses.scans.append(Scan(ts(e), obs))
        elif ty == "rtt":
            d = e.get("distance_mm")
            sd = e.get("distance_std_mm")
            ses.rtts.append(RttObs(ts(e), e["bssid"].lower(), None if d is None else d / 1000.0,
                                   None if sd is None else sd / 1000.0, e.get("rssi"),
                                   int(e.get("attempted", 0)), int(e.get("successful", 0))))
        elif ty in buf:
            buf[ty].append((ts(e), e["v"]))
        elif ty == "baro":
            baro.append((ts(e), e["hpa"]))
        elif ty == "gps":
            ses.gps.append({**e, "t_s": ts(e)})
        elif ty == "waypoint":
            ses.waypoints.append({"t_s": ts(e), "marker": e["marker"], "label": e.get("label", "")})
    for k, v in buf.items():
        if v:
            ses.imu[k] = (np.array([a for a, _ in v]), np.array([b for _, b in v], dtype=float))
    if baro:
        ses.baro = (np.array([a for a, _ in baro]), np.array([b for _, b in baro]))
    return ses


class SessionWriter:
    """Writes events in the real format. Used by the simulator and by tests."""

    def __init__(self, path: str | Path, meta: dict):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.meta = {"format_version": FORMAT_VERSION, **meta}
        self.events: list[dict] = []

    def add(self, t_ns: int, type_: str, **kw):
        self.events.append({"t_ns": int(t_ns), "type": type_, **kw})

    def close(self):
        (self.path / "meta.json").write_text(json.dumps(self.meta, indent=2))
        self.events.sort(key=lambda e: e["t_ns"])
        with open(self.path / "events.jsonl", "w") as f:
            for e in self.events:
                f.write(json.dumps(e, separators=(",", ":")) + "\n")
