"""Ground truth containers. Sim truth.json and hand-written ground_truth.yaml both load into GroundTruth."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common.geometry import rot2


@dataclass
class TruthAP:
    name: str
    xy: np.ndarray
    bssids: list[str]
    mine: bool = True


@dataclass
class GroundTruth:
    walls: np.ndarray                           # (K,4) in the truth frame
    aps: list[TruthAP]
    markers: dict[int, np.ndarray]              # marker id -> xy
    traj: tuple[np.ndarray, np.ndarray] | None = None   # (t_s, xy) full-resolution truth (sim only)
    bounds: tuple[float, float, float, float] | None = None
    to_enu: tuple[np.ndarray, np.ndarray] | None = None  # sim: R,t mapping truth frame -> solver ENU frame (for tests only)


def from_sim_truth(path: str | Path) -> GroundTruth:
    p = Path(path)
    p = p / "truth.json" if p.is_dir() else p
    T = json.loads(p.read_text())
    walls = np.array([[w["x1"], w["y1"], w["x2"], w["y2"]] for w in T["plan"]["walls"]], float)
    aps = [TruthAP(a["name"], np.array(a["xy"], float), [b["mac"] for b in a["bssids"]], bool(a["mine"])) for a in T["aps"]]
    markers = {int(k): np.array(v, float) for k, v in T["markers"].items()}
    tr = (np.array(T["traj"]["t_s"]), np.array(T["traj"]["xy"]))
    return GroundTruth(walls, aps, markers, tr, tuple(T["plan"]["bounds"]),
                       (rot2(T["world_rot"]), np.zeros(2)))


def from_yaml(path: str | Path) -> GroundTruth:
    import yaml
    d = yaml.safe_load(Path(path).read_text())
    walls = np.array([[*w["from"], *w["to"]] for w in d.get("walls", [])], float).reshape(-1, 4)
    aps = [TruthAP(a.get("name", "ap"), np.array(a["xy"], float), [b.lower() for b in a.get("bssids", [])])
           for a in d.get("aps", [])]
    markers = {int(k): np.array(v, float) for k, v in d.get("markers", {}).items()}
    b = d.get("bounds")
    return GroundTruth(walls, aps, markers, None, tuple(b) if b else None)


def load_truth(path: str | Path) -> GroundTruth:
    p = Path(path)
    if p.suffix in (".yaml", ".yml"):
        return from_yaml(p)
    return from_sim_truth(p)
