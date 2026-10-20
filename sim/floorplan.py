"""Floor plans: wall segments with material, rooms and doors (for walk planning)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# nominal attenuation (dB at 2.4 GHz) and RTT excess is derived from it
MATERIALS = {"drywall": 4.0, "brick": 10.0, "concrete": 14.0, "glass": 2.5}


@dataclass
class Wall:
    x1: float
    y1: float
    x2: float
    y2: float
    material: str = "drywall"
    atten_db: float = 4.0
    thickness: float = 0.1

    @property
    def length(self) -> float:
        return float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))


@dataclass
class Room:
    name: str
    x0: float
    y0: float
    x1: float
    y1: float

    def center(self):
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def contains(self, p, margin=0.0):
        return self.x0 + margin <= p[0] <= self.x1 - margin and self.y0 + margin <= p[1] <= self.y1 - margin


@dataclass
class FloorPlan:
    walls: list[Wall]
    rooms: list[Room]
    doors: list[tuple[str, str, float, float]] = field(default_factory=list)  # room a, room b, x, y (midpoint)
    bounds: tuple[float, float, float, float] = (0, 0, 10, 8)

    def wall_array(self) -> np.ndarray:
        return np.array([[w.x1, w.y1, w.x2, w.y2] for w in self.walls], float).reshape(-1, 4)

    def atten_array(self) -> np.ndarray:
        return np.array([w.atten_db for w in self.walls], float)

    def to_dict(self):
        return {"walls": [w.__dict__ for w in self.walls], "rooms": [r.__dict__ for r in self.rooms],
                "doors": self.doors, "bounds": self.bounds}


def _wall_with_doors(p1, p2, doors, material, rng, jitter=0.15):
    """Split segment p1->p2 around door gaps (list of (s0,s1) distances along the segment)."""
    p1, p2 = np.array(p1, float), np.array(p2, float)
    L = np.linalg.norm(p2 - p1)
    u = (p2 - p1) / L
    cuts = [0.0]
    for s0, s1 in sorted(doors):
        cuts += [s0, s1]
    cuts.append(L)
    nominal = MATERIALS[material]
    out = []
    for a, b in zip(cuts[0::2], cuts[1::2]):
        if b - a > 1e-6:
            q1, q2 = p1 + u * a, p1 + u * b
            att = nominal * (1 + rng.uniform(-jitter, jitter))
            out.append(Wall(*q1, *q2, material=material, atten_db=float(att),
                            thickness=0.25 if material in ("brick", "concrete") else 0.1))
    return out


def home_plan(seed: int = 0) -> FloorPlan:
    """Hand-made 10 x 8 m flat: living room, kitchen, hall, two bedrooms and a bathroom. Mostly drywall."""
    rng = np.random.default_rng(seed)
    W = []
    # outer brick shell
    W += _wall_with_doors((0, 0), (10, 0), [], "brick", rng)
    W += _wall_with_doors((10, 0), (10, 8), [], "brick", rng)
    W += _wall_with_doors((10, 8), (0, 8), [], "brick", rng)
    W += _wall_with_doors((0, 8), (0, 0), [], "brick", rng)
    # hall south wall y=4 (load bearing brick), doors to living and kitchen
    W += _wall_with_doors((0, 4), (10, 4), [(1.5, 2.4), (7.0, 7.9)], "brick", rng)
    # hall north wall y=5.5, doors to bed1, bath, bed2
    W += _wall_with_doors((0, 5.5), (10, 5.5), [(1.0, 1.9), (4.6, 5.4), (8.0, 8.9)], "drywall", rng)
    # living / kitchen divider x=5 (y 0..4), wide opening
    W += _wall_with_doors((5, 0), (5, 4), [(1.4, 2.5)], "drywall", rng)
    # north rooms dividers
    W += _wall_with_doors((4, 5.5), (4, 8), [], "drywall", rng)
    W += _wall_with_doors((6.5, 5.5), (6.5, 8), [], "drywall", rng)
    rooms = [Room("living", 0, 0, 5, 4), Room("kitchen", 5, 0, 10, 4), Room("hall", 0, 4, 10, 5.5),
             Room("bed1", 0, 5.5, 4, 8), Room("bath", 4, 5.5, 6.5, 8), Room("bed2", 6.5, 5.5, 10, 8)]
    doors = [("living", "hall", 1.95, 4.0), ("kitchen", "hall", 7.45, 4.0), ("living", "kitchen", 5.0, 1.95),
             ("hall", "bed1", 1.45, 5.5), ("hall", "bath", 5.0, 5.5), ("hall", "bed2", 8.45, 5.5)]
    return FloorPlan(W, rooms, doors, (0, 0, 10, 8))


def random_plan(seed: int, width: float = 11.0, height: float = 8.0, min_room: float = 2.6) -> FloorPlan:
    """BSP partition into rooms with doors on every shared wall; outer shell brick, inner mostly drywall."""
    rng = np.random.default_rng(seed)
    rooms: list[tuple[float, float, float, float]] = []

    def split(x0, y0, x1, y1, depth):
        w, h = x1 - x0, y1 - y0
        can_v, can_h = w >= 2 * min_room, h >= 2 * min_room
        if depth == 0 or (not can_v and not can_h) or (w * h < 18 and rng.random() < 0.5):
            rooms.append((x0, y0, x1, y1))
            return
        vertical = can_v and (not can_h or w > h or (w == h and rng.random() < .5))
        if vertical:
            c = round(rng.uniform(x0 + min_room, x1 - min_room) * 4) / 4
            split(x0, y0, c, y1, depth - 1)
            split(c, y0, x1, y1, depth - 1)
        else:
            c = round(rng.uniform(y0 + min_room, y1 - min_room) * 4) / 4
            split(x0, y0, x1, c, depth - 1)
            split(x0, c, x1, y1, depth - 1)

    split(0, 0, width, height, 3)
    R = [Room(f"r{i}", *r) for i, r in enumerate(rooms)]
    W = []
    for p, q in [((0, 0), (width, 0)), ((width, 0), (width, height)), ((width, height), (0, height)), ((0, height), (0, 0))]:
        W += _wall_with_doors(p, q, [], "brick", rng)
    doors = []
    for i, a in enumerate(R):
        for b in R[i + 1:]:
            # shared boundary?
            if abs(a.x1 - b.x0) < 1e-9 or abs(b.x1 - a.x0) < 1e-9:
                x = a.x1 if abs(a.x1 - b.x0) < 1e-9 else a.x0
                lo, hi = max(a.y0, b.y0), min(a.y1, b.y1)
                if hi - lo > 1.4:
                    s = rng.uniform(0.3, hi - lo - 1.2)
                    mat = "brick" if rng.random() < 0.25 else "drywall"
                    W += _wall_with_doors((x, lo), (x, hi), [(s, s + 0.9)], mat, rng)
                    doors.append((a.name, b.name, x, lo + s + 0.45))
            elif abs(a.y1 - b.y0) < 1e-9 or abs(b.y1 - a.y0) < 1e-9:
                y = a.y1 if abs(a.y1 - b.y0) < 1e-9 else a.y0
                lo, hi = max(a.x0, b.x0), min(a.x1, b.x1)
                if hi - lo > 1.4:
                    s = rng.uniform(0.3, hi - lo - 1.2)
                    mat = "brick" if rng.random() < 0.25 else "drywall"
                    W += _wall_with_doors((lo, y), (hi, y), [(s, s + 0.9)], mat, rng)
                    doors.append((a.name, b.name, lo + s + 0.45, y))
    return FloorPlan(W, R, doors, (0, 0, width, height))
