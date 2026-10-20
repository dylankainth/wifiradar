"""Access point specs for the simulator, including a mesh extender and LA-bit BSSID variants."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BssidSpec:
    mac: str
    freq_mhz: int
    ssid: str


@dataclass
class APSpec:
    name: str
    xy: tuple[float, float]
    bssids: list[BssidSpec]
    mine: bool = True
    rtt: bool = False
    p0_2g: float = -40.0   # RSS at 1 m, 2.4 GHz (dBm)

    def to_dict(self):
        return {"name": self.name, "xy": list(self.xy), "mine": self.mine, "rtt": self.rtt, "p0_2g": self.p0_2g,
                "bssids": [b.__dict__ for b in self.bssids]}


def mac_add(mac: str, delta: int, set_la: bool = False) -> str:
    b = bytearray(int(x, 16) for x in mac.split(":"))
    v = (int.from_bytes(b[3:], "big") + delta) % (1 << 24)
    b[3:] = v.to_bytes(3, "big")
    if set_la:
        b[0] |= 0x02
    return ":".join(f"{x:02x}" for x in b)


def home_aps(rng: np.random.Generator, n_neighbours: int = 3, esp32_anchors: int = 0) -> list[APSpec]:
    """Router (3 BSSIDs incl. a locally-administered guest variant), a mesh extender whose MACs sit right next
    to the router's (same OUI, last byte +8: a trap for MAC-only grouping), a TV stick, a printer, neighbours."""
    base = "c4:e9:84:5a:10:10"
    aps = [
        APSpec("router", (2.0, 4.75), [
            BssidSpec(base, 2437, "HomeNet"),
            BssidSpec(mac_add(base, 1), 5180, "HomeNet"),
            BssidSpec(mac_add(base, 2, set_la=True), 2437, "HomeNet-Guest")], p0_2g=-38),
        APSpec("extender", (8.2, 6.9), [
            BssidSpec(mac_add(base, 8), 2437, "HomeNet"),
            BssidSpec(mac_add(base, 9), 5240, "HomeNet"),
            BssidSpec(mac_add(base, 10, set_la=True), 2437, "HomeNet-Guest")], p0_2g=-42),
        APSpec("tv_stick", (3.6, 0.7), [BssidSpec("fa:8f:ca:31:77:0c", 2462, "DIRECT-tv")], p0_2g=-48),
        APSpec("printer", (1.2, 7.2), [BssidSpec("3c:2a:f4:90:0b:d2", 2412, "DIRECT-print")], p0_2g=-50),
    ]
    for i in range(esp32_anchors):
        xy = [(4.3, 2.6), (9.0, 1.2), (5.0, 4.75)][i % 3]
        aps.append(APSpec(f"esp32_{i}", xy, [BssidSpec(f"24:6f:28:a0:00:{0x10 + i:02x}", 2437, f"anchor{i}")],
                          rtt=True, p0_2g=-45))
    spots = [(-5.0, 2.5), (15.0, 6.5), (4.0, -6.0), (14.0, -3.0)]
    for i in range(n_neighbours):
        oui = ":".join(f"{x:02x}" for x in rng.integers(0, 256, 3))
        oui = f"{int(oui[:2], 16) & 0xFC:02x}" + oui[2:]
        tail = ":".join(f"{x:02x}" for x in rng.integers(0, 256, 3))
        m = f"{oui}:{tail}"
        aps.append(APSpec(f"neighbour{i}", spots[i % len(spots)],
                          [BssidSpec(m, 2412 + 25 * int(rng.integers(0, 3)), f"Neighbour{i}"),
                           BssidSpec(mac_add(m, 1), 5200, f"Neighbour{i}")], mine=False, p0_2g=-38))
    return aps
