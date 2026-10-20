"""Turn a simulator run into a 'fake real' session: what the Android logger would plausibly have written, plus a
tape-measure style ground_truth.yaml in the plan frame. Exercises the real loaders and the M4 pipeline end to end.

Quirks added on purpose: large uptime clock offset, ScanResult timestamps + age, cached scan re-delivery (same beacon
timestamps again), scan_request / info events, out-of-order lines (sensor batching latency), a truncated last line
(app killed), waypoint labels, no truth.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

UPTIME_OFFSET_NS = 98_765_000_000_000   # phone had been up ~27 h


def make_fake_real(sim_dir: str | Path, out_dir: str | Path, seed: int = 0, truncate_last: bool = True) -> Path:
    sim_dir, out_dir = Path(sim_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    meta = json.loads((sim_dir / "meta.json").read_text())
    truth = json.loads((sim_dir / "truth.json").read_text())
    events = [json.loads(l) for l in (sim_dir / "events.jsonl").read_text().splitlines() if l.strip()]
    start = meta["start_elapsed_ns"]
    out, arrival = [], []

    def add(e, delay_ns=0):
        out.append(e)
        arrival.append(e["t_ns"] + delay_ns)

    scan_idx = 0
    for e in events:
        e = dict(e)
        e["t_ns"] += UPTIME_OFFSET_NS
        ty = e["type"]
        if ty == "wifi_scan":
            age_ms = int(rng.integers(200, 2500))
            ts_ns = e["t_ns"] - age_ms * 1_000_000
            rows = []
            for r in e["results"]:
                r = dict(r)
                r["age_ms"] = age_ms
                r["ts_ns"] = ts_ns - int(rng.integers(0, 50_000_000))
                rows.append(r)
            e["results"] = rows
            e["source"] = "app" if rng.random() < 0.8 else "system"
            add({"t_ns": e["t_ns"] - 300_000_000, "type": "scan_request", "accepted": bool(rng.random() > 0.08)})
            add(e)
            if rng.random() < 0.3:   # cached re-delivery: same beacon timestamps, later delivery
                dup = {**e, "t_ns": e["t_ns"] + 700_000_000, "scan_id": e.get("scan_id", 0) + 100000, "source": "system"}
                add(dup)
            scan_idx += 1
        elif ty in ("acc", "gyro", "mag", "rotvec", "baro"):
            add(e, delay_ns=int(rng.integers(0, 300_000_000)))   # SensorManager batching latency
        elif ty == "waypoint":
            e["label"] = f"marker {e['marker']}"
            add(e)
        else:
            add(e)
    add({"t_ns": start + UPTIME_OFFSET_NS + 1_000_000, "type": "info", "msg": "RTT: feature not present"})
    order = np.argsort(arrival, kind="stable")
    lines = [json.dumps(out[i], separators=(",", ":")) for i in order]
    if truncate_last:
        lines[-1] = lines[-1][: len(lines[-1]) // 2]
    (out_dir / "events.jsonl").write_text("\n".join(lines) + "\n")
    meta = {**meta, "start_elapsed_ns": start + UPTIME_OFFSET_NS, "simulated": False,
            "device": {"manufacturer": "samsung", "model": "SM-S931B", "android_release": "15", "sdk_int": 35},
            "session_name": out_dir.name, "app_version": "0.1.0", "notes": "fake-real session converted from a simulator run"}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # tape-measured truth, plan frame, my APs only
    walls = [{"from": [round(w["x1"], 2), round(w["y1"], 2)], "to": [round(w["x2"], 2), round(w["y2"], 2)],
              "material": w["material"]} for w in truth["plan"]["walls"]]
    aps = [{"name": a["name"], "xy": [round(a["xy"][0], 2), round(a["xy"][1], 2)], "bssids": [b["mac"] for b in a["bssids"]]}
           for a in truth["aps"] if a["mine"]]
    markers = {int(k): [round(v[0], 2), round(v[1], 2)] for k, v in truth["markers"].items()}
    doc = {"bounds": [float(x) for x in truth["plan"]["bounds"]], "walls": walls, "aps": aps, "markers": markers}
    (out_dir / "ground_truth.yaml").write_text(
        "# tape-measured ground truth (fake: derived from the simulator)\n" + yaml.safe_dump(doc, sort_keys=False, default_flow_style=None, width=110))
    return out_dir


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="convert a simulator session into a fake-real session + ground_truth.yaml")
    ap.add_argument("--sim", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    print(make_fake_real(a.sim, a.out))


if __name__ == "__main__":
    main()
