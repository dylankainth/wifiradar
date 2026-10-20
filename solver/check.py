"""Quick sanity report for a freshly recorded session: run this on the laptop right after the first walk."""
from __future__ import annotations

import numpy as np

from common.session import load_session
from solver.pdr import run_pdr


def check_session(path) -> tuple[list[tuple[str, str]], dict]:
    """Returns ([(level, message)], stats). level in OK / WARN / FAIL."""
    ses = load_session(path)
    out, st = [], {}
    add = lambda lvl, msg: out.append((lvl, msg))
    dur = ses.duration_s
    st["duration_s"] = dur
    add("OK" if dur >= 180 else "WARN", f"duration {dur:.0f} s" + ("" if dur >= 180 else " (short: aim for 5-10 min)"))

    t = np.array([s.t_s for s in ses.scans])
    if len(t) < 2:
        add("FAIL", f"{len(t)} WiFi scans: location off, permissions denied, or scanning blocked")
    else:
        per = float(np.median(np.diff(t)))
        st["scan_period_s"] = per
        lvl = "OK" if per <= 8 else "WARN" if per <= 20 else "FAIL"
        msg = f"{len(t)} scans, median period {per:.1f} s"
        if lvl != "OK":
            msg += " - too slow; is 'Wi-Fi scan throttling' still on? (wall quality drops sharply above ~8 s)"
        add(lvl, msg)
        aps = {o.bssid for s in ses.scans for o in s.obs}
        strong = {o.bssid for s in ses.scans for o in s.obs if o.rssi > -75}
        add("OK" if len(strong) >= 4 else "WARN", f"{len(aps)} BSSIDs seen, {len(strong)} ever stronger than -75 dBm (need ~4+ nearby)")
    for k in ("acc", "gyro", "rotvec"):
        if k not in ses.imu or len(ses.imu[k][0]) < 100:
            add("FAIL", f"no '{k}' stream (needed for dead reckoning)")
        else:
            hz = 1 / np.median(np.diff(ses.imu[k][0]))
            add("OK" if hz > 25 else "WARN", f"{k} {hz:.0f} Hz")
    if "acc" in ses.imu and "rotvec" in ses.imu:
        pdr = run_pdr(ses)
        n = len(pdr.step_t)
        path_len = float(np.linalg.norm(np.diff(pdr.step_xy, axis=0), axis=1).sum()) if n > 1 else 0.0
        st["steps"], st["path_m"] = n, path_len
        add("OK" if n >= 100 else "WARN" if n >= 10 else "FAIL",
            f"{n} steps detected (~{path_len:.0f} m walked)" + ("" if n >= 100 else " - phone held flat in front? walk longer?"))
    m = ses.waypoints
    ids = [w["marker"] for w in m]
    add("OK" if len(ids) >= 10 and len(set(ids)) < len(ids) else "WARN",
        f"{len(ids)} marker presses, {len(set(ids))} distinct" + ("" if len(set(ids)) < len(ids) else " - no revisits; loop closure needs revisits"))
    r = ses.meta.get("rtt", {})
    add("OK", f"RTT feature={r.get('feature_present')} available={r.get('available')} ranges={len(ses.rtts)} (optional)")
    return out, st


def main(path) -> int:
    res, _ = check_session(path)
    for lvl, msg in res:
        print(f"[{lvl:4s}] {msg}")
    return 1 if any(l == "FAIL" for l, _ in res) else 0
