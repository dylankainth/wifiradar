from __future__ import annotations

import argparse
from pathlib import Path

from sim.generate import SimConfig, generate
from sim.plot import plot_truth


def main(argv=None):
    ap = argparse.ArgumentParser(description="Simulate a WiFi survey walk through a floor plan")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan", choices=["home", "random"], default="home")
    ap.add_argument("--rtt", action="store_true", help="add ESP32 FTM anchors and RTT events")
    ap.add_argument("--scan-period", type=float, default=4.0)
    ap.add_argument("--rss-noise", type=float, default=None, help="per-scan RSS sigma (dB)")
    ap.add_argument("--shadow", type=float, default=None, help="spatially-correlated shadowing sigma (dB)")
    ap.add_argument("--drift", type=float, default=1.0, help="heading drift scale")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)
    cfg = SimConfig(seed=a.seed, plan=a.plan, rtt=a.rtt, scan_period_s=a.scan_period, heading_drift_scale=a.drift)
    if a.rss_noise is not None:
        cfg.radio.meas_sigma_db = a.rss_noise
    if a.shadow is not None:
        cfg.radio.shadow_sigma_db = a.shadow
    out = generate(cfg, a.out)
    if not a.no_plot:
        png = Path("eval/out") / f"{Path(a.out).name}_truth.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        plot_truth(out, png)
        print("plot:", png)
    print("session:", out)


if __name__ == "__main__":
    main()
