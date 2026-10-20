from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common.session import load_session


def run(args):
    from eval.evaluate import evaluate
    from eval.plots import plot_all
    from eval.truth import load_truth
    from solver.export import result_dict, write_exports
    from solver.pipeline import PipelineConfig, run_pipeline

    ses = load_session(args.session)
    out = Path(args.out or Path("eval/out") / Path(args.session).name)
    out.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig(verbose=args.verbose, n_boot=args.boots)
    if args.step_length:
        cfg.pdr.step_len_m = args.step_length
    gt = None
    truth_path = args.truth or (Path(args.session) / "truth.json" if (Path(args.session) / "truth.json").exists() else None)
    if truth_path:
        gt = load_truth(truth_path)
    if args.labelled:
        cfg.pg.use_labels = True
        cfg.marker_xy = gt.markers if gt is not None else None
    t0 = time.time()
    res = run_pipeline(ses, cfg)
    ev = evaluate(res, gt) if gt is not None else None
    write_exports(res, out, *(((ev.R, ev.t)) if ev else (None, None)))
    paths = plot_all(res, ev, gt, out)
    summary = {"session": str(args.session), "runtime_s": round(time.time() - t0, 1), "timings_s": {k: round(v, 1) for k, v in res.timings.items()},
               "n_scans": len(ses.scans), "n_groups": len(res.groups), "theta_deg": float(res.fit.theta * 180 / 3.14159265) if res.fit else None,
               **result_dict(res), "metrics": ev.metrics if ev else None}
    (out / "result.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"wrote {out}/ (walls.geojson, walls.svg, result.json, plots) in {summary['runtime_s']} s")
    if ev:
        m = ev.metrics
        print("trajectory marker RMSE: PDR-only %.2f m -> pose graph %.2f m" % (m["marker_rmse_pdr_only_m"], m["marker_rmse_m"]))
        print("AP error inside house: %.2f m (multilat) -> %.2f m (joint)" % (m["ap_err_multilat_inside_m"] or -1, m["ap_err_joint_inside_m"] or -1))
        if "walls@0.5" in m:
            print("walls @0.5 m:", m["walls@0.5"])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="solver")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the full pipeline on a session folder")
    r.add_argument("--session", required=True)
    r.add_argument("--out")
    r.add_argument("--truth", help="truth.json (sim) or ground_truth.yaml; default: <session>/truth.json if present")
    r.add_argument("--labelled", action="store_true", help="ablation: feed waypoint markers to the solver")
    r.add_argument("--boots", type=int, default=6, help="bootstrap resamples for wall confidence")
    r.add_argument("--step-length", type=float, default=None,
                   help="your step length in metres (default 0.70). Odometry scale is not observable from loop closures, "
                        "so measure it: walk 10 m, count steps")
    r.add_argument("--verbose", action="store_true")
    r.set_defaults(fn=run)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
