"""Stage 6: GeoJSON and SVG export (local metric coordinates, solver frame unless a transform is given)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from solver.pipeline import Result


def to_geojson(res: Result, R=None, t=None) -> dict:
    """Coordinates are local metres (x east-ish, y north-ish), NOT lon/lat; stated in `properties.crs`."""
    tf = (lambda p: np.asarray(p) @ R.T + t) if R is not None else (lambda p: np.asarray(p))
    feats = []
    fit = res.fit
    if fit is not None:
        for k, w in enumerate(fit.walls):
            a, b = tf(w[:2]), tf(w[2:])
            feats.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": [a.tolist(), b.tolist()]},
                          "properties": {"kind": "wall", "id": k, "confidence": float(res.support[k]),
                                         "attenuation_db": float(fit.atten[k]), "evidence_dloss": float(fit.dL[k]),
                                         "length_m": float(np.linalg.norm(w[2:] - w[:2]))}})
    for j, ap in enumerate(res.ap_list):
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": tf(ap["xy"]).tolist()},
                      "properties": {"kind": "ap", "id": j, "bssids": ap["bssids"]}})
    traj = tf(res.pg.P)
    feats.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": traj.tolist()},
                  "properties": {"kind": "trajectory"}})
    return {"type": "FeatureCollection", "features": feats,
            "properties": {"crs": "local metres (solver frame)" if R is None else "local metres (aligned to ground truth)",
                           "theta_deg": float(np.rad2deg(fit.theta)) if fit else None}}


def to_svg(res: Result, R=None, t=None, scale=60, pad=2.0) -> str:
    tf = (lambda p: np.asarray(p) @ R.T + t) if R is not None else (lambda p: np.asarray(p))
    fit = res.fit
    pts = [tf(res.pg.P)]
    walls = [(tf(w[:2]), tf(w[2:])) for w in fit.walls] if fit else []
    aps = [(tf(a["xy"]), a["bssids"]) for a in res.ap_list]
    for a, b in walls:
        pts.append(np.array([a, b]))
    allp = np.concatenate(pts)
    lo, hi = allp.min(0) - pad, allp.max(0) + pad
    W, H = (hi - lo) * scale
    X = lambda p: (p[0] - lo[0]) * scale
    Y = lambda p: (hi[1] - p[1]) * scale
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" viewBox="0 0 {W:.0f} {H:.0f}">',
           f'<rect width="100%" height="100%" fill="white"/>']
    tr = tf(res.pg.P)
    out.append('<polyline fill="none" stroke="#9ecae1" stroke-width="1" points="' +
               " ".join(f"{X(p):.1f},{Y(p):.1f}" for p in tr) + '"/>')
    for k, (a, b) in enumerate(walls):
        c = float(res.support[k])
        out.append(f'<line x1="{X(a):.1f}" y1="{Y(a):.1f}" x2="{X(b):.1f}" y2="{Y(b):.1f}" stroke="#b2182b" '
                   f'stroke-width="{3 + 3 * c:.1f}" stroke-opacity="{0.25 + 0.75 * c:.2f}" stroke-linecap="butt">'
                   f'<title>wall {k}: confidence {c:.2f}, {fit.atten[k]:.1f} dB</title></line>')
    for xy, bs in aps:
        out.append(f'<polygon points="{X(xy):.1f},{Y(xy) - 7:.1f} {X(xy) - 6:.1f},{Y(xy) + 5:.1f} {X(xy) + 6:.1f},{Y(xy) + 5:.1f}" '
                   f'fill="#2166ac"><title>{", ".join(bs)}</title></polygon>')
    out.append("</svg>")
    return "\n".join(out)


def write_exports(res: Result, out_dir: str | Path, R=None, t=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "walls.geojson").write_text(json.dumps(to_geojson(res, R, t), indent=1))
    (out_dir / "walls.svg").write_text(to_svg(res, R, t))


def result_dict(res: Result) -> dict:
    """Everything needed to evaluate/overlay later without re-running (solver frame)."""
    fit = res.fit
    markers = []
    for w in res.session.waypoints:
        markers.append({"marker": int(w["marker"]), "t_s": float(w["t_s"]),
                        "xy": res.pg.at(w["t_s"] + 1.5)[0].tolist()})
    return {
        "frame": "solver (arbitrary rigid frame; align with markers)",
        "theta_deg": float(np.rad2deg(fit.theta)) if fit else None,
        "walls": [{"xy": w.tolist(), "confidence": float(s), "attenuation_db": float(a), "evidence_dloss": float(d)}
                  for w, s, a, d in zip(fit.walls, res.support, fit.atten, fit.dL)] if fit else [],
        "aps": res.ap_list,
        "aps_multilat": [{"bssids": g, "xy": res.aps0.xy[j].tolist()} for j, g in enumerate(res.ds.groups)],
        "markers_est": markers,
        "nodes_xy": res.pg.P.tolist(),
        "trajectory_pdr_only": res.pg.nodes.pdr_xy.tolist(),
        "path_loss_exponent": fit.n if fit else None,
        "groups": res.groups,
    }
