# ground_truth.yaml (tape-measured truth for real sessions)

Real sessions have no `truth.json`. You measure the flat with a tape and write one small YAML file. It is used
**only for evaluation / overlays** (never fed to the solver, except in the labelled ablation `--labelled`).

```yaml
# All lengths in metres, in ONE coordinate system of your choosing: pick a corner as (0,0), x along one wall,
# y along the perpendicular wall (the solver aligns its own arbitrary frame to this one via the markers).
bounds: [0, 0, 10, 8]            # optional: x0, y0, x1, y1 of the flat (used to tell "inside" APs from outside ones)

walls:                           # straight wall segments; split a wall at every doorway (leave the door gap open)
  - {from: [0, 0],   to: [10, 0], material: brick}      # material is a free-text note
  - {from: [0, 4],   to: [1.5, 4], material: brick}
  - {from: [2.4, 4], to: [7.0, 4], material: brick}

aps:                             # your OWN access points (position of the box) and ALL their BSSIDs
  - name: router
    xy: [2.0, 4.75]
    bssids: ["c4:e9:84:5a:10:10", "c4:e9:84:5a:10:11", "c6:e9:84:5a:10:12"]   # 2.4 GHz, 5 GHz, guest ...

markers:                         # floor markers; id = the number you press in the app
  1: [1.2, 0.8]
  2: [4.0, 2.2]
```

Rules and tips

* **Markers are what align the solver's frame to yours** (rigid fit, no scale): measure at least 4 well-spread
  markers carefully (+-2 cm is plenty). Revisits of a marker simply reuse its id.
* **BSSIDs**: Settings -> Wi-Fi -> your network -> Advanced, or the router's admin page, or the logger's *My APs*
  list. If you export with *Anonymise*, your own BSSIDs stay in clear text, so this file still matches.
  Mesh nodes: list each node as its own `aps:` entry with its own BSSIDs.
* Walls: only interior and exterior *structural* walls matter. Skip furniture. The evaluation also reports an
  "observable-only" score because RSS alone cannot see some walls (e.g. the outer shell seen only from APs outside,
  see README).
* A complete example for the simulated house: `eval/ground_truth.example.yaml`.

Use it:

```bash
uv run python -m solver.cli run --session data/home_walk_1 --truth ground_truth.yaml
uv run python -m eval.overlay --result eval/out/home_walk_1/result.json --truth ground_truth.yaml
```
