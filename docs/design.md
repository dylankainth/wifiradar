# Design notes

## Pipeline

```
 session (meta.json + events.jsonl)
   |
   |-- imu (acc, gyro, rotvec) ---> [1 PDR] steps + heading --------------------------+
   |-- wifi_scan -----------------> [2 BSSID grouping] -> physical APs                |
   |                                        |                                         v
   |                                        +-----> [3 pose graph] <-- odometry (PDR), fingerprint loop closures
   |                                                       |
   |                                       corrected positions of every scan ("nodes")
   |-- rtt (optional) ----+                                |
   |                      v                                v
   |              [4 AP localisation: log-distance (+RTT) multilateration]  -> AP xy, per-BSSID offsets, path-loss n
   |                                                       |
   |                       [5a Manhattan angle search + greedy wall selection (OMP) on a wall-atlas lattice]
   |                                                       |
   |                       [5b joint differentiable fit (JAX): walls + APs + offsets + n, soft crossings,
   |                           Manhattan prior, leave-one-out pruning]  x 3 angle candidates -> best
   |                                                       |
   |                       [5c bootstrap ensemble] -> per-wall confidence
   |                                                       v
   +----------------------------------------------> [6 export] GeoJSON / SVG / result.json / plots
```

`waypoint` events and `truth.json` / `ground_truth.yaml` are never read by stages 1-6 (except in the explicit
labelled ablation `solver run --labelled`, which feeds marker ids to the pose graph).

## 1. PDR (`solver/pdr.py`)

* Steps: low-pass (3 Hz) of `|acc|`, peak detection with a minimum interval of 0.33 s and a height threshold on the
  AC part (0.8 m/s^2). Independent of phone orientation, no steps while standing still.
* Heading = world-frame horizontal direction of the device +y axis (phone flat in front of the user, top edge forward).
  Three modes: `rotvec` (absolute but disturbed by indoor magnetic fields), `gyro` (smooth, drifts), `fused`
  (complementary filter, tau = 20 s; default).
* Step length is a constant 0.70 m (`--step-length` to change it). The pose graph carries a global odometry scale, but with a
  tight prior (sigma 1.5 %): **loop closures cannot estimate scale**. A fingerprint match means "within about 1.5 m", never
  "identical", so closures are one-sidedly biased towards shrinking the trajectory; with many closures (long walks) and a loose prior
  the fitted scale collapsed to 0.4-0.5. Tightening the prior fixed it (long-walk marker error 1.7 m -> 0.7 m); a dead-zone closure
  loss (no penalty inside the matching radius) was tried and was worse.

## 2. BSSID grouping (`solver/grouping.py`)

One box broadcasts several BSSIDs (2.4/5 GHz, guest SSID) with nearly identical MACs, sometimes with the
locally-administered bit set. Candidate pairs: same OUI after clearing the LA bit and a small difference in the low
24 bits. A pair is merged only if the RSSI traces also agree (correlation >= 0.75 *or* both traces nearly flat, and
a small residual spread after removing the constant 2.4-vs-5 GHz offset); merging is complete-linkage. A mesh extender
whose MACs are 8 counts from the router's is therefore kept separate (tested). Thresholds: `GroupingConfig`.

## 3. Pose graph (`solver/posegraph.py`)

Variables: (x, y, theta) per scan + a global odometry scale. Factors: relative-pose odometry from PDR (sigma grows with
distance / sqrt(dt)), weak compass prior on theta, an anchor on node 0, fingerprint loop closures. Loop closures: RMS RSSI
difference over BSSIDs above -85 dBm between scans >= 25 s apart, accepted below 1.7x the median distance between
consecutive *stationary* scans (an adaptive noise floor), at most 2 per node, robust (soft-L1) loss, then re-solved
without the ones that disagree by > 3 sigma. GTSAM was not used: a 2D graph with a few hundred nodes is solved by
`scipy.optimize.least_squares` (sparse Jacobian) in ~0.2 s, and GTSAM is a heavy optional dependency.

## 4. AP localisation (`solver/aploc.py`)

Per-BSSID `c_b - 10 n log10 d` fit with shared AP positions and a global exponent n, soft-L1 loss, 4 starts,
RTT ranges added as residuals when present. Walls are ignored here (they inflate n, typically to 3-4), so this is only the
initial guess for stage 5b.

## 5. Wall fitting

### Model
```
RSS_ib = c_b - 10 n log10 |p_i - x_j| - w_band(b) * sum_k a_k * cross_k(p_i, x_j)
RTT_ij = |p_i - x_j|                  +       alpha * sum_k a_k * cross_k(p_i, x_j)
```
`cross_k` is a product of two sigmoid "straddle" tests (the ray straddles the wall's line, and the wall's endpoints
straddle the ray's line) with temperature tau, annealed 0.7 -> 0.35 -> 0.15 m. It tends to the exact crossing indicator
as tau -> 0 and is differentiable in the wall endpoints and in the AP positions. Loss: pseudo-Huber RSS residuals
(sigma 3 dB) + pseudo-Huber RTT residuals (when present) + Manhattan prior (cross-axis extent, penalising walls off the
axes) + minimum-length penalty + a small sparsity term + a weak pull of AP positions to their multilateration estimate.
Optimised with L-BFGS-B on jit-compiled JAX gradients.

### Initialisation: why not "random walls" or an image?
* A pixel tomography image (L1 / FISTA) gives speckle, not lines (tried; orientation from its structure tensor was wrong).
* The **wall atlas** (`solver/atlas.py`, `solver/omp.py`) restricts candidates to axis-aligned pieces on a fine lattice in a
  frame rotated by theta; ray-vs-piece crossings are exact and the problem is linear in the per-piece attenuations. Greedy
  selection (orthogonal matching pursuit with non-negative refits; candidates = runs of 1..20 pieces, so a candidate is a
  whole segment) stops when a segment no longer pays a BIC-like penalty. LASSO was tried first and was worse (shrinkage,
  near-collinear neighbouring lines).
* **Manhattan angle**: scan theta in 5 degree steps with the greedy objective, take the 3 best local minima, refine
  each with the joint fit (which also learns a small angle offset), keep the lowest penalised loss. In 8 simulated houses the
  chosen angle was on average 10 degrees off but wall F1 was the same as with the true angle, so this is not the bottleneck.

### Pruning and confidence
Leave-one-out loss increase per wall; walls below a threshold (12 loss units, about a BIC penalty) are removed one by one with
short refits. Confidence = **bootstrap support**: the fraction of each wall's length that is reproduced (within 0.5 m) by
fits on block-bootstrap resamples (70 % of the scans, blocks of 6 consecutive scans). The leave-one-out evidence
(`evidence_dloss`) is exported next to it. A Hessian-based uncertainty was not attempted: the loss is piecewise
smooth only after annealing, and the ensemble captures the dominant (structural) uncertainty.

## What RSS cannot see (important for reading the metrics)

For every BSSID the model has a free offset `c_b`. A wall that **every** ray of that AP crosses (or none does) is
indistinguishable from the offset. Consequences, all confirmed in simulation:

* the **outer shell** is invisible when the only APs outside the building are far away and every phone position is on the
  same side: the walker is always inside, so every ray to an outside AP crosses the same outer wall;
* walls that no ray from any AP crosses on only part of the nodes are unobservable;
* very short stubs (< 1 m) have too few rays.

`eval/metrics.py::observable_mask` computes which ground-truth walls are observable (>= 5 positions on each side of the
wall for at least one AP) so recall/F1 are reported both over observable walls ("obs") and over all walls ("all").
In the default simulated house only about 36 % of the wall length is observable.

## Simulator (`sim/`)

Floor plan with doors (walls = segments with material and attenuation), APs including a router with a locally
administered guest BSSID, a mesh extender whose MACs are next to the router's, a TV stick, a printer and neighbours outside
(optionally ESP32 FTM anchors), waypoint walks with loops/revisits, 50 Hz IMU with a step-bounce model, gyro bias, indoor
compass disturbance (Ornstein-Uhlenbeck, 8 degrees rms), scans every ~4 s with log-distance + per-wall loss + spatially
correlated shadowing (random Fourier field, 2.5 dB, 2 m) + per-scan noise (1.5 dB) + soft detection floor, RTT with wall excess
delay, an exponential NLOS tail and 6 % outliers. Sessions are written in the real format, so the real loaders are exercised.

**The simulator and the solver share assumptions** (log-distance + additive per-wall loss, axis-aligned walls, a phone
held flat). Results on simulated data are an upper bound on what real data will give; see the README.
