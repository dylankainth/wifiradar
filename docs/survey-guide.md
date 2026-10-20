# How to survey (Android logger)

Goal of a survey: collect enough WiFi scans, IMU data and revisits that the solver can (1) reconstruct where you
walked, (2) locate the access points, (3) see where walls attenuate the signal. A good session is **5-10 minutes,
15-30 stops, with several loops and revisits**.

## 0. One-time phone setup (Samsung Galaxy S25 / One UI, Android 15)

1. **Developer options**: Settings -> About phone -> Software information -> tap **Build number** 7 times.
   Enter your PIN. Developer options now appears at the bottom of Settings.
2. **Disable Wi-Fi scan throttling**: Settings -> Developer options -> **Networking** -> toggle
   **Wi-Fi scan throttling** *off*.
   Without this Android allows a foreground app only **4 scans per 2 minutes**, i.e. one scan every 30 s - far too
   few. With it off you get roughly one scan every 3-6 s (the logger records the real rate in `meta.json`).
3. Keep **Wi-Fi on** and *connected or not, it does not matter*. Turn **Wi-Fi "Intelligent Wi-Fi"/"Auto switch"**
   off if you see the radio hopping to mobile data during the walk.
4. **Location** must be ON (Android hides scan results from apps otherwise). Grant the app
   *Precise location*, *Nearby Wi-Fi devices* (Android 13+), *Physical activity* and notifications when asked.
5. Battery: set the app to **Unrestricted** (Settings -> Apps -> wallmap logger -> Battery) so the foreground
   service is not killed.
6. Optional: if you have your own router/extender, open the app's **My APs** screen *before the first walk* and tick
   your own BSSIDs. They are kept in clear text on export; everyone else's are salted-hashed.

> **WiFi RTT (FTM ranging)**: the logger checks `FEATURE_WIFI_RTT` and `WifiRttManager.isAvailable()` and only
> ranges to scan results with `is80211mcResponder`. Typical home routers are *not* responders (Google's
> WifiRttScan app listing 0 RTT-capable APs is normal). The logger works fine without RTT and records
> `rtt.available=false`. RTT needs the **screen on and the app in the foreground**; the logger keeps the screen on.

## 1. Prepare the building (10 minutes, once)

You need **ground truth** to evaluate (and, later, to calibrate):

* Stick 8-15 numbered **markers** (sticky notes with a number) on the floor, at least 1.5 m apart, spread over all
  rooms, including some **in doorways** and at corridor junctions.
* **Tape-measure** the floor plan: wall endpoints and each marker's (x, y), all from one corner of the flat as origin
  (x along one wall, y along the perpendicular one). Write these into `ground_truth.yaml`
  (format and example: `docs/ground-truth.md`, `eval/ground_truth.example.yaml`).
* Note where your own access points sit (x, y) and their BSSIDs (Settings -> Wi-Fi -> (gear) -> Advanced, or the
  router's admin page).

## 2. How to walk

* Hold the phone **flat in front of you, screen up, top edge pointing the way you walk**, like reading a map.
  Do not swing it, do not put it in a pocket: the dead-reckoning heading assumes this.
* Walk at a **normal, steady pace**. Stop **at each marker for 4-5 seconds** (stand still, do not rotate), and
  press **"I am at marker N"** when you arrive. At least one scan lands in every pause.
* **Visit every room, go through every doorway**, and visit the **most distinct places more than once**:
  the route should contain **loops** (e.g. living room -> hall -> kitchen -> living room) and
  **revisits of the same marker from different directions**. Return to marker 1 at the end.
  Loop closure is what removes IMU drift - a straight out-and-back walk gives the solver very little.
* Do not run. Avoid sharp U-turns while moving; turn at the marker instead.
* Do not walk with other people carrying phones close to yours (it is harmless to the data but the people's bodies
  shadow the signal).
* If you use the "marker id" stepper because you revisit a marker, set the id to that marker's number first, then
  press the button; the app otherwise increments by one.

## 2b. Calibrate your step length (1 minute, worth it)

Dead reckoning assumes 0.70 m per step. Loop closures **cannot** correct the scale (they bias it low, see
`docs/design.md`), so a 10 % step-length error becomes a 10 % error in every distance. Measure it once:
walk 10 m at your normal pace counting steps (or just look at the first leg between two markers whose distance you
know), then pass `--step-length <10 / steps>` to `solver run` (e.g. 14 steps -> `--step-length 0.71`).

## 3. Using the app

1. Open the app, grant permissions, type a **session name** (e.g. `home_walk_1`), tap **Start**.
2. Keep the screen on and the app visible (the app does this for you). Check the counters: *scans/min* should be
   10-20 with throttling off, *IMU Hz* about 50, *RTT success %* only appears if the phone supports it.
3. Press **Marker** at each stop (optional label field for "kitchen door").
4. **Stop** at the end. The session is written to
   `Android/data/<app id>/files/sessions/<name>/` (`meta.json`, `events.jsonl`).
5. **Export**: tap *Export*, choose *Anonymise other people's APs* (recommended before sharing anything) and share
   the zip (to yourself via Drive/USB/`adb pull`).

## 4. Run the solver on the laptop

First, **sanity-check the recording** (takes a second, tells you if the walk is worth keeping):

```bash
uv run solver check --session data/home_walk_1
```

It prints OK/WARN/FAIL for scan rate, IMU rates, steps detected, APs heard, marker revisits. Fix any FAIL and re-walk before solving.

```bash
unzip home_walk_1.zip -d data/home_walk_1
uv run python -m solver.cli run --session data/home_walk_1 --truth ground_truth.yaml --out eval/out/home_walk_1
```

Results (`walls.geojson`, `walls.svg`, plots, `result.json`) land in `eval/out/home_walk_1/`.
If you export with *Anonymise*, the solver still groups BSSIDs correctly (see `docs/data-format.md#privacy`), but
`ground_truth.yaml` must list your own APs' BSSIDs (those are never hashed).

## 5. Common problems

| symptom | cause / fix |
|---|---|
| Only a scan every ~30 s | Scan throttling still on (step 0.2), or Wi-Fi scanning disabled. |
| `meta.json` says `throttling_disabled_hint: false` | The achieved period is >20 s; redo step 0.2. |
| 0 scan results | Location off, or permission "Nearby devices" denied. |
| Trajectory curls into a spiral | You swung the phone, or held it vertically. Hold it flat in front of you. |
| Wall estimate empty | Too few revisits or too few APs heard (need ~4+ APs with decent signal). Walk more loops. |
| `rtt.available=false` | Normal for most phones with home routers. RTT needs 802.11mc/ax FTM-capable APs. |
