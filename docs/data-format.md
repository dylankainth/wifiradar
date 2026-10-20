# Session data format (v1)

One **session** is a folder:

```
my_session/
  meta.json        # device + run metadata
  events.jsonl     # one JSON object per line, time-ordered (not guaranteed; readers sort by t_ns)
```

Simulated sessions additionally contain `truth.json` (floor plan, APs, trajectory). Real sessions never do;
real ground truth lives in a hand-written `ground_truth.yaml` (see `docs/ground-truth.md`).
The solver never reads `truth.json` or `waypoint` events (except in the explicit labelled-ablation mode).

## Time base

Every event has `t_ns` = `SystemClock.elapsedRealtimeNanos()` at the moment the app received the event
(for sensor events: `SensorEvent.timestamp`, which is on the same clock). Integer nanoseconds.
Sessions are not rebased: `meta.json.start_elapsed_ns` gives the clock value when logging started.

## Events

Common fields: `t_ns` (int), `type` (string). Units are SI unless the name says otherwise.

### `wifi_scan`
One scan result batch (one `WifiManager.getScanResults()` delivery, including system-triggered scans).

```json
{"t_ns": 1234, "type": "wifi_scan", "scan_id": 17, "source": "app|system",
 "results": [
   {"bssid": "aa:bb:cc:dd:ee:01", "ssid": "HomeNet", "rssi_dbm": -57, "freq_mhz": 5180,
    "channel_width_mhz": 80, "is_rtt_responder": false, "age_ms": 1500}
 ]}
```
* `is_rtt_responder`: `ScanResult.is80211mcResponder()`, `null` if unknown.
* `age_ms`: scan result age at delivery (`elapsedRealtime - ScanResult.timestamp/1000`), optional. Android
  often re-delivers cached results from an older scan, so readers de-duplicate by (bssid, `ts_ns`) when present.
* `ts_ns` (optional): `ScanResult.timestamp * 1000`, the AP-beacon time on the elapsedRealtime clock. Used to
  de-duplicate re-delivered cached results.

### `scan_request`
`{"t_ns":..,"type":"scan_request","accepted":true}` — result of `WifiManager.startScan()` (false = throttled).
Used to compute the achieved scan rate.

### `rtt`
One row per ranging result. Only written when WiFi RTT is available.

```json
{"t_ns": 1, "type": "rtt", "bssid": "aa:..", "distance_mm": 4210, "distance_std_mm": 310,
 "rssi": -60, "attempted": 8, "successful": 7, "status": "SUCCESS"}
```
For failed ranges `distance_mm` is `null` and `successful` is 0.

### `acc`, `gyro`, `mag`, `rotvec`
```json
{"t_ns": 1, "type": "acc",    "v": [x, y, z]}        // m/s^2, includes gravity (TYPE_ACCELEROMETER)
{"t_ns": 1, "type": "gyro",   "v": [x, y, z]}        // rad/s (TYPE_GYROSCOPE), device frame
{"t_ns": 1, "type": "mag",    "v": [x, y, z]}        // microtesla (TYPE_MAGNETIC_FIELD)
{"t_ns": 1, "type": "rotvec", "v": [x, y, z, w]}     // unit quaternion, device -> world (ENU), as
                                                     // SensorManager.getQuaternionFromVector (w last)
```
Device frame is the Android frame (x right, y up the screen, z out of the screen). World is East-North-Up
(magnetic, so yaw is only as good as the compass).

### `baro`
`{"t_ns":1,"type":"baro","hpa":1013.2}`

### `gps`
`{"t_ns":1,"type":"gps","lat":51.5,"lon":-0.1,"acc_m":8.0,"alt_m":null}` — only when a fix is available.

### `waypoint`
`{"t_ns":1,"type":"waypoint","marker":3,"label":"kitchen door"}` — user pressed "I am at marker N".
**Ground truth for evaluation only.** Marker ids map to coordinates in `ground_truth.yaml`.

### `info`
Free-form app notes (`{"type":"info","msg":"..."}`), e.g. "RTT unavailable", "permission denied".

## meta.json

```json
{
  "format_version": 1,
  "session_name": "home_walk_1",
  "app_version": "0.1.0",
  "device": {"manufacturer": "samsung", "model": "SM-S931B", "android_release": "15", "sdk_int": 35},
  "start_wall_time": "2026-10-20T10:00:00Z",
  "start_elapsed_ns": 1000000000,
  "duration_s": 412.2,
  "rtt": {"feature_present": false, "available": false, "attempts": 0, "successes": 0},
  "scan": {"requests": 90, "accepted": 71, "batches": 80, "median_period_s": 4.9,
           "throttling_disabled_hint": true},
  "sensors": {"acc": {"requested_hz": 50, "achieved_hz": 49.8},
              "gyro": {...}, "mag": {...}, "rotvec": {...}, "baro": {...}},
  "my_bssids": ["aa:bb:cc:dd:ee:01"],
  "anonymised": false,
  "notes": ""
}
```
`my_bssids` are the APs the user marked as theirs; they are never hashed on export.
Simulated sessions set `"simulated": true`.

## Privacy

* Only AP metadata (beacon/scan results) is logged. No packet capture, no payloads, no client devices.
* Export with "anonymise" replaces non-`my_bssids` BSSIDs/SSIDs with salted pseudonyms
  (`common/anonymize.py`, mirrored in Kotlin). The pseudonym keeps what the solver needs for AP grouping:
  - the locally-administered bit of byte 0 and the last byte are preserved,
  - the other five bytes map to a keyed hash of `(bytes 0..4 with the LA bit cleared)`,
    so BSSIDs of one radio that differ only in the last byte (or LA bit) stay adjacent,
  - SSIDs become `ssid_<8 hex of HMAC>`; hidden SSIDs stay empty.
  The salt is random per export and not stored (`meta.json.anonymised=true`).
* `data/`, `*.jsonl` and raw logs are git-ignored.
