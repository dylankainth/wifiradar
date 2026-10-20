# wallmap logger (Android)

Kotlin + Jetpack Compose logger that writes the session format in `docs/data-format.md`:
WiFi scan results (incl. system-triggered scans), optional WiFi RTT ranges, accelerometer / gyro / magnetometer /
rotation vector, barometer, GPS (if any) and waypoint markers. Min SDK 28, compile/target SDK 36.

**Status:** it compiles (`assembleDebug`) and its pure-Kotlin logic has 18 JVM unit tests. It has **not been run on a
phone** by the author: no device was available. Treat the first walk as the acceptance test (see
`docs/survey-guide.md`), and read the "untested on hardware" list below.

## Build

Requirements: JDK 17+ (21 is fine), Android SDK with platform 36 and build-tools (Android Studio installs them),
and `ANDROID_HOME` / `local.properties` (`sdk.dir=...`) pointing at the SDK.

```bash
cd android
./gradlew assembleDebug            # -> app/build/outputs/apk/debug/app-debug.apk
./gradlew testDebugUnitTest        # 18 unit tests (Events, Bssid, Anonymizer, SessionStore, ScanStats, ...)
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Or open the `android/` folder in Android Studio and press Run.

If Maven Central answers `429 Too Many Requests` (it does from some shared networks), point Gradle at a mirror:

```bash
export WALLMAP_MAVEN_MIRROR=https://maven-central.storage-download.googleapis.com/maven2/
```

## Layout (`app/src/main/java/io/wallmap/logger/`)

| file | role | unit-tested |
|---|---|---|
| `Events.kt` | one JSON line per event type (kotlinx.serialization), channel-width codes | yes |
| `Bssid.kt`, `Anonymizer.kt` | MAC helpers, salted BSSID/SSID pseudonyms, **byte-compatible with `common/anonymize.py`** (shared test vectors) | yes |
| `SessionStore.kt`, `SessionNaming.kt`, `SessionMeta.kt` | session folders, unique names, zip export with optional anonymisation, `meta.json` | yes |
| `ScanStats.kt`, `JsonlWriter.kt` | achieved scan/sensor rates, background JSONL writer | yes |
| `SensorLogger.kt`, `WifiScanner.kt`, `RttRanger.kt`, `GpsLogger.kt` | Android framework glue | no (needs a device) |
| `SessionRecorder.kt`, `LoggerService.kt` | one recording session; foreground service + wake/WiFi locks | no |
| `MainActivity.kt` | Compose UI: permissions, start/stop, markers, live counters, My APs, export | no |

## Design notes

* **Scan throttling.** Android allows a foreground app 4 `startScan()` calls per 2 minutes unless Developer options
  -> Networking -> *Wi-Fi scan throttling* is off. The scanner requests a new scan as soon as results arrive, backs off
  (3 s, then 10 s) when `startScan()` is refused, logs `scan_request{accepted}` events and records the achieved median
  period and a `throttling_disabled_hint` in `meta.json`. It also listens to `SCAN_RESULTS_AVAILABLE_ACTION`, so scans
  triggered by the system are captured too (`source: "system"`). Results re-delivered from the OS cache keep their old
  `ScanResult.timestamp`, logged as `ts_ns`; the Python loader drops such duplicates.
* **RTT.** Checked with `FEATURE_WIFI_RTT` and `WifiRttManager.isAvailable()`; ranges only to scan results with
  `is80211mcResponder` (strongest first, up to `RangingRequest.getMaxPeers()`), about once per second. If unavailable it logs an
  `info` event and `meta.rtt.available=false` and carries on. RTT only runs in the foreground, so the activity sets
  `FLAG_KEEP_SCREEN_ON`.
* **Permissions.** Fine location + Nearby Wi-Fi devices (API 33+) are required to start; notifications (API 33+) are
  requested but optional. Activity-recognition permission is **not** needed (step detection is done offline from the
  raw accelerometer), so it is not requested.
* **Privacy.** Only AP metadata is logged. Export with *Anonymise* replaces BSSIDs/SSIDs of APs you did not mark in
  *My APs* by salted HMAC pseudonyms (salt random per export, not stored).

## Untested on hardware (please check on the first walk)

* Whether `SENSOR_DELAY_GAME`-class sampling (20 ms) is honoured on the S25 (see `meta.json` -> `sensors.*.achieved_hz`).
* Achieved scan period with throttling off (`meta.json` -> `scan.median_period_s`).
* `FOREGROUND_SERVICE_TYPE_LOCATION` start behaviour on your One UI build, and that logging continues when the
  screen locks (the app asks to keep the screen on anyway).
* One UI may show an extra permission prompt for "Nearby devices".
* `RttRanger` code path (typical home routers are not RTT responders, so it may never fire without e.g. an ESP32 FTM responder).
