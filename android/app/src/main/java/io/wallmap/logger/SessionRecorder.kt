package io.wallmap.logger

import android.content.Context
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import java.io.File
import java.time.Instant
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

data class UiState(
    val recording: Boolean = false,
    val sessionName: String = "",
    val elapsedS: Double = 0.0,
    val scanRequests: Int = 0,
    val scanAccepted: Int = 0,
    val scanBatches: Int = 0,
    val scansPerMin: Double = 0.0,
    val lastScanAps: Int = 0,
    val accHz: Double? = null,
    val rotvecHz: Double? = null,
    val rttFeature: Boolean = false,
    val rttAvailable: Boolean = false,
    val rttAttempts: Int = 0,
    val rttSuccesses: Int = 0,
    val markersPressed: Int = 0,
    val nextMarker: Int = 1,
    val gpsFixes: Int = 0,
    val message: String = "",
    val latestBssids: List<Pair<String, String>> = emptyList(),  // bssid to ssid, strongest first
)

/** Owns one recording session: writer, sensors, WiFi, RTT, GPS. Lives in the foreground service. */
class SessionRecorder(private val context: Context, private val store: SessionStore) {
    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state.asStateFlow()

    private var writer: JsonlWriter? = null
    private var sensors: SensorLogger? = null
    private var wifi: WifiScanner? = null
    private var rtt: RttRanger? = null
    private var gps: GpsLogger? = null
    private var dir: File? = null
    private var startElapsedNs = 0L
    private var startWall = ""
    private var name = ""
    private var myBssids: Set<String> = emptySet()
    val markers = MarkerCounter()
    private val handler = Handler(Looper.getMainLooper())
    private val ticker = object : Runnable {
        override fun run() { publish(); if (_state.value.recording) handler.postDelayed(this, 500) }
    }

    val isRecording: Boolean get() = _state.value.recording

    fun start(requestedName: String, mine: Set<String>) {
        if (isRecording) return
        val d = store.create(requestedName.ifBlank { SessionNaming.defaultName(Instant.now().toString()) })
        dir = d; name = d.name; myBssids = mine.map { it.lowercase() }.toSet()
        startElapsedNs = SystemClock.elapsedRealtimeNanos()
        startWall = Instant.now().toString()
        val w = JsonlWriter(File(d, "events.jsonl"))
        writer = w
        val sink = EventSink { w.write(it) }
        writeMeta(final = false)
        markers.set(1)
        sensors = SensorLogger(context, sink).also { it.start() }
        wifi = WifiScanner(context, sink, ScanStats()).also { it.start() }
        rtt = RttRanger(context, sink) { wifi?.latest ?: emptyList() }.also { it.start() }
        gps = GpsLogger(context, sink).also { it.start() }
        _state.value = _state.value.copy(recording = true, sessionName = name, message = "")
        handler.post(ticker)
    }

    /** "I am at marker N"; returns the id used. */
    fun marker(label: String): Int {
        val id = markers.press()
        writer?.write(Events.waypoint(SystemClock.elapsedRealtimeNanos(), id, label))
        publish()
        return id
    }

    fun setNextMarker(id: Int) { markers.set(id); publish() }
    fun stepMarker(d: Int) { markers.step(d); publish() }

    fun stop() {
        if (!isRecording) return
        handler.removeCallbacks(ticker)
        sensors?.stop(); wifi?.stop(); rtt?.stop(); gps?.stop()
        writeMeta(final = true)
        writer?.close()
        _state.value = _state.value.copy(recording = false, message = "saved ${dir?.absolutePath}")
        writer = null; sensors = null; wifi = null; rtt = null; gps = null
    }

    private fun writeMeta(final: Boolean) {
        val d = dir ?: return
        val now = SystemClock.elapsedRealtimeNanos()
        val s = sensors
        val sensorMeta = s?.rates?.mapValues { (k, v) ->
            SessionMeta.Sensor(if (k == "baro") 5.0 else if (k == "mag") 25.0 else s.requestedHz, if (final) v.achievedHz() else null)
        } ?: emptyMap()
        val meta = SessionMeta.build(
            sessionName = name, appVersion = BuildConfig.APP_VERSION,
            device = SessionMeta.Device(Build.MANUFACTURER, Build.MODEL, Build.VERSION.RELEASE, Build.VERSION.SDK_INT),
            startWallTimeIso = startWall, startElapsedNs = startElapsedNs,
            durationS = (now - startElapsedNs) / 1e9,
            rttFeaturePresent = rtt?.featurePresent ?: false, rttAvailable = rtt?.available ?: false,
            rttAttempts = rtt?.attempts ?: 0, rttSuccesses = rtt?.successes ?: 0,
            scan = wifi?.stats ?: ScanStats(), sensors = sensorMeta, myBssids = myBssids,
            notes = if (final) "" else "in progress",
        )
        File(d, "meta.json").writeText(meta.toString())
    }

    private fun publish() {
        val now = SystemClock.elapsedRealtimeNanos()
        val st = wifi?.stats
        val latest = wifi?.latest.orEmpty().sortedByDescending { it.level }
            .map { (it.BSSID ?: "") to (@Suppress("DEPRECATION") it.SSID ?: "") }
        _state.value = _state.value.copy(
            elapsedS = if (isRecording) (now - startElapsedNs) / 1e9 else _state.value.elapsedS,
            scanRequests = st?.requests ?: 0, scanAccepted = st?.accepted ?: 0, scanBatches = st?.batches ?: 0,
            scansPerMin = st?.perMinute(now) ?: 0.0, lastScanAps = wifi?.lastBatchSize ?: 0,
            accHz = sensors?.rates?.get("acc")?.achievedHz(), rotvecHz = sensors?.rates?.get("rotvec")?.achievedHz(),
            rttFeature = rtt?.featurePresent ?: _state.value.rttFeature, rttAvailable = rtt?.available ?: false,
            rttAttempts = rtt?.attempts ?: 0, rttSuccesses = rtt?.successes ?: 0,
            markersPressed = markers.pressed, nextMarker = markers.next, gpsFixes = gps?.fixes ?: 0,
            latestBssids = if (latest.isNotEmpty()) latest else _state.value.latestBssids,
            message = rtt?.lastMessage?.takeIf { it.isNotEmpty() } ?: _state.value.message,
        )
    }

    /** Single instance shared between service and UI. */
    companion object {
        @Volatile var instance: SessionRecorder? = null
        val idleState = MutableStateFlow(UiState())
    }
}
