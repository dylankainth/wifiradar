package io.wallmap.logger

import android.annotation.SuppressLint
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.net.wifi.ScanResult
import android.net.wifi.WifiManager
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import androidx.core.content.ContextCompat

/**
 * Keeps WiFi scans flowing and logs every scan-result delivery, including system-triggered ones.
 *
 * Android limits foreground apps to 4 startScan() calls per 2 minutes unless "Wi-Fi scan throttling" is switched
 * off in Developer options. We request a new scan as soon as the previous results arrive; when startScan()
 * is refused we back off and the achieved rate shows up in ScanStats (and meta.json). Results re-delivered from a
 * cache carry the old ScanResult.timestamp, which the solver uses to de-duplicate.
 */
class WifiScanner(
    context: Context,
    private val sink: EventSink,
    val stats: ScanStats,
) {
    private val ctx = context.applicationContext
    private val wm = ctx.getSystemService(Context.WIFI_SERVICE) as WifiManager
    private val handler = Handler(Looper.getMainLooper())
    private var running = false
    private var scanId = 0
    private var pendingAppScan = false
    @Volatile var latest: List<ScanResult> = emptyList()
        private set
    @Volatile var lastBatchSize = 0
        private set
    var refusedStreak = 0
        private set

    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(c: Context, i: Intent) { onResults() }
    }

    private val requestRunnable = Runnable { requestScan() }

    fun start() {
        running = true
        ContextCompat.registerReceiver(ctx, receiver, IntentFilter(WifiManager.SCAN_RESULTS_AVAILABLE_ACTION),
            ContextCompat.RECEIVER_NOT_EXPORTED)
        onResults()          // capture whatever the system already has
        requestScan()
    }

    fun stop() {
        running = false
        handler.removeCallbacks(requestRunnable)
        try { ctx.unregisterReceiver(receiver) } catch (_: IllegalArgumentException) {}
    }

    @Suppress("DEPRECATION")
    private fun requestScan() {
        if (!running) return
        val ok = try { wm.startScan() } catch (_: SecurityException) { false }
        stats.onRequest(ok)
        sink.emit(Events.scanRequest(SystemClock.elapsedRealtimeNanos(), ok))
        pendingAppScan = ok
        if (ok) {
            refusedStreak = 0
            // results broadcast triggers the next request; this is only a watchdog if it never comes
            handler.postDelayed(requestRunnable, 15_000)
        } else {
            refusedStreak++
            if (refusedStreak == 1) sink.emit(Events.info(SystemClock.elapsedRealtimeNanos(),
                "startScan() refused (scan throttling?). Disable 'Wi-Fi scan throttling' in Developer options."))
            handler.postDelayed(requestRunnable, if (refusedStreak < 3) 3_000 else 10_000)
        }
    }

    @SuppressLint("MissingPermission")
    private fun onResults() {
        if (!running) return
        val now = SystemClock.elapsedRealtimeNanos()
        val results = try { wm.scanResults } catch (_: SecurityException) { emptyList() }
        val source = if (pendingAppScan) "app" else "system"
        pendingAppScan = false
        latest = results
        lastBatchSize = results.size
        val rows = results.map { r ->
            @Suppress("DEPRECATION") val ssid = r.SSID ?: ""
            ScanRow(
                bssid = r.BSSID ?: "00:00:00:00:00:00", ssid = ssid, rssiDbm = r.level, freqMhz = r.frequency,
                channelWidthMhz = ChannelWidth.toMhz(r.channelWidth),
                isRttResponder = r.is80211mcResponder,
                ageMs = (now - r.timestamp * 1000L) / 1_000_000L,
                tsNs = r.timestamp * 1000L,
            )
        }
        stats.onBatch(now)
        sink.emit(Events.wifiScan(now, scanId++, source, rows))
        handler.removeCallbacks(requestRunnable)
        handler.postDelayed(requestRunnable, 300)   // next scan right away; the OS decides whether it is allowed
    }
}
