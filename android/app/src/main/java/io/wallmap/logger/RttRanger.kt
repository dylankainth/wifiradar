package io.wallmap.logger

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.net.wifi.ScanResult
import android.net.wifi.rtt.RangingRequest
import android.net.wifi.rtt.RangingResult
import android.net.wifi.rtt.RangingResultCallback
import android.net.wifi.rtt.WifiRttManager
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import java.util.concurrent.Executors

/**
 * WiFi RTT (802.11mc FTM) ranging to scan results that advertise is80211mcResponder.
 * Optional: if FEATURE_WIFI_RTT is missing, or WifiRttManager.isAvailable() is false, or no responders are in range,
 * nothing is ranged, the reason is logged and the session carries on. RTT only works with the app in the foreground.
 */
class RttRanger(
    context: Context,
    private val sink: EventSink,
    private val targets: () -> List<ScanResult>,
) {
    private val ctx = context.applicationContext
    val featurePresent: Boolean = ctx.packageManager.hasSystemFeature(PackageManager.FEATURE_WIFI_RTT)
    private val rtt: WifiRttManager? = if (featurePresent) ctx.getSystemService(Context.WIFI_RTT_RANGING_SERVICE) as? WifiRttManager else null
    private val handler = Handler(Looper.getMainLooper())
    private val executor = Executors.newSingleThreadExecutor()
    private var running = false
    var attempts = 0
        private set
    var successes = 0
        private set
    @Volatile var lastMessage = ""
        private set

    val available: Boolean get() = featurePresent && rtt?.isAvailable == true

    fun start() {
        running = true
        if (!available) {
            lastMessage = if (!featurePresent) "RTT: feature not present" else "RTT: not available right now (WiFi off or location off?)"
            sink.emit(Events.info(SystemClock.elapsedRealtimeNanos(), lastMessage))
            return
        }
        handler.postDelayed({ rangeOnce() }, 2000)
    }

    fun stop() {
        running = false
        handler.removeCallbacksAndMessages(null)
        executor.shutdown()
    }

    @SuppressLint("MissingPermission")
    private fun rangeOnce() {
        if (!running) return
        val mgr = rtt
        val responders = targets().filter { it.is80211mcResponder }
            .sortedByDescending { it.level }.take(RangingRequest.getMaxPeers())
        if (mgr == null || !mgr.isAvailable || responders.isEmpty()) {
            handler.postDelayed({ rangeOnce() }, 3000)
            return
        }
        val req = RangingRequest.Builder().addAccessPoints(responders).build()
        try {
            mgr.startRanging(req, executor, object : RangingResultCallback() {
                override fun onRangingFailure(code: Int) {
                    attempts += responders.size
                    sink.emit(Events.info(SystemClock.elapsedRealtimeNanos(), "RTT ranging failure code=$code"))
                    next()
                }

                override fun onRangingResults(results: List<RangingResult>) {
                    val now = SystemClock.elapsedRealtimeNanos()
                    for (r in results) {
                        attempts++
                        val ok = r.status == RangingResult.STATUS_SUCCESS
                        if (ok) successes++
                        val mac = r.macAddress?.toString() ?: continue
                        sink.emit(Events.rtt(now, mac,
                            if (ok) r.distanceMm else null, if (ok) r.distanceStdDevMm else null,
                            if (ok) r.rssi else null, r.numAttemptedMeasurements, r.numSuccessfulMeasurements,
                            if (ok) "SUCCESS" else "FAIL(${r.status})"))
                    }
                    next()
                }
            })
        } catch (e: SecurityException) {
            lastMessage = "RTT: permission denied"
            sink.emit(Events.info(SystemClock.elapsedRealtimeNanos(), lastMessage))
        }
    }

    private fun next() { handler.postDelayed({ rangeOnce() }, 1000) }
}
