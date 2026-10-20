package io.wallmap.logger

import android.annotation.SuppressLint
import android.content.Context
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Looper

/** GPS fixes, only when available (indoors there are usually none). */
class GpsLogger(context: Context, private val sink: EventSink) : LocationListener {
    private val lm = context.applicationContext.getSystemService(Context.LOCATION_SERVICE) as LocationManager
    var fixes = 0
        private set

    @SuppressLint("MissingPermission")
    fun start(): Boolean = try {
        if (lm.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
            lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, 2000L, 0f, this, Looper.getMainLooper()); true
        } else false
    } catch (_: SecurityException) { false }

    fun stop() { try { lm.removeUpdates(this) } catch (_: SecurityException) {} }

    override fun onLocationChanged(l: Location) {
        fixes++
        sink.emit(Events.gps(l.elapsedRealtimeNanos, l.latitude, l.longitude,
            if (l.hasAccuracy()) l.accuracy else null, if (l.hasAltitude()) l.altitude else null))
    }

    @Deprecated("required on API < 30") override fun onStatusChanged(p: String?, s: Int, e: android.os.Bundle?) {}
    override fun onProviderEnabled(p: String) {}
    override fun onProviderDisabled(p: String) {}
}
