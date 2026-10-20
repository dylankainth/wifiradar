package io.wallmap.logger

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.wifi.WifiManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import java.io.File

/** Foreground service so logging survives screen dimming and the app being backgrounded briefly. */
class LoggerService : Service() {
    private var wake: PowerManager.WakeLock? = null
    private var wifiLock: WifiManager.WifiLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> {
                startForegroundCompat()
                val rec = recorder(this)
                val mine = intent.getStringArrayListExtra(EXTRA_MINE)?.toSet() ?: emptySet()
                acquireLocks()
                rec.start(intent.getStringExtra(EXTRA_NAME) ?: "", mine)
            }
            ACTION_STOP -> {
                SessionRecorder.instance?.stop()
                releaseLocks()
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        SessionRecorder.instance?.stop()
        releaseLocks()
        super.onDestroy()
    }

    private fun acquireLocks() {
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        wake = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "wallmap:logging").apply { acquire(4 * 60 * 60 * 1000L) }
        val wm = applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
        @Suppress("DEPRECATION")
        val mode = if (Build.VERSION.SDK_INT >= 29) WifiManager.WIFI_MODE_FULL_LOW_LATENCY else WifiManager.WIFI_MODE_FULL_HIGH_PERF
        wifiLock = wm.createWifiLock(mode, "wallmap:wifi").apply { acquire() }
    }

    private fun releaseLocks() {
        try { wake?.takeIf { it.isHeld }?.release() } catch (_: Exception) {}
        try { wifiLock?.takeIf { it.isHeld }?.release() } catch (_: Exception) {}
    }

    private fun startForegroundCompat() {
        val nm = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= 26) {
            nm.createNotificationChannel(NotificationChannel(CHANNEL, "Logging", NotificationManager.IMPORTANCE_LOW))
        }
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val n: Notification = NotificationCompat.Builder(this, CHANNEL)
            .setContentTitle("wallmap logger recording")
            .setContentText("WiFi scans, IMU, barometer. Keep the screen on for RTT.")
            .setSmallIcon(android.R.drawable.ic_menu_mylocation)
            .setOngoing(true).setContentIntent(open).build()
        if (Build.VERSION.SDK_INT >= 29) {
            startForeground(NOTIF_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION)
        } else {
            startForeground(NOTIF_ID, n)
        }
    }

    companion object {
        const val ACTION_START = "io.wallmap.logger.START"
        const val ACTION_STOP = "io.wallmap.logger.STOP"
        const val EXTRA_NAME = "name"
        const val EXTRA_MINE = "mine"
        private const val CHANNEL = "logging"
        private const val NOTIF_ID = 1

        fun sessionsDir(ctx: Context): File = File(ctx.getExternalFilesDir(null) ?: ctx.filesDir, "sessions")

        fun recorder(ctx: Context): SessionRecorder = SessionRecorder.instance
            ?: SessionRecorder(ctx.applicationContext, SessionStore(sessionsDir(ctx))).also { SessionRecorder.instance = it }
    }
}
