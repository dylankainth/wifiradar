package io.wallmap.logger

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread

/** Receives one finished JSONL line. */
fun interface EventSink { fun emit(line: String) }

/**
 * Accelerometer, gyroscope, magnetometer, rotation vector and barometer at ~50 Hz (SENSOR_DELAY_GAME class).
 * SensorEvent.timestamp is on the elapsedRealtimeNanos clock, the same as every other event.
 */
class SensorLogger(context: Context, private val sink: EventSink) : SensorEventListener {
    private val sm = context.applicationContext.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val thread = HandlerThread("wallmap-sensors")
    val rates: Map<String, RateCounter> = listOf("acc", "gyro", "mag", "rotvec", "baro").associateWith { RateCounter() }
    val requestedHz = 50.0
    val present = HashMap<String, Boolean>()
    private val quat = FloatArray(4)

    private val wanted = listOf(
        Triple(Sensor.TYPE_ACCELEROMETER, "acc", 20_000),
        Triple(Sensor.TYPE_GYROSCOPE, "gyro", 20_000),
        Triple(Sensor.TYPE_MAGNETIC_FIELD, "mag", 40_000),
        Triple(Sensor.TYPE_ROTATION_VECTOR, "rotvec", 20_000),
        Triple(Sensor.TYPE_PRESSURE, "baro", 200_000),
    )

    fun start() {
        thread.start()
        val handler = Handler(thread.looper)
        for ((type, name, periodUs) in wanted) {
            val s = sm.getDefaultSensor(type)
            present[name] = s != null
            if (s != null) sm.registerListener(this, s, periodUs, 0, handler)
        }
    }

    fun stop() {
        sm.unregisterListener(this)
        thread.quitSafely()
    }

    override fun onSensorChanged(e: SensorEvent) {
        val t = e.timestamp
        when (e.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> { rates["acc"]!!.add(t); sink.emit(Events.vec(t, "acc", e.values.copyOf(3))) }
            Sensor.TYPE_GYROSCOPE -> { rates["gyro"]!!.add(t); sink.emit(Events.vec(t, "gyro", e.values.copyOf(3))) }
            Sensor.TYPE_MAGNETIC_FIELD -> { rates["mag"]!!.add(t); sink.emit(Events.vec(t, "mag", e.values.copyOf(3))) }
            Sensor.TYPE_ROTATION_VECTOR -> {
                rates["rotvec"]!!.add(t)
                SensorManager.getQuaternionFromVector(quat, e.values)   // [w, x, y, z]
                sink.emit(Events.vec(t, "rotvec", floatArrayOf(quat[1], quat[2], quat[3], quat[0])))
            }
            Sensor.TYPE_PRESSURE -> { rates["baro"]!!.add(t); sink.emit(Events.baro(t, e.values[0])) }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
