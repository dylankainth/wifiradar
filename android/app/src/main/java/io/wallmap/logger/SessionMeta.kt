package io.wallmap.logger

import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonObjectBuilder
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray

/** Builds meta.json (docs/data-format.md). */
object SessionMeta {
    data class Device(val manufacturer: String, val model: String, val androidRelease: String, val sdkInt: Int)
    data class Sensor(val requestedHz: Double, val achievedHz: Double?)

    fun build(
        sessionName: String,
        appVersion: String,
        device: Device,
        startWallTimeIso: String,
        startElapsedNs: Long,
        durationS: Double,
        rttFeaturePresent: Boolean,
        rttAvailable: Boolean,
        rttAttempts: Int,
        rttSuccesses: Int,
        scan: ScanStats,
        sensors: Map<String, Sensor>,
        myBssids: Collection<String>,
        notes: String,
    ): JsonObject = buildJsonObject {
        put("format_version", 1)
        put("session_name", sessionName)
        put("app_version", appVersion)
        put("device", buildJsonObject {
            put("manufacturer", device.manufacturer); put("model", device.model)
            put("android_release", device.androidRelease); put("sdk_int", device.sdkInt)
        })
        put("start_wall_time", startWallTimeIso)
        put("start_elapsed_ns", startElapsedNs)
        put("duration_s", durationS)
        put("rtt", buildJsonObject {
            put("feature_present", rttFeaturePresent); put("available", rttAvailable)
            put("attempts", rttAttempts); put("successes", rttSuccesses)
        })
        put("scan", buildJsonObject {
            put("requests", scan.requests); put("accepted", scan.accepted); put("batches", scan.batches)
            val m = scan.medianPeriodS()
            if (m == null) put("median_period_s", JsonNull) else put("median_period_s", m)
            val h = scan.throttlingDisabledHint()
            if (h == null) put("throttling_disabled_hint", JsonNull) else put("throttling_disabled_hint", h)
        })
        put("sensors", buildJsonObject {
            for ((k, v) in sensors) put(k, buildJsonObject {
                put("requested_hz", v.requestedHz)
                if (v.achievedHz == null) put("achieved_hz", JsonNull) else put("achieved_hz", v.achievedHz)
            })
        })
        putJsonArray("my_bssids") { myBssids.sorted().forEach { add(kotlinx.serialization.json.JsonPrimitive(it.lowercase())) } }
        put("anonymised", false)
        put("notes", notes)
    }
}
