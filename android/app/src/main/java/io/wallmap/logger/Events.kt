package io.wallmap.logger

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObjectBuilder
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/** One BSSID observation inside a `wifi_scan` event (see docs/data-format.md). */
data class ScanRow(
    val bssid: String,
    val ssid: String,
    val rssiDbm: Int,
    val freqMhz: Int,
    val channelWidthMhz: Int?,
    val isRttResponder: Boolean?,
    val ageMs: Long?,
    val tsNs: Long?,
)

/** Pure serialisation of every event type to one compact JSON line. No Android types, so it is unit-testable. */
object Events {
    private fun obj(tNs: Long, type: String, body: JsonObjectBuilder.() -> Unit): String =
        buildJsonObject {
            put("t_ns", tNs)
            put("type", type)
            body()
        }.toString()

    private fun nullable(b: JsonObjectBuilder, key: String, v: Long?) { if (v == null) b.put(key, JsonNull) else b.put(key, v) }

    fun wifiScan(tNs: Long, scanId: Int, source: String, rows: List<ScanRow>): String = obj(tNs, "wifi_scan") {
        put("scan_id", scanId)
        put("source", source)
        put("results", JsonArray(rows.map { r ->
            buildJsonObject {
                put("bssid", r.bssid.lowercase())
                put("ssid", r.ssid)
                put("rssi_dbm", r.rssiDbm)
                put("freq_mhz", r.freqMhz)
                if (r.channelWidthMhz != null) put("channel_width_mhz", r.channelWidthMhz)
                if (r.isRttResponder != null) put("is_rtt_responder", r.isRttResponder) else put("is_rtt_responder", JsonNull)
                if (r.ageMs != null) put("age_ms", r.ageMs)
                if (r.tsNs != null) put("ts_ns", r.tsNs)
            }
        }))
    }

    fun scanRequest(tNs: Long, accepted: Boolean): String = obj(tNs, "scan_request") { put("accepted", accepted) }

    /** distanceMm == null for failed ranges. */
    fun rtt(
        tNs: Long, bssid: String, distanceMm: Int?, distanceStdMm: Int?, rssi: Int?,
        attempted: Int, successful: Int, status: String,
    ): String = obj(tNs, "rtt") {
        put("bssid", bssid.lowercase())
        if (distanceMm == null) put("distance_mm", JsonNull) else put("distance_mm", distanceMm)
        if (distanceStdMm == null) put("distance_std_mm", JsonNull) else put("distance_std_mm", distanceStdMm)
        if (rssi == null) put("rssi", JsonNull) else put("rssi", rssi)
        put("attempted", attempted)
        put("successful", successful)
        put("status", status)
    }

    /** type is one of acc, gyro, mag, rotvec. rotvec values are [x, y, z, w]. */
    fun vec(tNs: Long, type: String, v: FloatArray): String = obj(tNs, "$type") {
        put("v", JsonArray(v.map { JsonPrimitive(it.toDouble()) }))
    }

    fun baro(tNs: Long, hpa: Float): String = obj(tNs, "baro") { put("hpa", hpa.toDouble()) }

    fun gps(tNs: Long, lat: Double, lon: Double, accM: Float?, altM: Double?): String = obj(tNs, "gps") {
        put("lat", lat)
        put("lon", lon)
        if (accM == null) put("acc_m", JsonNull) else put("acc_m", accM.toDouble())
        if (altM == null) put("alt_m", JsonNull) else put("alt_m", altM)
    }

    fun waypoint(tNs: Long, marker: Int, label: String): String = obj(tNs, "waypoint") {
        put("marker", marker)
        put("label", label)
    }

    fun info(tNs: Long, msg: String): String = obj(tNs, "info") { put("msg", msg) }
}

/** ScanResult.channelWidth constant -> MHz. */
object ChannelWidth {
    fun toMhz(code: Int): Int? = when (code) {
        0 -> 20
        1 -> 40
        2 -> 80
        3 -> 160
        4 -> 160 // 80+80 MHz: reported as 160 MHz of spectrum
        5 -> 320
        else -> null
    }
}
