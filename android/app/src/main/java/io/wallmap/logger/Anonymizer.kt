package io.wallmap.logger

import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * Salted pseudonymisation of BSSIDs and SSIDs; byte-for-byte compatible with common/anonymize.py.
 * Keeps the LA bit and the last byte; bytes 0..4 (LA bit cleared) map to a keyed hash so BSSIDs of one radio
 * that differ only in the last byte (or the LA bit) stay adjacent.
 */
object Anonymizer {
    private fun hmac(salt: ByteArray, msg: ByteArray): ByteArray {
        val mac = Mac.getInstance("HmacSHA256")
        mac.init(SecretKeySpec(if (salt.isEmpty()) ByteArray(1) else salt, "HmacSHA256"))
        return mac.doFinal(msg)
    }

    fun anonBssid(bssid: String, salt: ByteArray): String {
        val b = Bssid.bytes(bssid)
        val la = b[0].toInt() and 0x02
        val prefix = ByteArray(5) { b[it] }
        prefix[0] = (prefix[0].toInt() and 0x02.inv() and 0xFF).toByte()
        val h = hmac(salt, "bssid:".toByteArray(Charsets.US_ASCII) + prefix)
        val out = ByteArray(6)
        out[0] = (((h[0].toInt() and 0xFC) or la) and 0xFF).toByte()
        for (i in 1..4) out[i] = h[i]
        out[5] = b[5]
        return Bssid.format(out)
    }

    fun anonSsid(ssid: String, salt: ByteArray): String {
        if (ssid.isEmpty()) return ""
        val h = hmac(salt, "ssid:".toByteArray(Charsets.UTF_8) + ssid.toByteArray(Charsets.UTF_8))
        return "ssid_" + h.take(4).joinToString("") { "%02x".format(it.toInt() and 0xFF) }
    }

    /** Anonymise one events.jsonl line. `keep` holds the user's own (lower-case) BSSIDs, left in clear. */
    fun anonymiseLine(line: String, salt: ByteArray, keep: Set<String>, cache: MutableMap<String, String> = HashMap()): String {
        if (line.isBlank()) return line
        val obj = Json.parseToJsonElement(line).jsonObject
        val type = obj["type"]?.jsonPrimitive?.contentOrNull
        fun ab(b: String): String {
            val k = b.lowercase()
            return if (k in keep) k else cache.getOrPut(k) { anonBssid(k, salt) }
        }
        return when (type) {
            "wifi_scan" -> {
                val rows = obj["results"]!!.jsonArray.map { r ->
                    val ro = r.jsonObject
                    val bssid = ro["bssid"]!!.jsonPrimitive.content
                    val mine = bssid.lowercase() in keep
                    val m = LinkedHashMap(ro)
                    m["ssid"] = JsonPrimitive(if (mine) ro["ssid"]!!.jsonPrimitive.content else anonSsid(ro["ssid"]?.jsonPrimitive?.contentOrNull ?: "", salt))
                    m["bssid"] = JsonPrimitive(ab(bssid))
                    JsonObject(m)
                }
                val m = LinkedHashMap(obj)
                m["results"] = JsonArray(rows)
                JsonObject(m).toString()
            }
            "rtt" -> {
                val m = LinkedHashMap(obj)
                m["bssid"] = JsonPrimitive(ab(obj["bssid"]!!.jsonPrimitive.content))
                JsonObject(m).toString()
            }
            else -> line
        }
    }
}
