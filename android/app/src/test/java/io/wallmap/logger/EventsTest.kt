package io.wallmap.logger

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class EventsTest {
    private fun parse(s: String) = Json.parseToJsonElement(s).jsonObject

    @Test fun wifiScanMatchesDataFormat() {
        val line = Events.wifiScan(
            1234L, 17, "app",
            listOf(
                ScanRow("AA:BB:CC:DD:EE:01", "Home \"net\"", -57, 5180, 80, false, 1500, 99_000_000L),
                ScanRow("aa:bb:cc:dd:ee:02", "", -81, 2437, null, null, null, null),
            ),
        )
        assertTrue("one line", !line.contains('\n'))
        val o = parse(line)
        assertEquals("wifi_scan", o["type"]!!.jsonPrimitive.content)
        assertEquals(1234L, o["t_ns"]!!.jsonPrimitive.long)
        assertEquals(17, o["scan_id"]!!.jsonPrimitive.int)
        val r = o["results"]!!.jsonArray
        assertEquals(2, r.size)
        val r0 = r[0].jsonObject
        assertEquals("aa:bb:cc:dd:ee:01", r0["bssid"]!!.jsonPrimitive.content)   // lower-cased
        assertEquals("Home \"net\"", r0["ssid"]!!.jsonPrimitive.content)          // escaping survives
        assertEquals(-57, r0["rssi_dbm"]!!.jsonPrimitive.int)
        assertEquals(80, r0["channel_width_mhz"]!!.jsonPrimitive.int)
        assertEquals(false, r0["is_rtt_responder"]!!.jsonPrimitive.boolean)
        assertEquals(99_000_000L, r0["ts_ns"]!!.jsonPrimitive.long)
        val r1 = r[1].jsonObject
        assertEquals(JsonNull, r1["is_rtt_responder"])                              // unknown -> null
        assertTrue("channel_width_mhz" !in r1)
    }

    @Test fun rttSuccessAndFailure() {
        val ok = parse(Events.rtt(5, "AA:BB:CC:00:00:01", 4210, 310, -60, 8, 7, "SUCCESS"))
        assertEquals(4210, ok["distance_mm"]!!.jsonPrimitive.int)
        assertEquals(7, ok["successful"]!!.jsonPrimitive.int)
        val bad = parse(Events.rtt(6, "aa:bb:cc:00:00:01", null, null, null, 8, 0, "FAIL(2)"))
        assertEquals(JsonNull, bad["distance_mm"])
        assertEquals(JsonNull, bad["rssi"])
        assertEquals(0, bad["successful"]!!.jsonPrimitive.int)
    }

    @Test fun imuTypesAndQuaternionOrder() {
        val acc = parse(Events.vec(1, "acc", floatArrayOf(0.5f, -1f, 9.81f)))
        assertEquals("acc", acc["type"]!!.jsonPrimitive.content)
        assertEquals(3, acc["v"]!!.jsonArray.size)
        assertEquals(9.81, acc["v"]!!.jsonArray[2].jsonPrimitive.double, 1e-5)
        val rv = parse(Events.vec(2, "rotvec", floatArrayOf(0.1f, 0.2f, 0.3f, 0.9f)))
        assertEquals(4, rv["v"]!!.jsonArray.size)
        assertEquals(0.9, rv["v"]!!.jsonArray[3].jsonPrimitive.double, 1e-6)   // w last
    }

    @Test fun baroGpsWaypointInfo() {
        assertEquals(1013.2, parse(Events.baro(1, 1013.2f))["hpa"]!!.jsonPrimitive.double, 1e-3)
        val g = parse(Events.gps(3, 51.5, -0.12, 8f, null))
        assertEquals(51.5, g["lat"]!!.jsonPrimitive.double, 1e-9)
        assertEquals(JsonNull, g["alt_m"])
        val w = parse(Events.waypoint(4, 3, "kitchen door"))
        assertEquals(3, w["marker"]!!.jsonPrimitive.int)
        assertEquals("kitchen door", w["label"]!!.jsonPrimitive.content)
        assertEquals("note", parse(Events.info(5, "note"))["msg"]!!.jsonPrimitive.content)
        assertEquals(false, parse(Events.scanRequest(6, false))["accepted"]!!.jsonPrimitive.boolean)
    }

    @Test fun channelWidthCodes() {
        assertEquals(20, ChannelWidth.toMhz(0)); assertEquals(80, ChannelWidth.toMhz(2)); assertEquals(160, ChannelWidth.toMhz(3))
        assertEquals(null, ChannelWidth.toMhz(42))
    }
}
