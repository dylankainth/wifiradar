package io.wallmap.logger

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class BssidAndAnonymizerTest {
    // Reference vectors produced by common/anonymize.py with this salt (see tests/test_session_io.py for the algorithm)
    private val salt = "00112233445566778899aabbccddeeff".chunked(2).map { it.toInt(16).toByte() }.toByteArray()

    @Test fun matchesPythonImplementation() {
        assertEquals("a8:cf:19:47:bc:10", Anonymizer.anonBssid("a0:b1:c2:d3:e4:10", salt))
        assertEquals("aa:cf:19:47:bc:11", Anonymizer.anonBssid("a2:b1:c2:d3:e4:11", salt))
        assertEquals("c2:f3:c2:00:46:0b", Anonymizer.anonBssid("12:34:56:78:9a:0b", salt))
        assertEquals("20:e2:f6:bc:0c:10", Anonymizer.anonBssid("C4:E9:84:5A:10:10", salt))
        assertEquals("22:e2:f6:bc:0c:12", Anonymizer.anonBssid("c6:e9:84:5a:10:12", salt))
        assertEquals("ssid_7b972307", Anonymizer.anonSsid("HomeNet", salt))
        assertEquals("ssid_51064a69", Anonymizer.anonSsid("Café 5G", salt))
        assertEquals("", Anonymizer.anonSsid("", salt))
    }

    @Test fun keepsGroupingStructure() {
        val a = Anonymizer.anonBssid("a0:b1:c2:d3:e4:10", salt)
        val b = Anonymizer.anonBssid("a2:b1:c2:d3:e4:11", salt)      // LA-bit variant, last byte + 1
        assertEquals(Bssid.macDistance(a, b), 1L)
        assertFalse(Bssid.isLocallyAdministered(a)); assertTrue(Bssid.isLocallyAdministered(b))
        assertNotEquals(a, Anonymizer.anonBssid("a0:b1:c2:d3:e4:10", ByteArray(16) { 1 }))   // salt matters
    }

    @Test fun anonymisesScanLinesButKeepsMine() {
        val line = Events.wifiScan(1, 0, "app", listOf(
            ScanRow("aa:bb:cc:dd:ee:01", "Home", -50, 2412, 20, false, null, null),
            ScanRow("12:34:56:78:9a:0b", "Neighbour", -80, 5180, 80, false, null, null)))
        val out = Json.parseToJsonElement(Anonymizer.anonymiseLine(line, salt, setOf("aa:bb:cc:dd:ee:01"))).jsonObject
        val r = out["results"]!!.jsonArray.map { it.jsonObject }
        assertEquals("aa:bb:cc:dd:ee:01", r[0]["bssid"]!!.jsonPrimitive.content)
        assertEquals("Home", r[0]["ssid"]!!.jsonPrimitive.content)
        assertEquals("c2:f3:c2:00:46:0b", r[1]["bssid"]!!.jsonPrimitive.content)
        assertTrue(r[1]["ssid"]!!.jsonPrimitive.content.startsWith("ssid_"))
        assertEquals(-80, r[1]["rssi_dbm"]!!.jsonPrimitive.content.toInt())              // data untouched
        val rtt = Anonymizer.anonymiseLine(Events.rtt(1, "12:34:56:78:9a:0b", 1000, 100, -50, 8, 8, "SUCCESS"), salt, emptySet())
        assertTrue("c2:f3:c2:00:46:0b" in rtt)
        val imu = Events.vec(1, "acc", floatArrayOf(1f, 2f, 3f))
        assertEquals(imu, Anonymizer.anonymiseLine(imu, salt, emptySet()))
    }

    @Test fun bssidParsing() {
        assertEquals("aa:bb:cc:dd:ee:ff", Bssid.normalise(" AA-BB-CC-DD-EE-FF "))
        assertFalse(Bssid.isValid("aa:bb:cc:dd:ee")); assertFalse(Bssid.isValid("zz:bb:cc:dd:ee:ff"))
        try { Bssid.normalise("nope"); fail() } catch (_: IllegalArgumentException) {}
    }

    @Test fun macDistanceClearsLocallyAdministeredBit() {
        assertEquals(2L, Bssid.macDistance("c4:e9:84:5a:10:10", "c6:e9:84:5a:10:12"))
        assertEquals(8L, Bssid.macDistance("c4:e9:84:5a:10:10", "c4:e9:84:5a:10:18"))   // the mesh-extender trap
        assertNull(Bssid.macDistance("c4:e9:84:5a:10:10", "00:e9:84:5a:10:12"))
        assertEquals("c4:e9:84:5a:10:10", Bssid.clearLocallyAdministered("c6:e9:84:5a:10:10"))
    }
}
