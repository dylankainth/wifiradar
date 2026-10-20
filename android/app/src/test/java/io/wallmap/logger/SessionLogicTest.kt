package io.wallmap.logger

import java.io.File
import java.util.zip.ZipFile
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class SessionLogicTest {
    @get:Rule val tmp = TemporaryFolder()

    @Test fun sessionNames() {
        assertEquals("home_walk_1", SessionNaming.sanitize("  home walk 1 "))
        assertEquals("a_b_c", SessionNaming.sanitize("a/b\\c"))
        assertEquals("session", SessionNaming.sanitize("///"))
        assertEquals(60, SessionNaming.sanitize("x".repeat(200)).length)
        assertEquals("walk", SessionNaming.unique("walk", emptyList()))
        assertEquals("walk_2", SessionNaming.unique("walk", listOf("walk")))
        assertEquals("walk_3", SessionNaming.unique("walk", listOf("Walk", "walk_2")))
        assertEquals("walk_202610200915", SessionNaming.defaultName("2026-10-20T09:15:33Z"))
    }

    @Test fun markerCounter() {
        val m = MarkerCounter()
        assertEquals(1, m.press()); assertEquals(2, m.press())
        m.set(1)                         // revisit marker 1
        assertEquals(1, m.press()); assertEquals(2, m.next)
        m.step(-5); assertEquals(0, m.next)
        assertEquals(3, m.pressed)
    }

    @Test fun scanStats() {
        val s = ScanStats()
        assertNull(s.medianPeriodS()); assertNull(s.throttlingDisabledHint())
        s.onRequest(true); s.onRequest(false); s.onRequest(true)
        assertEquals(3, s.requests); assertEquals(2, s.accepted)
        val sec = 1_000_000_000L
        for (t in listOf(0L, 4 * sec, 8 * sec, 14 * sec, 18 * sec)) s.onBatch(t)
        assertEquals(5, s.batches)
        assertEquals(4.0, s.medianPeriodS()!!, 1e-9)
        assertEquals(true, s.throttlingDisabledHint())
        assertEquals(5.0, s.perMinute(60 * sec), 1e-9)      // all five inside a 60 s window ending at t=60
        val slow = ScanStats(); for (t in listOf(0L, 30 * sec, 60 * sec)) slow.onBatch(t)
        assertEquals(false, slow.throttlingDisabledHint())  // stock throttling: ~1 per 30 s
    }

    @Test fun rateCounter() {
        val r = RateCounter()
        assertNull(r.achievedHz())
        for (i in 0..100) r.add(i * 20_000_000L)            // 50 Hz
        assertEquals(50.0, r.achievedHz()!!, 1e-6)
    }

    @Test fun jsonlWriterWritesAllLinesInOrderAndClosesCleanly() {
        val f = tmp.newFile("events.jsonl")
        val w = JsonlWriter(f, flushEveryMs = 10)
        for (i in 0 until 5000) w.write(Events.baro(i.toLong(), 1000f))
        w.close()
        val lines = f.readLines()
        assertEquals(5000, lines.size)
        assertEquals(4999L, Json.parseToJsonElement(lines.last()).jsonObject["t_ns"]!!.jsonPrimitive.content.toLong())
    }

    private fun mkSession(store: SessionStore, name: String): File {
        val d = store.create(name)
        File(d, "events.jsonl").writeText(listOf(
            Events.wifiScan(10, 0, "app", listOf(
                ScanRow("aa:bb:cc:dd:ee:01", "Home", -50, 2412, 20, false, null, null),
                ScanRow("12:34:56:78:9a:0b", "Neighbour", -80, 5180, 80, false, null, null))),
            Events.waypoint(11, 1, ""),
        ).joinToString("\n", postfix = "\n"))
        val meta = SessionMeta.build(d.name, "0.1.0", SessionMeta.Device("samsung", "SM-S931B", "15", 35), "2026-10-20T10:00:00Z",
            1000L, 12.5, false, false, 0, 0, ScanStats().also { it.onBatch(0); it.onBatch(4_000_000_000L) },
            mapOf("acc" to SessionMeta.Sensor(50.0, 49.8)), listOf("AA:BB:CC:DD:EE:01"), "")
        File(d, "meta.json").writeText(meta.toString())
        return d
    }

    @Test fun metaJsonHasRequiredFields() {
        val store = SessionStore(tmp.newFolder("sessions"))
        val d = mkSession(store, "s1")
        val m = Json.parseToJsonElement(File(d, "meta.json").readText()).jsonObject
        for (k in listOf("format_version", "session_name", "app_version", "device", "start_elapsed_ns", "duration_s", "rtt", "scan", "sensors", "my_bssids", "anonymised"))
            assertNotNull("missing $k", m[k])
        assertEquals(false, m["rtt"]!!.jsonObject["available"]!!.jsonPrimitive.boolean)
        assertEquals(4.0, m["scan"]!!.jsonObject["median_period_s"]!!.jsonPrimitive.content.toDouble(), 1e-9)
        assertEquals(JsonNull, SessionMeta.build("x", "v", SessionMeta.Device("", "", "", 0), "", 0, 0.0, false, false, 0, 0,
            ScanStats(), emptyMap(), emptyList(), "")["scan"]!!.jsonObject["median_period_s"])
        assertEquals("aa:bb:cc:dd:ee:01", m["my_bssids"]!!.jsonArray[0].jsonPrimitive.content)
    }

    @Test fun storeCreatesUniqueDirsAndListsThem() {
        val store = SessionStore(tmp.newFolder("sessions"))
        val a = mkSession(store, "walk"); val b = mkSession(store, "walk")
        assertEquals("walk", a.name); assertEquals("walk_2", b.name)
        assertEquals(setOf("walk", "walk_2"), store.list().toSet())
    }

    @Test fun exportZipPlainAndAnonymised() {
        val store = SessionStore(tmp.newFolder("sessions"))
        mkSession(store, "s1")
        val plain = store.exportZip("s1", File(tmp.root, "plain.zip"), anonymise = false)
        ZipFile(plain).use { z ->
            assertEquals(setOf("s1/meta.json", "s1/events.jsonl"), z.entries().asSequence().map { it.name }.toSet())
            val ev = z.getInputStream(z.getEntry("s1/events.jsonl")).bufferedReader().readText()
            assertTrue("12:34:56:78:9a:0b" in ev && "Neighbour" in ev)
        }
        val salt = ByteArray(16) { it.toByte() }
        val anon = store.exportZip("s1", File(tmp.root, "anon.zip"), anonymise = true, salt = salt)
        ZipFile(anon).use { z ->
            val ev = z.getInputStream(z.getEntry("s1/events.jsonl")).bufferedReader().readLines()
            assertEquals(2, ev.size)
            assertTrue("aa:bb:cc:dd:ee:01" in ev[0] && "Home" in ev[0])                  // mine stays clear (from meta.my_bssids)
            assertTrue("12:34:56:78:9a:0b" !in ev[0] && "Neighbour" !in ev[0])           // neighbour hashed
            assertTrue(":0b" in ev[0])                                                    // last byte kept for AP grouping
            assertEquals(Events.waypoint(11, 1, ""), ev[1])                               // other events untouched
            val meta = Json.parseToJsonElement(z.getInputStream(z.getEntry("s1/meta.json")).bufferedReader().readText()).jsonObject
            assertEquals(true, meta["anonymised"]!!.jsonPrimitive.boolean)
        }
    }
}
