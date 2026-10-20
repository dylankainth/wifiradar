package io.wallmap.logger

import java.io.File
import java.security.SecureRandom
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/** Sessions live in <root>/<name>/{meta.json,events.jsonl}. Pure java.io, unit-tested. */
class SessionStore(val root: File) {
    init { root.mkdirs() }

    fun list(): List<String> = (root.listFiles() ?: emptyArray()).filter { File(it, "meta.json").exists() || File(it, "events.jsonl").exists() }
        .map { it.name }.sortedDescending()

    fun create(requested: String): File {
        val name = SessionNaming.unique(SessionNaming.sanitize(requested), (root.listFiles() ?: emptyArray()).map { it.name })
        return File(root, name).also { it.mkdirs() }
    }

    fun dir(name: String) = File(root, name)

    /**
     * Zip a session. With anonymise=true BSSIDs/SSIDs of APs not listed in meta.my_bssids (or `keepOverride`)
     * are replaced by salted pseudonyms; the salt is random per export and is not stored.
     */
    fun exportZip(name: String, outFile: File, anonymise: Boolean, salt: ByteArray = randomSalt(), keepOverride: Set<String>? = null): File {
        val dir = dir(name)
        outFile.parentFile?.mkdirs()
        ZipOutputStream(outFile.outputStream().buffered()).use { zip ->
            val metaFile = File(dir, "meta.json")
            var keep = keepOverride ?: emptySet()
            if (metaFile.exists()) {
                val meta = Json.parseToJsonElement(metaFile.readText()).jsonObject
                if (keepOverride == null) keep = meta["my_bssids"]?.jsonArray?.map { it.jsonPrimitive.content.lowercase() }?.toSet() ?: emptySet()
                val m = LinkedHashMap(meta)
                if (anonymise) m["anonymised"] = kotlinx.serialization.json.JsonPrimitive(true)
                zip.putNextEntry(ZipEntry("$name/meta.json"))
                zip.write(JsonObject(m).toString().toByteArray())
                zip.closeEntry()
            }
            val ev = File(dir, "events.jsonl")
            if (ev.exists()) {
                zip.putNextEntry(ZipEntry("$name/events.jsonl"))
                val cache = HashMap<String, String>()
                ev.bufferedReader().useLines { lines ->
                    for (l in lines) {
                        if (l.isBlank()) continue
                        val o = if (anonymise) Anonymizer.anonymiseLine(l, salt, keep, cache) else l
                        zip.write(o.toByteArray()); zip.write('\n'.code)
                    }
                }
                zip.closeEntry()
            }
        }
        return outFile
    }

    companion object {
        fun randomSalt(): ByteArray = ByteArray(16).also { SecureRandom().nextBytes(it) }
    }
}
