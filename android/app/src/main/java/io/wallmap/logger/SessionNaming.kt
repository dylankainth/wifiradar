package io.wallmap.logger

/** Session folder naming and marker bookkeeping (pure logic). */
object SessionNaming {
    fun sanitize(name: String, fallback: String = "session"): String {
        val s = name.trim().replace(Regex("[^A-Za-z0-9._-]+"), "_").trim('_', '.', '-').take(60)
        return s.ifEmpty { fallback }
    }

    /** base, base_2, base_3, ... whichever is not in `existing` (case-insensitive, as many filesystems are). */
    fun unique(base: String, existing: Collection<String>): String {
        val taken = existing.map { it.lowercase() }.toSet()
        if (base.lowercase() !in taken) return base
        var i = 2
        while ("${base}_$i".lowercase() in taken) i++
        return "${base}_$i"
    }

    fun defaultName(isoLocalStamp: String): String = "walk_" + isoLocalStamp.replace(Regex("[^0-9]"), "").take(12)
}

/** "I am at marker N": the id increments after every press; the user may set it back to revisit a marker. */
class MarkerCounter(start: Int = 1) {
    var next: Int = start
        private set
    var pressed: Int = 0
        private set

    fun press(): Int { val id = next; next += 1; pressed += 1; return id }
    fun set(id: Int) { next = id.coerceAtLeast(0) }
    fun step(delta: Int) { next = (next + delta).coerceAtLeast(0) }
}
