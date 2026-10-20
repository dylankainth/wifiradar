package io.wallmap.logger

/** BSSID helpers mirroring solver/grouping.py (MAC similarity with the locally-administered bit cleared). */
object Bssid {
    private val rx = Regex("^([0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}$")

    fun isValid(s: String): Boolean = rx.matches(s.trim())

    /** lower-case, colon separated. Throws IllegalArgumentException for malformed input. */
    fun normalise(s: String): String {
        require(isValid(s)) { "not a MAC address: '$s'" }
        return s.trim().lowercase().replace('-', ':')
    }

    fun bytes(s: String): ByteArray = normalise(s).split(':').map { it.toInt(16).toByte() }.toByteArray()

    fun format(b: ByteArray): String = b.joinToString(":") { "%02x".format(it.toInt() and 0xFF) }

    fun isLocallyAdministered(s: String): Boolean = (bytes(s)[0].toInt() and 0x02) != 0

    fun clearLocallyAdministered(s: String): String {
        val b = bytes(s)
        b[0] = (b[0].toInt() and 0x02.inv()).toByte()
        return format(b)
    }

    /** |difference of the low 24 bits| if the (LA-cleared) OUIs agree, otherwise null. */
    fun macDistance(a: String, b: String): Long? {
        val x = bytes(clearLocallyAdministered(a))
        val y = bytes(clearLocallyAdministered(b))
        for (i in 0..2) if (x[i] != y[i]) return null
        fun low(v: ByteArray) = ((v[3].toLong() and 0xFF) shl 16) or ((v[4].toLong() and 0xFF) shl 8) or (v[5].toLong() and 0xFF)
        return kotlin.math.abs(low(x) - low(y))
    }
}
