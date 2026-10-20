package io.wallmap.logger

/** Achieved WiFi scan statistics, recorded in meta.json and shown live (pure logic, times in ns). */
class ScanStats {
    var requests = 0
        private set
    var accepted = 0
        private set
    private val batchTimes = ArrayList<Long>()
    val batches: Int get() = batchTimes.size

    fun onRequest(ok: Boolean) { requests++; if (ok) accepted++ }
    fun onBatch(tNs: Long) { batchTimes.add(tNs) }

    fun medianPeriodS(): Double? {
        if (batchTimes.size < 2) return null
        val d = batchTimes.zipWithNext { a, b -> (b - a) / 1e9 }.sorted()
        val n = d.size
        return if (n % 2 == 1) d[n / 2] else (d[n / 2 - 1] + d[n / 2]) / 2
    }

    /** Batches in the trailing window ending at nowNs, scaled to per minute. */
    fun perMinute(nowNs: Long, windowS: Double = 60.0): Double {
        val lo = nowNs - (windowS * 1e9).toLong()
        return batchTimes.count { it >= lo } * (60.0 / windowS)
    }

    /** Throttling is considered disabled if scans arrive faster than once per 20 s (stock Android: ~1 per 30 s). */
    fun throttlingDisabledHint(): Boolean? = medianPeriodS()?.let { it < 20.0 }
}

/** Achieved sample rate per sensor from first/last timestamps. */
class RateCounter {
    private var n = 0L
    private var first = 0L
    private var last = 0L
    fun add(tNs: Long) { if (n == 0L) first = tNs; last = tNs; n++ }
    val count: Long get() = n
    fun achievedHz(): Double? = if (n < 2 || last <= first) null else (n - 1) / ((last - first) / 1e9)
}
