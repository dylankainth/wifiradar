package io.wallmap.logger

import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.atomic.AtomicLong

/** Append-only JSONL writer on its own thread so sensor callbacks never block on disk. */
class JsonlWriter(file: File, private val flushEveryMs: Long = 1000) {
    private val queue = LinkedBlockingQueue<String>()
    private val fos = FileOutputStream(file, true)
    private val out = BufferedWriter(OutputStreamWriter(fos, Charsets.UTF_8), 1 shl 16)
    private val written = AtomicLong(0)
    @Volatile private var closing = false
    private val thread = Thread({ loop() }, "jsonl-writer").apply { isDaemon = true; start() }

    val linesWritten: Long get() = written.get()

    fun write(line: String) { if (!closing) queue.offer(line) }

    private fun loop() {
        var lastFlush = System.currentTimeMillis()
        try {
            while (true) {
                val line = queue.poll(200, java.util.concurrent.TimeUnit.MILLISECONDS)
                if (line != null) {
                    out.write(line); out.write('\n'.code); written.incrementAndGet()
                } else if (closing) break
                val now = System.currentTimeMillis()
                if (now - lastFlush >= flushEveryMs) { out.flush(); lastFlush = now }
            }
        } finally {
            try { out.flush(); fos.fd.sync() } catch (_: Exception) {}
            try { out.close() } catch (_: Exception) {}
        }
    }

    /** Drains the queue, fsyncs and closes. */
    fun close() {
        closing = true
        thread.join(5000)
    }
}
