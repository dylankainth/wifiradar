package io.wallmap.logger

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.Checkbox
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import java.io.File

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // RTT and fast scanning only work in the foreground: keep the screen on while the app is visible
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContent { MaterialTheme { Screen() } }
    }
}

/** Needed to scan and range at all. Notifications (API 33+) are requested too but logging works without them. */
private fun requiredPermissions(): List<String> = buildList {
    add(Manifest.permission.ACCESS_FINE_LOCATION)
    if (Build.VERSION.SDK_INT >= 33) add(Manifest.permission.NEARBY_WIFI_DEVICES)
}

private fun requestablePermissions(): List<String> =
    requiredPermissions() + if (Build.VERSION.SDK_INT >= 33) listOf(Manifest.permission.POST_NOTIFICATIONS) else emptyList()

private fun missing(ctx: Context, list: List<String> = requiredPermissions()) = list.filter {
    ContextCompat.checkSelfPermission(ctx, it) != PackageManager.PERMISSION_GRANTED
}

private const val PREFS = "wallmap"

@Composable
private fun Screen() {
    val ctx = LocalContext.current
    val prefs = remember { ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE) }
    val rec = remember { LoggerService.recorder(ctx) }
    val ui by rec.state.collectAsStateWithLifecycle()
    var name by remember { mutableStateOf(prefs.getString("name", "") ?: "") }
    var label by remember { mutableStateOf("") }
    var mine by remember { mutableStateOf(prefs.getStringSet("mine", emptySet())!!.toSet()) }
    var showMine by remember { mutableStateOf(false) }
    var denied by remember { mutableStateOf(missing(ctx, requestablePermissions())) }
    var anonymise by remember { mutableStateOf(true) }
    val store = remember { SessionStore(LoggerService.sessionsDir(ctx)) }
    var sessions by remember { mutableStateOf(store.list()) }
    var exportMsg by remember { mutableStateOf("") }

    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { denied = missing(ctx, requestablePermissions()) }

    Column(Modifier.padding(16.dp).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text("wallmap logger", style = MaterialTheme.typography.headlineSmall)

        if (denied.isNotEmpty()) {
            Card { Column(Modifier.padding(12.dp)) {
                Text("Permissions needed: " + denied.joinToString { it.substringAfterLast('.') })
                Button(onClick = { launcher.launch(denied.toTypedArray()) }) { Text("Grant permissions") }
            } }
        }

        OutlinedTextField(value = name, onValueChange = { name = it; prefs.edit().putString("name", it).apply() },
            label = { Text("Session name") }, singleLine = true, enabled = !ui.recording, modifier = Modifier.fillMaxWidth())

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (!ui.recording) {
                Button(enabled = missing(ctx).isEmpty(), onClick = {
                    val i = Intent(ctx, LoggerService::class.java).setAction(LoggerService.ACTION_START)
                        .putExtra(LoggerService.EXTRA_NAME, name)
                        .putStringArrayListExtra(LoggerService.EXTRA_MINE, ArrayList(mine))
                    ContextCompat.startForegroundService(ctx, i)
                }) { Text("Start") }
            } else {
                Button(onClick = {
                    ctx.startService(Intent(ctx, LoggerService::class.java).setAction(LoggerService.ACTION_STOP))
                    sessions = store.list()
                }) { Text("Stop") }
            }
            OutlinedButton(onClick = { showMine = true }) { Text("My APs (${mine.size})") }
        }

        if (ui.recording) {
            Card { Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("Marker ${ui.nextMarker}", style = MaterialTheme.typography.titleMedium)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = { rec.stepMarker(-1) }) { Text("-") }
                    OutlinedButton(onClick = { rec.stepMarker(+1) }) { Text("+") }
                    OutlinedTextField(value = label, onValueChange = { label = it }, label = { Text("label (optional)") },
                        singleLine = true, modifier = Modifier.weight(1f))
                }
                Button(onClick = { rec.marker(label); label = "" }, modifier = Modifier.fillMaxWidth()) {
                    Text("I am at marker ${ui.nextMarker}")
                }
            } }
        }

        Card { Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(if (ui.recording) "Recording ${ui.sessionName}  %.0f s".format(ui.elapsedS) else "Idle")
            Text("WiFi scans: ${ui.scanBatches} batches, ${"%.1f".format(ui.scansPerMin)}/min, ${ui.lastScanAps} APs in last; " +
                "startScan accepted ${ui.scanAccepted}/${ui.scanRequests}")
            Text("IMU: acc ${ui.accHz?.let { "%.0f Hz".format(it) } ?: "-"}, rotvec ${ui.rotvecHz?.let { "%.0f Hz".format(it) } ?: "-"}")
            Text("RTT: feature ${ui.rttFeature}, available ${ui.rttAvailable}, success " +
                if (ui.rttAttempts > 0) "%.0f %% (%d/%d)".format(100.0 * ui.rttSuccesses / ui.rttAttempts, ui.rttSuccesses, ui.rttAttempts) else "n/a")
            Text("Markers pressed: ${ui.markersPressed}   GPS fixes: ${ui.gpsFixes}")
            if (ui.message.isNotEmpty()) Text(ui.message, style = MaterialTheme.typography.bodySmall)
        } }

        Text("Sessions", style = MaterialTheme.typography.titleMedium)
        Row { Checkbox(checked = anonymise, onCheckedChange = { anonymise = it }); Text("Anonymise other people's APs on export", Modifier.padding(top = 12.dp)) }
        for (s in sessions) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(s, Modifier.padding(top = 12.dp))
                OutlinedButton(enabled = !ui.recording, onClick = {
                    try {
                        val out = File(ctx.cacheDir, "exports/$s${if (anonymise) "_anon" else ""}.zip")
                        store.exportZip(s, out, anonymise)
                        val uri = FileProvider.getUriForFile(ctx, "${ctx.packageName}.fileprovider", out)
                        val send = Intent(Intent.ACTION_SEND).setType("application/zip").putExtra(Intent.EXTRA_STREAM, uri)
                            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                        ctx.startActivity(Intent.createChooser(send, "Export $s"))
                        exportMsg = "exported ${out.name}"
                    } catch (e: Exception) { exportMsg = "export failed: ${e.message}" }
                }) { Text("Export") }
            }
        }
        if (exportMsg.isNotEmpty()) Text(exportMsg, style = MaterialTheme.typography.bodySmall)
    }

    if (showMine) {
        val seen = ui.latestBssids
        val picked = remember { mutableStateListOf<String>().also { it.addAll(mine) } }
        AlertDialog(
            onDismissRequest = { showMine = false },
            title = { Text("My access points") },
            text = {
                Column(Modifier.verticalScroll(rememberScrollState())) {
                    Text("Tick your own APs; they stay readable on export. Open the app and wait for a scan first.",
                        style = MaterialTheme.typography.bodySmall)
                    val all = (seen.map { it.first.lowercase() } + picked).distinct()
                    for (b in all) {
                        val ssid = seen.firstOrNull { it.first.equals(b, true) }?.second ?: ""
                        Row { Checkbox(checked = b in picked, onCheckedChange = { c -> if (c) picked.add(b) else picked.remove(b) })
                            Text("$b  $ssid", Modifier.padding(top = 12.dp)) }
                    }
                }
            },
            confirmButton = { TextButton(onClick = {
                mine = picked.toSet(); prefs.edit().putStringSet("mine", mine).apply(); showMine = false
            }) { Text("Save") } },
        )
    }
}
