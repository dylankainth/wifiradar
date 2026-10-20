"""Salted pseudonymisation of BSSIDs/SSIDs for export. Mirrored by android/.../Anonymizer.kt.

Keeps what AP grouping needs: LA bit of byte 0 and the last byte are unchanged; bytes 0..4 (LA bit
cleared) map to a keyed hash, so radios whose BSSIDs differ only in the last byte stay adjacent.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path


def _hmac(salt: bytes, msg: bytes) -> bytes:
    return hmac.new(salt, msg, hashlib.sha256).digest()


def anon_bssid(bssid: str, salt: bytes) -> str:
    b = bytes(int(x, 16) for x in bssid.lower().split(":"))
    la = b[0] & 0x02
    prefix = bytes([b[0] & ~0x02 & 0xFF]) + b[1:5]
    h = _hmac(salt, b"bssid:" + prefix)
    out = bytes([(h[0] & 0xFC) | la]) + h[1:5] + bytes([b[5]])  # keep unicast bit clear
    return ":".join(f"{x:02x}" for x in out)


def anon_ssid(ssid: str, salt: bytes) -> str:
    if not ssid:
        return ""
    return "ssid_" + _hmac(salt, b"ssid:" + ssid.encode()).hex()[:8]


def anonymize_session(src: str | Path, dst: str | Path, salt: bytes, keep: set[str] | None = None):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "meta.json").read_text())
    keep = {k.lower() for k in (keep if keep is not None else meta.get("my_bssids", []))}
    cache: dict[str, str] = {}

    def ab(b):
        b = b.lower()
        if b in keep:
            return b
        if b not in cache:
            cache[b] = anon_bssid(b, salt)
        return cache[b]

    def assid(bssid, s):
        return s if bssid.lower() in keep else anon_ssid(s, salt)

    with open(src / "events.jsonl") as fi, open(dst / "events.jsonl", "w") as fo:
        for line in fi:
            if not line.strip():
                continue
            e = json.loads(line)
            if e["type"] == "wifi_scan":
                for r in e["results"]:
                    r["ssid"] = assid(r["bssid"], r.get("ssid", ""))
                    r["bssid"] = ab(r["bssid"])
            elif e["type"] == "rtt":
                e["bssid"] = ab(e["bssid"])
            fo.write(json.dumps(e, separators=(",", ":")) + "\n")
    meta["anonymised"] = True
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
