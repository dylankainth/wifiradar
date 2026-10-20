from common.anonymize import anon_bssid, anon_ssid, anonymize_session
from common.session import SessionWriter, load_session

SALT = b"salt"


def _mk(tmp_path):
    w = SessionWriter(tmp_path / "s", {"start_elapsed_ns": 1000})
    w.add(2_000_000_000, "wifi_scan", results=[
        {"bssid": "AA:BB:CC:DD:EE:01", "ssid": "Home", "rssi_dbm": -50, "freq_mhz": 2412},
        {"bssid": "12:34:56:78:9a:0b", "ssid": "Neighbour", "rssi_dbm": -80, "freq_mhz": 5180}])
    w.add(3_000_000_000, "acc", v=[0, 0, 9.8])
    w.add(3_000_000_001, "waypoint", marker=1, label="a")
    w.meta["my_bssids"] = ["aa:bb:cc:dd:ee:01"]
    w.close()
    return tmp_path / "s"


def test_roundtrip(tmp_path):
    p = _mk(tmp_path)
    s = load_session(p)
    assert len(s.scans) == 1 and s.scans[0].obs[0].bssid == "aa:bb:cc:dd:ee:01"
    assert abs(s.scans[0].t_s - 2.0) < 1e-6 and s.imu["acc"][1].shape == (1, 3)
    assert s.waypoints[0]["marker"] == 1


def test_truncated_line_tolerated(tmp_path):
    p = _mk(tmp_path)
    with open(p / "events.jsonl", "a") as f:
        f.write('{"t_ns": 5, "type": "ac')
    assert len(load_session(p).scans) == 1


def test_anonymize_keeps_mine_and_structure(tmp_path):
    p = _mk(tmp_path)
    anonymize_session(p, tmp_path / "a", SALT)
    s = load_session(tmp_path / "a")
    mine, other = s.scans[0].obs
    assert mine.bssid == "aa:bb:cc:dd:ee:01" and mine.ssid == "Home"
    assert other.bssid != "12:34:56:78:9a:0b" and other.ssid.startswith("ssid_")
    assert other.bssid.endswith(":0b")


def test_anon_bssid_adjacency_and_la_bit():
    a = anon_bssid("a0:b1:c2:d3:e4:10", SALT)
    b = anon_bssid("a2:b1:c2:d3:e4:11", SALT)   # LA-bit variant, last byte +1
    assert a.split(":")[1:5] == b.split(":")[1:5]
    assert int(a[:2], 16) & 0x02 == 0 and int(b[:2], 16) & 0x02 == 0x02
    assert (int(a[:2], 16) & ~2) == (int(b[:2], 16) & ~2)
    assert anon_bssid("a0:b1:c2:d3:e4:10", b"other") != a
    assert anon_ssid("", SALT) == ""
