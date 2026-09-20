import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd
import soundfile as sf

from annotation.labels import load_labels, weighted_rate
from annotation.sampling import Near, merge_events, order_queue, thin
from annotation.server import State, make_handler


def test_near_finds_flagged_times_within_the_same_file_only():
    near = Near(np.array([0, 0, 1]), np.array([10.0, 50.0, 30.0]))
    assert near(0, 12, 3) and not near(0, 20, 3) and near(1, 31, 2) and not near(1, 10, 3)


def test_thin_keeps_best_of_each_cluster():
    c = pd.DataFrame({"fid": [0, 0, 0, 1], "focal": [10.0, 12.0, 30.0, 11.0]})
    assert list(thin(c, 6).focal) == [10.0, 30.0, 11.0]


def test_merge_events_uses_hop_and_keeps_the_best_window():
    d = pd.DataFrame({"fid": [0, 0, 0, 0], "k": [3, 3, 3, 3], "start": [0, 3, 6, 15], "p": [0.2, 0.9, 0.3, 0.5]})
    ev = merge_events(d, 3)
    assert list(ev.start) == [3, 15] and list(ev.n) == [3, 1]


def item(i, stratum):
    return dict(id=f"m{i}", kind="main", site="murie", rel="a.wav", clip=[0, 10], focal=[3, 8], stratum=stratum, pop=100, tier=1,
                hidden=dict(cx_p=[0] * 5, bn_p=[0] * 5, hd_p=[0] * 5, hour=1.0))


def test_order_queue_interleaves_strata_adds_hidden_repeats_and_tiers():
    items = [item(i, "a" if i % 2 else "b") for i in range(400)]
    q = order_queue(items, np.random.default_rng(0))
    first = [x["stratum"] for x in q[:100] if x["stratum"] != "dup"]
    assert abs(first.count("a") - first.count("b")) <= 12  # any prefix is roughly proportional
    dups = [x for x in q if x["stratum"] == "dup"]
    assert len(dups) == 20 and all(x["id"] == x["dup_of"] + "d" for x in dups)
    assert q[0]["tier"] == 1 and q[-1]["tier"] == 2 and all(x["id"] != x["dup_of"] for x in dups)


def make_dir(tmp_path):
    (tmp_path / "reference").mkdir()
    sf.write(tmp_path / "reference" / "x.wav", np.zeros(8000, np.float32), 8000, subtype="PCM_16")
    it = dict(item(1, "cx_test"), wav="reference/x.wav", title="t")
    (tmp_path / "manifest.json").write_text(json.dumps(dict(queue=[it], reference=[], practice=[])))
    return tmp_path


def test_server_hides_scores_requires_key_serves_clips_and_stores_labels(tmp_path):
    st = State(make_dir(tmp_path))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(st))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        try:
            urllib.request.urlopen(base + "/api/manifest")
            raise AssertionError("expected 403")
        except urllib.error.HTTPError as e:
            assert e.code == 403
        man = json.load(urllib.request.urlopen(f"{base}/api/manifest?k={st.key}"))
        assert set(man["queue"][0]) <= {"id", "kind", "clip", "focal", "truth", "title", "species", "tier"}
        wav = urllib.request.urlopen(f"{base}/api/clip/m1.wav?k={st.key}").read()
        assert wav[:4] == b"RIFF"
        rec = dict(id="m1", labeler="a", species=["brdowl"], certainty="sure", none=False, unsure=False, tags=[], edge=False,
                   discuss=False, notes="", ms=1200, plays=1)
        req = urllib.request.Request(f"{base}/api/label?k={st.key}", json.dumps(rec).encode(), {"Content-Type": "application/json"})
        assert json.load(urllib.request.urlopen(req))["ok"]
        got = json.load(urllib.request.urlopen(f"{base}/api/labels?k={st.key}&labeler=a"))
        assert got["m1"]["species"] == ["brdowl"]
    finally:
        srv.shutdown()
    df = load_labels(tmp_path)
    assert list(df.is_brdowl) == [True] and df.weight.iloc[0] == 100 and not df.unblinded.any()
    assert weighted_rate(df, "is_brdowl", "stratum").iloc[0] == 1.0
