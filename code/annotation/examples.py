"""Browsable example clips with their model scores (not blinded), for getting to know typical detections.

    python -m annotation.examples select --dir annotation [--species borowl norgos]
    python -m annotation.examples finish --dir annotation

`select` draws ConvNeXT baseline events of the given species in the dawn and dusk hours, one per
confidence bin and file, skipping anything that overlaps a queue clip (so the blind queue stays blind).
`finish` writes the group "examples" into manifest.json, refreshes scores.json and prints both models' logits.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from annotation.sampling import PAD, Site, cx_events
from training.rescorer import CODES

BINS = [(0.6, 1.01), (0.35, 0.6), (0.2, 0.35), (0.1, 0.2)]  # ConvNeXT probability bins, high to low
HOURS = [(5, 8), (18, 22)]  # dawn and dusk, where the boreal and goshawk detections concentrate
NAMES = {"borowl": "Boreal owl", "norgos": "Northern goshawk", "brdowl": "Barred owl", "grgowl": "Great gray owl", "flaowl": "Flammulated owl"}
SHORT = {"flaowl": "flam", "grgowl": "gg", "norgos": "gos", "brdowl": "barred", "borowl": "boreal"}


def select(folder, species=("borowl", "norgos"), seed=0, per_bin=1):
    folder = Path(folder)
    man = json.loads((folder / "manifest.json").read_text())
    s = Site("murie")
    ev = cx_events(s)
    ev["hour"] = [s.hours(f, t) for f, t in zip(ev.fid, ev.start)]
    ev["ok"] = [any(h0 <= h < h1 for h0, h1 in HOURS) for h in ev.hour]
    busy = {}
    for it in man["queue"]:
        if it["site"] == "murie":
            busy.setdefault(s.rels.index(it["rel"]), []).append((it["clip"][0] - 5, it["clip"][0] + it["clip"][1] + 5))
    ev["ok"] &= [not any(a < hi and a + 5 > lo for lo, hi in busy.get(f, [])) for f, a in zip(ev.fid, ev.start)]  # overlaps a queue clip (padded 5 s)
    rng = np.random.default_rng(seed)
    items = []
    for code in species:
        k, used = CODES.index(code), set()
        for lo, hi in BINS:
            c = ev[(ev.k == k) & ev.ok & (ev.p >= lo) & (ev.p < hi)]
            c = c.sample(frac=1, random_state=int(rng.integers(1 << 30)))
            got = 0
            for r in c.itertuples():
                if got >= per_bin:
                    break
                if r.fid in used:
                    continue
                used.add(r.fid)
                rel, a = s.rels[r.fid], float(r.start)
                c0 = max(a - PAD, 0)
                hh, mm = int(r.hour), int((r.hour % 1) * 60)
                iid = "e" + hashlib.md5(f"{rel}|{a}|{code}".encode()).hexdigest()[:8]
                items.append(dict(id=iid, kind="example", site="murie", rel=rel, clip=[round(c0, 2), 5 + PAD + (a - c0)],
                                  focal=[round(a - c0, 2), round(a - c0 + 5, 2)], species=code, truth=[], a=a, p=float(r.p), k=k,
                                  title=f"{NAMES[code]}: ConvNeXT p={r.p:.2f}, {hh:02d}:{mm:02d}, {Path(rel).stem} at {a:.0f} s"))
                got += 1
    (folder / "examples_sel.json").write_text(json.dumps(items))
    print(len(items), "examples selected")


def finish(folder):
    """Write the examples into manifest.json, recompute scores.json for all clips and print the logits."""
    from annotation.scores import compute_scores

    folder = Path(folder)
    items = json.loads((folder / "examples_sel.json").read_text())
    man = json.loads((folder / "manifest.json").read_text())
    man["examples"] = [{k: v for k, v in it.items() if k not in ("a", "k", "p")} for it in items]
    (folder / "manifest.json").write_text(json.dumps(man))
    scores = compute_scores(folder)
    rows = []
    for it in items:
        sc = scores[it["id"]]
        c = max(sc["cx"], key=lambda w: w["logits"][it["species"]])
        row = dict(id=it["id"], sample=f"{NAMES[it['species']]} p={it['p']:.2f}", file=Path(it["rel"]).stem, at_s=int(it["a"]))
        row.update({f"cx_{SHORT[k]}": v for k, v in c["logits"].items()})
        row["cx_top1"] = "{} {}".format(*c["top"][0])
        bmax = {k: max(w["logits"][k] for w in sc["bn"]) for k in CODES}
        row.update({f"bn_{SHORT[k]}": v for k, v in bmax.items()})
        row["bn_top1"] = "{} {}".format(*max((t for w in sc["bn"] for t in w["top"]), key=lambda t: t[1]))
        rows.append(row)
    df = pd.DataFrame(rows)
    out = folder.parent / "results" / "lookup"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "examples_logits.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(df.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "finish"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--species", nargs="+", default=["borowl", "norgos"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    select(a.dir, a.species, a.seed) if a.step == "select" else finish(a.dir)
