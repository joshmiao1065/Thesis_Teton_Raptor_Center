"""Choose which audio windows a human should label, and write the queue.

Each item is a short focal window (the window a model flagged, or a random one) plus a few seconds
of context. Items are drawn from strata that answer the thesis questions:

* ``cx_*``   ConvNeXT baseline detections, by species and confidence bin (precision of the baseline)
* ``bn_*``   BirdNET detections, split into agreeing with ConvNeXT or not (who is right when they differ)
* ``hd_*``   windows the rescorer heads flag but neither model does (do the heads find real calls?)
* ``nt_*``   ConvNeXT windows just below the 0.1 threshold (what the threshold misses)
* ``rnd_*``  random windows nobody flagged, by time of day (how many calls no method finds)
* ``ctl_*``  human-confirmed calls (Kaleidoscope manual IDs), as positive controls

The annotator never sees the stratum or any score. ``manifest_full.json`` keeps everything
(stratum, population size, model scores at the focal window) for the analysis afterwards.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_processing.paths import INTERMEDIATE, KALEIDOSCOPE, MURIE_AUDIO, REGION_AUDIO
from training.rescorer import CODES, Rescorer, gate

PAD = 3.5  # seconds of context before and after the focal window
HEAD = "m_all_raw/hard_mlp"  # rescorer head used to look for calls the models miss
GATE = 0.001
HOUR_BINS = {"night": (22, 4), "dawn": (4, 8), "day": (8, 18), "dusk": (18, 22)}


def audio_root(site):
    return Path(MURIE_AUDIO) if site == "murie" else Path(REGION_AUDIO[site])


def item_id(site, rel, a, d):
    return "m" + hashlib.md5(f"{site}|{rel}|{a:.1f}|{d:.1f}".encode()).hexdigest()[:8]


class Site:
    """ConvNeXT windows (1 s hop), BirdNET windows (3 s hop) and head scores of one recording site."""

    def __init__(self, name):
        self.name = name
        fdir = INTERMEDIATE / "features" / name
        self.rels, fid, start, P, maxnon, emb, tl = [], [], [], [], [], [], []
        for i, f in enumerate(sorted(fdir.glob("*.npz"))):
            z = np.load(f)
            self.rels.append(f.stem.replace("__", "/") + ".wav")
            n = len(z["start_s"])
            fid.append(np.full(n, i, np.int32))
            start.append(z["start_s"].astype(np.float32))
            P.append(1 / (1 + np.exp(-z["tlogit"])))
            maxnon.append(z["maxnon"])
            tl.append(z["tlogit"])
            emb.append(z["emb"])
        self.fid, self.start = np.concatenate(fid), np.concatenate(start)
        self.P, self.maxnon = np.concatenate(P), np.concatenate(maxnon)
        self.off = np.r_[0, np.cumsum([len(s) for s in start])]
        self.flen = np.array([s.max() + 5 for s in start])
        tag, name_ = HEAD.split("/")
        head = Rescorer.load(INTERMEDIATE / "rescorer" / tag / f"{name_}.pt")
        self.H = gate(head.predict(np.concatenate(emb)), np.concatenate(tl), GATE)
        self._load_birdnet()

    def _load_birdnet(self):
        bdir = INTERMEDIATE / f"birdnet_{self.name}"
        ix = {r: i for i, r in enumerate(self.rels)}
        fid, start, p = [], [], []
        for f in sorted(bdir.glob("*.npz")):
            rel = f.stem.replace("__", "/") + ".wav"
            if rel not in ix:
                continue
            z = np.load(f)
            fid.append(np.full(len(z["start_s"]), ix[rel], np.int32))
            start.append(z["start_s"]); p.append(z["target_p"])
        self.bn_fid, self.bn_start, self.bn_P = np.concatenate(fid), np.concatenate(start), np.concatenate(p)
        o = np.argsort(self.bn_fid, kind="stable")
        self.bn_fid, self.bn_start, self.bn_P = self.bn_fid[o], self.bn_start[o], self.bn_P[o]
        self.bn_off = np.searchsorted(self.bn_fid, np.arange(len(self.rels) + 1))

    def hours(self, fid, t):
        """Local clock hour of second t in file fid, from the file name MMDD_HHMMSS."""
        name = Path(self.rels[fid]).stem
        hh, mm, ss = int(name[5:7]), int(name[7:9]), int(name[9:11])
        return ((hh * 3600 + mm * 60 + ss + t) / 3600) % 24

    def context(self, fid, a, d):
        """Model scores (max over the windows overlapping [a, a + d]), stored for the analysis only."""
        lo, hi = self.off[fid], self.off[fid + 1]
        st = self.start[lo:hi]
        m = (st < a + d) & (st + 5 > a)
        cx = self.P[lo:hi][m].max(0) if m.any() else np.zeros(5)
        hd = self.H[lo:hi][m].max(0) if m.any() else np.zeros(5)
        blo, bhi = self.bn_off[fid], self.bn_off[fid + 1]
        bs = self.bn_start[blo:bhi]
        mb = (bs < a + d) & (bs + 3 > a)
        bn = self.bn_P[blo:bhi][mb].max(0) if mb.any() else np.zeros(5)
        return dict(cx_p=cx.round(4).tolist(), bn_p=bn.round(4).tolist(), hd_p=hd.round(4).tolist(),
                    hour=round(float(self.hours(fid, a)), 2))


def merge_events(d, hop):
    """Consecutive windows of the same file and species become one event; focal = its best window."""
    d = d.sort_values(["fid", "k", "start"]).reset_index(drop=True)
    new = (d.fid != d.fid.shift()) | (d.k != d.k.shift()) | (d.start != d.start.shift() + hop)
    d["ev"] = new.cumsum()
    best = d.loc[d.groupby("ev").p.idxmax()].drop(columns="ev").reset_index(drop=True)
    best["n"] = d.groupby("ev").size().values
    return best


def cx_events(s, thr=0.1):
    """Baseline rule: the top-1 over all classes is a target and p > thr; merged into events."""
    top, k = s.P.max(1), s.P.argmax(1)
    flag = (top > thr) & (top >= s.maxnon)
    ev = merge_events(pd.DataFrame(dict(fid=s.fid[flag], k=k[flag], start=s.start[flag], p=top[flag])), 1)
    ev["dur"] = 5.0
    return ev


def bn_events(s, thr=0.1):
    """BirdNET rule used in the earlier runs: largest of the five target probabilities, p >= thr."""
    top, k = s.bn_P.max(1), s.bn_P.argmax(1)
    flag = top >= thr
    ev = merge_events(pd.DataFrame(dict(fid=s.bn_fid[flag], k=k[flag], start=s.bn_start[flag], p=top[flag])), 3)
    ev["dur"] = 3.0
    return ev


class Near:
    """Is there a flagged time close to (fid, t)? Built from arrays of file ids and times."""

    def __init__(self, fid, t):
        o = np.lexsort((t, fid))
        self.fid, self.t = np.asarray(fid)[o], np.asarray(t)[o]

    def __call__(self, fid, t, w):
        lo = np.searchsorted(self.fid, fid, "left")
        hi = np.searchsorted(self.fid, fid, "right")
        tt = self.t[lo:hi]
        j = np.searchsorted(tt, t - w, "left")
        return bool(j < len(tt) and tt[j] <= t + w)


class Picker:
    """Greedy sampling without overlaps: takes candidates in random order until n are chosen."""

    def __init__(self, site, rng):
        self.site, self.rng, self.rows, self.taken = site, rng, [], {}

    def free(self, fid, a, d, gap=1.0):
        c = a + d / 2
        return all(abs(c - c2) >= (d + d2) / 2 + gap for c2, d2 in self.taken.get(fid, []))

    def take(self, cands, n, stratum, **extra):
        pop = len(cands)
        got = 0
        for r in cands.sample(frac=1, random_state=int(self.rng.integers(1 << 30))).itertuples():
            if got >= n:
                break
            if not self.free(r.fid, r.focal, r.dur):
                continue
            self.taken.setdefault(r.fid, []).append((r.focal + r.dur / 2, r.dur))
            self.rows.append(dict(site=self.site.name, fid=int(r.fid), focal=float(r.focal), dur=float(r.dur),
                                  stratum=stratum, pop=pop, **extra))
            got += 1
        return got


def thin(c, gap):
    """Keep the best-scoring candidate of each cluster: no two kept windows of a file within gap seconds."""
    kept, seen = [], {}
    for r in c.itertuples():
        if all(abs(r.focal - t) >= gap for t in seen.get(r.fid, [])):
            seen.setdefault(r.fid, []).append(r.focal)
            kept.append(r.Index)
    return c.loc[kept]


def bin_edges(p, edges):
    return pd.cut(p, edges, right=False, labels=False)


def build_site(s, rng, alloc, control_rows=None):
    """Sample the strata of one site. alloc: dict of stratum-kind -> sizes (see ALLOC)."""
    pk = Picker(s, rng)
    if control_rows is not None and len(control_rows):
        for sp, g in control_rows.groupby("species"):
            g = g.assign(fid=g.fid.astype(int))
            pk.take(g, alloc.get("ctl", 12), f"ctl_{sp}")
    cx, bn = cx_events(s), bn_events(s)
    cx = cx.assign(focal=cx.start)
    bn = bn.assign(focal=bn.start)
    cx_near = {k: Near(cx.fid[cx.k == k], cx.start[cx.k == k]) for k in range(5)}
    for k, code in enumerate(CODES):
        e = cx[cx.k == k]
        if len(e) == 0:
            continue
        edges = alloc["cx_bins"][code]
        e = e.assign(b=bin_edges(e.p, edges))
        for b, g in e.groupby("b"):
            pk.take(g, alloc["cx_per_bin"][code], f"cx_{code}_p{edges[int(b)]:.2f}-{min(edges[int(b) + 1], 1):.2f}")
    for k, code in enumerate(CODES):
        e = bn[bn.k == k]
        if len(e) == 0:
            continue
        agree = np.array([cx_near[k](r.fid, r.start, 4) for r in e.itertuples()])
        pk.take(e[agree], alloc["bn_agree"][code], f"bn_{code}_agree")
        only = e[~agree]
        edges = alloc["bn_bins"]
        for b, g in only.assign(b=bin_edges(only.p, edges)).groupby("b"):
            pk.take(g, alloc["bn_only_per_bin"][code], f"bn_{code}_only_p{edges[int(b)]:.2f}-{min(edges[int(b) + 1], 1):.2f}")
    flagged = Near(np.r_[cx.fid, bn.fid], np.r_[cx.start, bn.start])
    for k, code in enumerate(CODES):  # heads: top-scoring windows nobody else flags
        n_top = 300
        top = np.argsort(-s.H[:, k])[: n_top * 8]
        c = pd.DataFrame(dict(fid=s.fid[top], focal=s.start[top], dur=5.0, sc=s.H[top, k]))
        c = thin(c[[not flagged(r.fid, r.focal, 6) for r in c.itertuples()]], 6).head(n_top)
        pk.take(c, alloc["hd"][code], f"hd_{code}")
    for k, code in enumerate(CODES):  # just below the threshold
        m = (s.P[:, k] >= 0.03) & (s.P[:, k] < 0.1)
        c = pd.DataFrame(dict(fid=s.fid[m], focal=s.start[m], dur=5.0))
        c = c[[not flagged(r.fid, r.focal, 6) for r in c.itertuples()]]
        pk.take(c, alloc["nt"][code], f"nt_{code}")
    head_hi = np.array([np.quantile(s.H[:, k], 0.995) for k in range(5)])
    quiet = (s.P.max(1) < 0.03) & (s.H <= head_hi).all(1)
    ts = s.start
    for hb, (h0, h1) in HOUR_BINS.items():
        hrs = np.array([s.hours(f, t) for f, t in zip(s.fid[quiet][::7], ts[quiet][::7])])
        idx = np.where(quiet)[0][::7]
        m = (hrs >= h0) & (hrs < h1) if h0 < h1 else (hrs >= h0) | (hrs < h1)
        c = pd.DataFrame(dict(fid=s.fid[idx[m]], focal=ts[idx[m]], dur=5.0))
        c = c[[not flagged(r.fid, r.focal, 8) for r in c.itertuples()]]
        pk.take(c, alloc["rnd"][hb], f"rnd_{hb}")
    return pd.DataFrame(pk.rows)


# sample sizes: Murie is the main site, the others only need enough to check transfer
ALLOC = {
    "murie": dict(
        cx_bins={"borowl": [0.1, 0.2, 0.35, 0.6, 1.01], "norgos": [0.1, 0.2, 0.35, 0.6, 1.01],
                 "brdowl": [0.1, 0.5, 0.9, 0.98, 1.01], "flaowl": [0.1, 1.01], "grgowl": [0.1, 1.01]},
        cx_per_bin={"borowl": 30, "norgos": 20, "brdowl": 20, "flaowl": 15, "grgowl": 5},
        bn_bins=[0.1, 0.2, 0.5, 1.01],
        bn_agree={"borowl": 8, "norgos": 10, "brdowl": 30, "flaowl": 10, "grgowl": 5},
        bn_only_per_bin={"borowl": 10, "norgos": 10, "brdowl": 30, "flaowl": 12, "grgowl": 6},
        hd={"borowl": 30, "norgos": 30, "brdowl": 30, "flaowl": 30, "grgowl": 30},
        nt={"borowl": 30, "norgos": 30, "brdowl": 20, "flaowl": 20, "grgowl": 15},
        rnd={"night": 40, "dawn": 40, "day": 30, "dusk": 40}, ctl=12),
    "other": dict(
        cx_bins={c: [0.1, 0.4, 1.01] for c in CODES}, cx_per_bin={c: 10 for c in CODES},
        bn_bins=[0.1, 0.5, 1.01],
        bn_agree={c: 6 for c in CODES}, bn_only_per_bin={c: 8 for c in CODES},
        hd={c: 12 for c in CODES}, nt={c: 5 for c in CODES},
        rnd={"night": 8, "dawn": 8, "day": 8, "dusk": 8}, ctl=8),
}


def confirmed_calls(site, s):
    """Human-confirmed calls of the site's species (Kaleidoscope manual IDs whose file is on disk)."""
    key = {"murie": ("Murie*", "Barred Owl", "brdowl"), "taylor": ("Taylor*", "NOGO", "norgos"),
           "riverview": ("Riverview*", "GGOW", "grgowl")}.get(site)
    if key is None:
        return pd.DataFrame()
    pat, prefix, code = key
    kc = pd.read_csv(next(KALEIDOSCOPE.glob(pat + "/cluster.csv")))
    col = next(c for c in kc.columns if c.startswith("MANUAL"))
    kc = kc[kc[col].astype(str).str.startswith(prefix)]
    names = {Path(r).name: i for i, r in enumerate(s.rels)}
    rows = []
    for f, off, dur in zip(kc["IN FILE"], kc["OFFSET"], kc["DURATION"]):
        if f in names:
            fid = names[f]
            a = float(np.clip(off + dur / 2 - 2.5, 0, max(s.flen[fid] - 5, 0)))
            rows.append(dict(fid=fid, focal=a, dur=5.0, species=code))
    return pd.DataFrame(rows)


def to_items(df, s):
    items = []
    for r in df.itertuples():
        a, d = r.focal, r.dur
        c0 = max(a - PAD, 0)
        c1 = min(a + d + PAD, float(s.flen[r.fid]))
        rel = s.rels[r.fid]
        items.append(dict(
            id=item_id(s.name, rel, a, d), kind="main", site=s.name, rel=rel,
            clip=[round(c0, 2), round(c1 - c0, 2)], focal=[round(a - c0, 2), round(a - c0 + d, 2)],
            stratum=r.stratum, pop=int(r.pop), hidden=s.context(r.fid, a, d)))
    return items


def order_queue(items, rng, dup_frac=0.05):
    """Proportional interleaving of the strata (any prefix is a stratified sample), then hidden repeats."""
    by = {}
    for it in items:
        by.setdefault(it["stratum"], []).append(it)
    keyed = []
    for g in by.values():
        for r, it in enumerate(rng.permutation(len(g))):
            keyed.append(((r + rng.random()) / len(g), g[it]))
    keyed = [(rng.random() if it["stratum"].startswith("ctl_") else u, it) for u, it in keyed]  # controls: spread out
    queue = [it for _, it in sorted(keyed, key=lambda x: x[0])]
    orig = list(queue)
    n_dup = min(int(len(orig) * dup_frac), max(len(orig) // 2 - 20, 0))
    for j in rng.choice(np.arange(20, max(len(orig) // 2, 21)), size=n_dup, replace=False):
        src = dict(orig[j], id=orig[j]["id"] + "d", dup_of=orig[j]["id"], stratum="dup")
        queue.insert(int(min(len(queue), j + rng.integers(60, 240))), src)
    for i, it in enumerate(queue):
        it["tier"] = 1 if i < 300 else (2 if i < 800 else 3)
    return queue


def build_queue(sites, out_dir, seed=0):
    """Sample every site, order the queue, write manifest_full.json (with a queue file for the app)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    items = []
    for name in sites:
        s = Site(name)
        ctl = confirmed_calls(name, s)
        alloc = ALLOC["murie"] if name == "murie" else ALLOC["other"]
        df = build_site(s, np.random.default_rng([seed, sum(map(ord, name))]), alloc, ctl if len(ctl) else None)
        items += to_items(df, s)
        print(name, len(df), "items;", df.stratum.str.split("_").str[0].value_counts().to_dict(), flush=True)
    queue = order_queue(items, rng)
    (out / "queue_full.json").write_text(json.dumps(queue))
    return queue
