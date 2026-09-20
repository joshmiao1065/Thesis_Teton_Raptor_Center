"""Reference library and practice set for the annotation tool.

Reference items always show their answer: human-confirmed field calls (Kaleidoscope manual IDs whose
audio is on disk) and, for species without confirmed field calls, recordings from the benchmark's
source audio (written as 8 kHz wav files next to the queue). Practice items are labelled first and
then reveal the answer. Field calls used as controls in the main queue are excluded here.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from annotation.sampling import PAD, audio_root
from data_processing.mixtures import load_calls, mix, read_bg
from data_processing.paths import KALEIDOSCOPE

FIELD = [  # (site, Kaleidoscope folder pattern, manual-ID prefix, species code)
    ("murie", "Murie*", "Barred Owl", "brdowl"),
    ("taylor", "Taylor*", "NOGO", "norgos"),
    ("riverview", "Riverview*", "GGOW", "grgowl"),
]
NAMES = {"brdowl": "Barred owl", "borowl": "Boreal owl", "grgowl": "Great gray owl",
         "flaowl": "Flammulated owl", "norgos": "Northern goshawk"}


def field_calls(site, pat, prefix, code):
    kc = pd.read_csv(next(KALEIDOSCOPE.glob(pat + "/cluster.csv")))
    col = next(c for c in kc.columns if c.startswith("MANUAL"))
    kc = kc[kc[col].astype(str).str.startswith(prefix)]
    root = audio_root(site)
    rels = {p.name: p.relative_to(root).as_posix() for p in root.rglob("*.wav")}
    rows = [dict(site=site, rel=rels[f], off=float(o), dur=float(d), sub=str(m), code=code)
            for f, o, d, m in zip(kc["IN FILE"], kc["OFFSET"], kc["DURATION"], kc[col]) if f in rels]
    return pd.DataFrame(rows)


def spread(df, n):
    """n rows spread evenly over the sorted table (diverse call types, times and durations)."""
    df = df.sort_values(["sub", "rel", "off"]).reset_index(drop=True)
    return df.iloc[np.unique(np.linspace(0, len(df) - 1, min(n, len(df))).round().astype(int))]


def field_item(r, kind, k, title):
    c0 = max(r.off - PAD, 0)
    return dict(id=f"{kind[0]}{r.code}{k:02d}", kind=kind, site=r.site, rel=r.rel, clip=[round(c0, 2), round(r.dur + 2 * PAD, 2)],
                focal=[round(r.off - c0, 2), round(r.off - c0 + r.dur, 2)], truth=[r.code], title=title, species=r.code)


def build_reference(queue, out_dir, n_ref=8, n_prac=6):
    out = Path(out_dir)
    (out / "reference").mkdir(parents=True, exist_ok=True)
    used = [(it["site"], it["rel"], it["clip"][0] + it["focal"][0], it["clip"][0] + it["focal"][1])
            for it in queue if it["stratum"].startswith("ctl_")]
    ref, prac = [], []
    for site, pat, prefix, code in FIELD:
        c = field_calls(site, pat, prefix, code)
        if len(c) == 0:
            continue
        free = c[[not any(s == r.site and rel == r.rel and a - 3 <= r.off + r.dur / 2 <= b + 3 for s, rel, a, b in used)
                  for r in c.itertuples()]]
        pick = spread(free, n_ref + n_prac)
        train = pick.iloc[np.linspace(0, len(pick) - 1, n_ref).round().astype(int)] if len(pick) > n_ref else pick
        rest = pick.drop(train.index)
        for k, r in enumerate(train.itertuples()):
            ref.append(field_item(r, "reference", k, f"{NAMES[code]}: confirmed field call ({r.sub}), {r.site}"))
        for k, r in enumerate(rest.itertuples()):
            prac.append(field_item(r, "practice", k, ""))
    calls = load_calls()  # benchmark source recordings (not field audio) for species without confirmed field calls
    for code, items in calls.items():
        for k, (sid, y) in enumerate(items[:6]):
            name = f"{code}_{k}.wav"
            sf.write(out / "reference" / name, y, 8000, subtype="PCM_16")
            it = dict(id=f"s{code}{k:02d}", kind="reference", wav=f"reference/{name}", clip=[0, round(len(y) / 8000, 2)],
                      focal=[0, round(len(y) / 8000, 2)], truth=[code], species=code,
                      title=f"{NAMES[code]}: benchmark source recording {k + 1} (not field audio)")
            if code in ("flaowl", "borowl") and k >= 4:  # species without field calls: the last two are practice
                prac.append(dict(it, id=f"p{code}{k:02d}", kind="practice", title=""))
            else:
                ref.append(it)
    prac += practice_mixtures(queue, out, calls)
    return ref, prac


def practice_mixtures(queue, out, calls, n=4, seed=0):
    """Practice clips for species without confirmed field calls: a benchmark call mixed into a real Murie
    background window (from the random-background stratum) at 2 to 9 dB signal-to-noise ratio."""
    rng = np.random.default_rng(seed)
    bgs = [it for it in queue if it["stratum"].startswith("rnd_") and it["site"] == "murie"]
    items = []
    for code in ("flaowl", "borowl", "grgowl"):
        for k in range(n):
            _, y = calls[code][k % len(calls[code])]
            b = bgs[int(rng.integers(len(bgs)))]
            bg = read_bg(audio_root("murie") / b["rel"], b["clip"][0] + b["focal"][0])
            snr = float(rng.uniform(2, 9))
            name = f"practice_{code}_{k}.wav"
            sf.write(Path(out) / "reference" / name, mix(bg, y, snr, rng), 8000, subtype="PCM_16")
            items.append(dict(id=f"x{code}{k:02d}", kind="practice", wav=f"reference/{name}", clip=[0, 5.0], focal=[0, 5.0],
                              truth=[code], species=code, title=f"a recorded call mixed into real field background ({snr:.0f} dB)"))
    return items
