"""Build a blind audit sheet (audio + spectrogram + label form) from stratified baseline events.

    python analysis/make_audit_sheet.py <events.csv> <out_dir>

Output: <out_dir>/index.html (self-contained labelling page; labels stay in the browser and are
exported as CSV), clips/*.wav, img/*.png, key.csv (true species/confidence; keep away from the
labeller until finished). Some manually confirmed calls are mixed in as hidden controls.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import spectrogram

from data_processing.paths import KALEIDOSCOPE, MURIE_AUDIO as FOREST
CELL = {"borowl": 12, "norgos": 12, "brdowl": 8, "flaowl": 13}
BINS = [(0.1, 0.2), (0.2, 0.5), (0.5, 0.9), (0.9, 1.01)]
PAD = 3


def find(name):
    for p in (FOREST / name, FOREST / "Night Recordings" / name):
        if p.exists():
            return p


def sample(ev, controls):
    rows = []
    for sp, n in CELL.items():
        e = ev[ev.species == sp]
        for lo, hi in BINS:
            b = e[(e.p_max >= lo) & (e.p_max < hi)]
            rows += [dict(file=r.file, start=r.start_s, end=r.end_s, species=sp, p=r.p_max, kind="event")
                     for r in b.sample(min(n, len(b)) if sp != "flaowl" else len(b), random_state=0).itertuples()]
        if sp == "flaowl":
            rows = list({(r["file"], r["start"]): r for r in rows}.values())
    rows += controls
    return pd.DataFrame(rows).drop_duplicates(["file", "start"]).sample(frac=1, random_state=1).reset_index(drop=True)


def render(r, out, i):
    f = find(r.file)
    sr = sf.info(f).samplerate
    a = max(r.start - PAD, 0)
    y, _ = sf.read(f, start=int(a * sr), frames=int((r.end - r.start + 2 * PAD) * sr), dtype="float32")
    sf.write(out / "clips" / f"{i:03d}.wav", y, sr, subtype="PCM_16")
    fr, t, S = spectrogram(y, sr, nperseg=512, noverlap=384)
    S = 10 * np.log10(S + 1e-12)
    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.imshow(S, origin="lower", aspect="auto", extent=[t[0], t[-1], fr[0], fr[-1]], cmap="magma",
              vmin=np.percentile(S, 40), vmax=np.percentile(S, 99.8))
    ax.set_ylim(0, 4000); ax.set_xlabel("s"); ax.set_ylabel("Hz")
    ax.axvline(r.start - a, color="c", lw=0.7); ax.axvline(r.end - a, color="c", lw=0.7)
    fig.tight_layout(); fig.savefig(out / "img" / f"{i:03d}.png", dpi=80); plt.close(fig)


def main(events_csv, out_dir):
    out = Path(out_dir)
    (out / "clips").mkdir(parents=True, exist_ok=True); (out / "img").mkdir(exist_ok=True)
    ev = pd.read_csv(events_csv)
    kc = pd.read_csv(next(KALEIDOSCOPE.glob("Murie*/cluster.csv")))
    bar = kc[kc["MANUAL ID"] == "Barred Owl"].sample(12, random_state=0)
    controls = [dict(file=f, start=int(o), end=int(o + max(d, 1)) + 1, species="control_barred", p=np.nan, kind="control")
                for f, o, d in zip(bar["IN FILE"], bar["OFFSET"], bar["DURATION"])]
    s = sample(ev, controls)
    cards = []
    for i, r in enumerate(s.itertuples()):
        render(r, out, i)
        cards.append(f"""<div class=c id=c{i}><b>#{i:03d}</b> <audio controls preload=none src="clips/{i:03d}.wav"></audio><br>
<img src="img/{i:03d}.png" width=560><br>
<select data-i={i}><option value="">(unlabelled)<option>clear target-type call<option>faint/uncertain target-type call
<option>other animal (bird, frog, insect, mammal)<option>non-biological noise (wind, rain, machine)<option>cannot tell</select>
<select data-w={i}><option value="">which species?<option>barred owl<option>boreal owl<option>great gray owl<option>flammulated owl
<option>northern goshawk<option>other owl<option>other bird<option>frog/insect<option>none</select></div>""")
    page = f"""<!doctype html><meta charset=utf-8><title>Audit</title>
<style>body{{font:14px sans-serif;max-width:640px;margin:1em auto}}.c{{border-bottom:1px solid #ccc;padding:8px 0}}</style>
<h3>Blind audit: {len(s)} clips</h3><p>The cyan lines in each spectrogram mark the flagged interval (audio has 3 s of context on both sides).
Choose what you hear/see. Labels are stored in this browser; press Export when done.</p>
<button onclick="ex()">Export CSV</button>{''.join(cards)}
<script>
const K='audit_labels';let L=JSON.parse(localStorage.getItem(K)||'{{}}');
document.querySelectorAll('select').forEach(s=>{{const k=(s.dataset.i!==undefined?'i':'w')+(s.dataset.i??s.dataset.w);if(L[k])s.value=L[k];
 s.onchange=()=>{{L[k]=s.value;localStorage.setItem(K,JSON.stringify(L))}}}});
function ex(){{let o='clip,quality,species\\n';for(let i=0;i<{len(s)};i++)o+=i+',"'+(L['i'+i]||'')+'","'+(L['w'+i]||'')+'"\\n';
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([o]));a.download='audit_labels.csv';a.click()}}
</script>"""
    (out / "index.html").write_text(page)
    s.insert(0, "clip", range(len(s)))
    s.to_csv(out / "key.csv", index=False)
    print(len(s), "clips;", s.kind.value_counts().to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:3])
