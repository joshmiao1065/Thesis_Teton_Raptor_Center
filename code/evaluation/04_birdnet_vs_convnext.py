# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python (thesis)
#     language: python
#     name: thesis
# ---

# %% [markdown]
# # 04 BirdNET versus ConvNeXT on Murie
#
# BirdNET v2.4 was scored on all Murie files with `evaluation/run_birdnet.py` (3 s windows, no
# overlap, sigmoid outputs; five target probabilities and the top-5 over all 6,522 classes are
# saved). ConvNeXT was scored with 5 s windows and a 1 s hop. Questions:
# 1. **Rule asymmetry.** The ConvNeXT baseline takes the top-1 over all 9,736 classes; the BirdNET run
#    took the top-1 among the five targets only. How much of the difference in counts is the rule?
# 2. **Independent evidence.** Do BirdNET's target probabilities agree with the ConvNeXT detections
#    (especially the boreal and goshawk ones we suspect are false)?
# 3. **Real calls.** On the manually confirmed barred-owl calls, which model, at the same amount of
#    flagged audio time per hour, finds more of them?
#
# Windows differ in length and hop, so rates are compared as **seconds of flagged audio per hour**
# (union of the flagged windows' time spans), not as windows per hour.

# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_processing.paths import INTERMEDIATE, KALEIDOSCOPE
from training.rescorer import CODES, DAYS, Rescorer, gate, load_region

plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
BN = INTERMEDIATE / "birdnet_murie"
tclass = json.loads((BN / "_targets.json").read_text())["class_idx"]  # BirdNET class ids of CODES, same order

files, starts, P_bn, top1_bn, top1p_bn = [], [], [], [], []
for f in sorted(BN.glob("*.npz")):
    z = np.load(f)
    n = len(z["start_s"])
    files += [f.stem.split("__")[-1] + ".wav"] * n
    starts.append(z["start_s"]); P_bn.append(z["target_p"]); top1_bn.append(z["top_idx"][:, 0]); top1p_bn.append(z["top_p"][:, 0])
bn = pd.DataFrame(np.concatenate(P_bn), columns=[f"p_{c}" for c in CODES])
bn.insert(0, "start_s", np.concatenate(starts)); bn.insert(0, "file", files)
bn["top1_all"] = np.concatenate(top1_bn); bn["top1_all_p"] = np.concatenate(top1p_bn)
print(f"BirdNET: {bn.file.nunique()} files, {len(bn):,} windows of 3 s ({len(bn) * 3 / 3600:.1f} h)")

cx = load_region("murie")  # ConvNeXT logits, 1 s hop
Pc = 1 / (1 + np.exp(-cx["tlogit"]))
print(f"ConvNeXT: {len(set(cx['file']))} files, {len(Pc):,} windows")

# %% [markdown]
# ## 1. Counts under the two rules (windows with p > 0.1)
# *Top-1 over all classes* is the ConvNeXT baseline rule. *Top-1 among the five* takes the largest
# of the five target probabilities. Applying both rules to both models separates rule from model.

# %%
T = 0.1
def count(flag_species):
    return pd.Series(flag_species).value_counts().reindex(CODES).fillna(0).astype(int)

cx_all = np.where((Pc >= cx["maxnon"][:, None]).any(1), Pc.argmax(1), -1)                   # target is the overall top-1
cx_all = np.where((Pc.max(1) > T) & (cx_all >= 0), cx_all, -1)
cx_five = np.where(Pc.max(1) > T, Pc.argmax(1), -1)                                         # top-1 among the five
bn_all = np.array([tclass.index(c) if c in tclass else -1 for c in bn.top1_all])
bn_all = np.where((bn.top1_all_p.to_numpy() > T) & (bn_all >= 0), bn_all, -1)
Pb = bn[[f"p_{c}" for c in CODES]].to_numpy()
bn_five = np.where(Pb.max(1) > T, Pb.argmax(1), -1)

hrs_cx, hrs_bn = len(Pc) / 3600, len(Pb) * 3 / 3600
tab = pd.DataFrame({
    "ConvNeXT top-1 over all classes (baseline)": count([CODES[i] for i in cx_all if i >= 0]) / hrs_cx,
    "ConvNeXT top-1 among five": count([CODES[i] for i in cx_five if i >= 0]) / hrs_cx,
    "BirdNET top-1 over all classes": count([CODES[i] for i in bn_all if i >= 0]) / hrs_bn,
    "BirdNET top-1 among five": count([CODES[i] for i in bn_five if i >= 0]) / hrs_bn,
})
print("flagged windows per hour of audio (note: ConvNeXT windows overlap 4/5, BirdNET's do not)")
tab.round(2)

# %%
ax = tab.plot.bar(figsize=(11, 3.8), width=0.85)
ax.set_ylabel("flagged windows / hour"); ax.set_yscale("log"); ax.legend(fontsize=7)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 2. Do the two models agree?
# For each ConvNeXT baseline detection (5 s window) we take the maximum BirdNET probability of the same
# species over the overlapping 3 s windows, and the reverse.

# %%
def overlap_max(src_files, src_starts, src_dur, tgt_df_files, tgt_starts, tgt_dur, tgt_p):
    """max of tgt_p over target windows that overlap each source window; tgt arrays sorted by (file, start)."""
    out = np.zeros(len(src_files), np.float32)
    by = {f: (tgt_starts[m], tgt_p[m]) for f in set(src_files) for m in [tgt_df_files == f]}
    for i, (f, s) in enumerate(zip(src_files, src_starts)):
        ts, tp = by[f]
        sel = (ts < s + src_dur) & (ts + tgt_dur > s)
        out[i] = tp[sel].max() if sel.any() else 0
    return out

bn_f, bn_s = bn.file.to_numpy(), bn.start_s.to_numpy()
fig, axs = plt.subplots(2, 5, figsize=(17, 5), sharey="row")
rows = []
for k, c in enumerate(CODES):
    m = cx_all == k
    ff, ss = cx["file"][m], cx["start_s"][m]
    pb_at = overlap_max(ff, ss, 5, bn_f, bn_s, 3, Pb[:, k]) if m.any() else np.array([])
    axs[0, k].hist(pb_at, bins=np.linspace(0, 1, 21), color="tab:blue")
    axs[0, k].set_title(f"{c}: BirdNET p at {m.sum()} ConvNeXT detections", fontsize=8)
    mb = bn_five == k
    pc_at = overlap_max(bn_f[mb], bn_s[mb], 3, cx["file"], cx["start_s"], 5, Pc[:, k]) if mb.any() else np.array([])
    axs[1, k].hist(pc_at, bins=np.linspace(0, 1, 21), color="tab:orange")
    axs[1, k].set_title(f"{c}: ConvNeXT p at {mb.sum()} BirdNET detections", fontsize=8)
    rows.append(dict(species=c, convnext_dets=int(m.sum()), bn_p_gt_0_1=float((pb_at > 0.1).mean()) if m.any() else np.nan,
                     bn_median_p=float(np.median(pb_at)) if m.any() else np.nan,
                     birdnet_dets=int(mb.sum()), cx_p_gt_0_1=float((pc_at > 0.1).mean()) if mb.any() else np.nan))
plt.tight_layout(); plt.show()
pd.DataFrame(rows).round(3)

# %% [markdown]
# ## 3. Manually confirmed barred-owl calls (35 on held-out days)
# Flagged time per hour is measured on the held-out days (0410 to 0415). Barred owls call often there,
# so the low-rate end of the curve is limited by real calls in the "negative" audio (as in the
# earlier notebooks). Heads use the multi-site rescorer (`m_all_raw`), with and without the gate.

# %%
held = DAYS["val"] + DAYS["test"]
kc = pd.read_csv(next(KALEIDOSCOPE.glob("Murie*/cluster.csv")))
calls = kc[(kc["MANUAL ID"] == "Barred Owl")]
calls = calls[calls["IN FILE"].str[:4].isin(held)]
print(len(calls), "manual barred calls on held-out days")

# per-file length (s) from the ConvNeXT windows (n windows + 4)
cx_file = cx["file"]
flen = pd.Series(cx["start_s"]).groupby(cx_file).max() + 5
hfiles = sorted(f for f in flen.index if f[:4] in held)
base = dict(zip(hfiles, np.r_[0, np.cumsum(np.ceil(flen[hfiles].to_numpy()))[:-1]].astype(int)))
total_s = int(np.ceil(flen[hfiles]).sum())

def mask_from_windows(f_arr, s_arr, dur, flag):
    """boolean per second of the concatenated held-out audio: covered by a flagged window."""
    d = np.zeros(total_s + 2, np.int32)
    for f, s in zip(f_arr[flag], s_arr[flag]):
        b = base.get(f)
        if b is None:
            continue
        lo = b + int(s); hi = min(b + int(np.ceil(s + dur)), b + int(np.ceil(flen[f])))
        d[lo] += 1; d[hi] -= 1
    return np.cumsum(d)[:total_s] > 0

def curve(f_arr, s_arr, dur, score, rates):
    """for each target flagged-seconds-per-hour, bisect a threshold on score (held-out days); recall on calls."""
    hm = np.isin([f[:4] for f in f_arr], held)
    f_h, s_h, sc_h = f_arr[hm], s_arr[hm], score[hm]
    out = []
    for R in rates:
        lo, hi = 0.0, 1.0
        for _ in range(22):
            mid = (lo + hi) / 2
            r = mask_from_windows(f_h, s_h, dur, sc_h > mid).sum() / (total_s / 3600)
            lo, hi = (mid, hi) if r > R else (lo, mid)
        thr = hi
        mk = mask_from_windows(f_h, s_h, dur, sc_h > thr)
        hit = 0
        for f, off, du in zip(calls["IN FILE"], calls["OFFSET"], calls["DURATION"]):
            b = base[f]; hit += mk[b + int(off): b + int(np.ceil(off + du)) + 1].any()
        out.append(hit / len(calls))
    return out

RATES = [1, 2, 3, 5, 7, 10, 15, 20, 40, 80, 160]  # flagged seconds per hour
k = CODES.index("brdowl")
results = {
    "BirdNET p": curve(bn_f, bn_s, 3, Pb[:, k], RATES),
    "ConvNeXT p": curve(cx["file"], cx["start_s"], 5, Pc[:, k], RATES),
}
for hn in ("hard_lin", "hard_mlp"):
    h = Rescorer.load(INTERMEDIATE / "rescorer" / "m_all_raw" / f"{hn}.pt")
    S = h.predict(cx["emb"])
    results[f"ConvNeXT head {hn}"] = curve(cx["file"], cx["start_s"], 5, S[:, k], RATES)
    results[f"ConvNeXT head {hn} + gate"] = curve(cx["file"], cx["start_s"], 5, gate(S, cx["tlogit"], 0.001)[:, k], RATES)
res = pd.DataFrame(results, index=RATES).rename_axis("flagged s/h")
ax = res.plot(marker="o", figsize=(8, 4.5), logx=True)
ax.set_ylabel(f"recall of the {len(calls)} manual barred calls"); ax.set_ylim(0, 1.02); plt.show()
res.round(2)

# %% [markdown]
# ## Reading the results
# **What the executed numbers say (Murie, 193 h).**
# * **The rule is not the explanation.** Restricting ConvNeXT to the top-1 among the five targets does
#   not lower its boreal rate (7.4 to 11.7 flagged windows/h); BirdNET flags 0.2/h under either rule.
#   The boreal gap between the models is a model difference, roughly 12x or more even allowing for
#   ConvNeXT's overlapping windows.
# * **Agreement.** Barred owl: 92% of ConvNeXT's barred windows have BirdNET p > 0.1 (median 0.85).
#   Boreal: only 7% (median 0.001), and goshawk 27%. In the other direction, 68% of BirdNET's
#   boreal and 67% of its goshawk windows are also ConvNeXT detections, so BirdNET's few boreal and
#   goshawk detections are a subset of ConvNeXT's, not the reverse. BirdNET being insensitive to
#   boreal owls is an alternative reading, so this is supporting evidence, not proof.
# * **BirdNET's own barred windows.** 60% of its 1,326 barred windows (p > 0.1) have ConvNeXT
#   p < 0.1. Whether those are real calls or BirdNET false positives cannot be settled without labels.
# * **Recall on the 35 confirmed calls.** At equal flagged time, BirdNET reaches recall 0.37 at 1 s/h,
#   0.83 at 5 s/h and 1.0 at 10 s/h; ConvNeXT probability 0.14, 0.57 and 0.94; the Murie-trained
#   heads are no better than ConvNeXT's probability here (the gate makes no difference for barred).
# * **Caveat.** ConvNeXT windows are 5 s and BirdNET's 3 s, so one call costs ConvNeXT at least 5 s of
#   flagged time and BirdNET 3 s: part of BirdNET's lead is granularity (a shift of the ConvNeXT
#   curves by at most a factor 5/3 along the x axis). Real barred calls inside the "negative" audio
#   inflate every rate equally. There are 35 calls, so one call is 0.03.
