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
# # 05 BirdNET and ConvNeXT at the other sites
#
# Notebook 04 compared the models on Murie. Here BirdNET (3 s windows) is scored on the other sites
# and compared with ConvNeXT (5 s windows, 1 s hop) and the rescorer heads:
# 1. How often does each rule flag windows at each site?
# 2. On the confirmed calls of the site's species (goshawk at Taylor, great gray at Riverview), which
#    method finds more at the same amount of flagged audio? Thresholds are set on **Murie held-out
#    days** (a site where these species are not expected) and transferred, as in the earlier notebooks.
#    Heads are the Murie-only head (`v2`) and the leave-one-site-out multi-site head, both gated.
# 3. Does BirdNET agree that the ConvNeXT detections at Butler North, and the great-gray head's
#    ungated misfires there, are noise?
#
# Rates are seconds of flagged audio per hour; "recall" is the share of confirmed calls overlapped by
# a flagged second.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from data_processing.paths import INTERMEDIATE
from evaluation.birdnet_compare import Timeline, load_birdnet, manual_calls, overlap_max
from training.rescorer import CODES, DAYS, Rescorer, gate, load_region

plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
SITES = ["murie", "taylor", "riverview", "butler"]
cx, bn = {}, {}
for s in SITES:
    cx[s] = load_region(s)
    bn[s] = load_birdnet(f"birdnet_{s}")
    print(f"{s}: ConvNeXT {len(cx[s]['tlogit']):,} windows, BirdNET {len(bn[s]):,} windows of 3 s")
P = {s: 1 / (1 + np.exp(-cx[s]["tlogit"])) for s in SITES}
LEN = {s: pd.Series(cx[s]["start_s"]).groupby(cx[s]["file"]).max().add(5).to_dict() for s in SITES}


def head(tag, site, gated=True):
    h = Rescorer.load(INTERMEDIATE / "rescorer" / tag / "hard_mlp.pt")
    S = h.predict(cx[site]["emb"])
    return gate(S, cx[site]["tlogit"], 0.001) if gated else S

# %% [markdown]
# ## 1. Flagged rate per site and rule (windows per hour of audio, p > 0.1)

# %%
rows = []
for s in SITES:
    hrs_c, hrs_b = len(P[s]) / 3600, len(bn[s]) * 3 / 3600
    top = P[s].max(1)
    base = (top > 0.1) & (top >= cx[s]["maxnon"])
    k_c = P[s].argmax(1)
    Pb = bn[s][[f"p_{c}" for c in CODES]].to_numpy()
    flag_b, k_b = Pb.max(1) >= 0.1, Pb.argmax(1)
    for k, c in enumerate(CODES):
        rows.append(dict(site=s, species=c, convnext_baseline=(base & (k_c == k)).sum() / hrs_c,
                         convnext_among_five=((top > 0.1) & (k_c == k)).sum() / hrs_c,
                         birdnet_among_five=(flag_b & (k_b == k)).sum() / hrs_b))
rates = pd.DataFrame(rows)
fig, axs = plt.subplots(1, len(SITES), figsize=(17, 3.6), sharey=True)
for ax, s in zip(axs, SITES):
    rates[rates.site == s].set_index("species")[["convnext_baseline", "convnext_among_five", "birdnet_among_five"]].plot.bar(ax=ax, legend=False, width=0.85)
    ax.set_title(s); ax.set_yscale("log")
axs[0].set_ylabel("flagged windows / hour"); axs[-1].legend(fontsize=7)
plt.tight_layout(); plt.show()
rates.set_index(["site", "species"]).round(2)

# %% [markdown]
# ## 2. Confirmed calls at Taylor (goshawk) and Riverview (great gray)
# The confirmed calls are from the field team's own detector and manual review. Only the calls whose
# file is on disk are used.

# %%
held = DAYS["val"] + DAYS["test"]
m_files = [f for f in LEN["murie"] if f[:4] in held]
TL_M = Timeline(LEN["murie"], sorted(m_files))
CASES = {"taylor": ("Taylor*", "NOGO", 2), "riverview": ("Riverview*", "GGOW", 1)}
RATES = [2, 5, 10, 20, 50]
out = []
for site, (pat, prefix, k) in CASES.items():
    calls = [c for c in manual_calls(pat, prefix) if c[0] in LEN[site]]
    tl_s = Timeline(LEN[site], sorted(LEN[site]))
    methods = {
        "BirdNET p": (lambda s, k=k: bn[s][f"p_{CODES[k]}"].to_numpy(), lambda s: bn[s].file.to_numpy(), lambda s: bn[s].start_s.to_numpy(), 3),
        "ConvNeXT p": (lambda s, k=k: P[s][:, k], lambda s: cx[s]["file"], lambda s: cx[s]["start_s"], 5),
    }
    scores = {}
    for name, (sc, ff, ss, dur) in methods.items():
        scores[name] = {s: (ff(s), ss(s), sc(s), dur) for s in ("murie", site)}
    for tag, label in (("v2", "head, Murie-only"), (f"m_loso_{site}_raw", "head, site held out")):
        scores[label + " + gate"] = {s: (cx[s]["file"], cx[s]["start_s"], head(tag, s)[:, k], 5) for s in ("murie", site)}
    for name, d in scores.items():
        fm, sm, scm, dur = d["murie"]
        hm = np.isin([f[:4] for f in fm], held)
        fs, ss_, scs, _ = d[site]
        for R in RATES:
            thr = TL_M.threshold(fm[hm], sm[hm], dur, scm[hm], R)
            mk = tl_s.mask(fs, ss_, dur, scs > thr)
            out.append(dict(site=site, method=name, murie_rate=R, recall=tl_s.recall(mk, calls), site_rate=tl_s.per_hour(mk), n_calls=len(calls)))
res = pd.DataFrame(out)
fig, axs = plt.subplots(1, 2, figsize=(13, 4))
for ax, site in zip(axs, CASES):
    g = res[res.site == site].pivot(index="murie_rate", columns="method", values="recall")
    g.plot(ax=ax, marker="o", logx=True); ax.set_ylim(0, 1.02)
    ax.set_title(f"{site}: recall of {int(res[res.site == site].n_calls.iloc[0])} confirmed calls"); ax.set_xlabel("flagged s/h on Murie held-out days (threshold source)")
plt.tight_layout(); plt.show()
res.pivot_table(index=["site", "murie_rate"], columns="method", values="recall").round(2)

# %% [markdown]
# The same thresholds also decide how much audio each method flags at the target site itself
# (`site_rate`). Rates much above the Murie rate mean the threshold does not transfer.

# %%
res.pivot_table(index=["site", "murie_rate"], columns="method", values="site_rate").round(1)

# %% [markdown]
# ## 3. Butler North: is what the models flag there noise?
# For each ConvNeXT baseline detection we take BirdNET's probability of the same species; for the
# windows the ungated great-gray head flags with score above 0.87 (the failure found earlier) we look
# at BirdNET and ConvNeXT probabilities of great gray.

# %%
s = "butler"
top, k_c = P[s].max(1), P[s].argmax(1)
base = (top > 0.1) & (top >= cx[s]["maxnon"])
Pb = bn[s][[f"p_{c}" for c in CODES]].to_numpy()
rows = []
for k, c in enumerate(CODES):
    m = base & (k_c == k)
    if m.sum():
        at = overlap_max(cx[s]["file"][m], cx[s]["start_s"][m], 5, bn[s].file.to_numpy(), bn[s].start_s.to_numpy(), 3, Pb[:, k])
        rows.append(dict(species=c, convnext_detections=int(m.sum()), birdnet_p_gt_0_1=float((at > 0.1).mean()), birdnet_median_p=float(np.median(at))))
print("ConvNeXT baseline detections at Butler North and BirdNET's probability of the same species")
display(pd.DataFrame(rows).round(3))

Hu = head("v2", s, gated=False)[:, 1]
mis = Hu > 0.87
at = overlap_max(cx[s]["file"][mis][::20], cx[s]["start_s"][mis][::20], 5, bn[s].file.to_numpy(), bn[s].start_s.to_numpy(), 3, Pb[:, 1])
print(f"ungated great-gray head > 0.87: {mis.sum():,} of {len(mis):,} windows ({100 * mis.mean():.1f}%); on a 1 in 20 sample "
      f"BirdNET great-gray p > 0.1 in {100 * (at > 0.1).mean():.1f}%, median {np.median(at):.4f}; "
      f"ConvNeXT p median {np.median(P[s][mis, 1]):.4f}")

# %% [markdown]
# ## Reading the results
# See the summary cell printed above. Confirmed calls are few (65 goshawk calls in 4 Taylor files, a
# handful of great gray at Riverview), so each recall value moves in steps of about 1/n. BirdNET at 3 s
# and ConvNeXT at 5 s are compared at equal flagged time, which favours the finer window a little.
