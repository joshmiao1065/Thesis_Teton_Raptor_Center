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
# # 04 Gated rescorer: keep the gains, drop the shortcut
#
# Notebook 02 showed that a head trained on limited negatives can learn a background shortcut
# (at Butler North the great-gray head scores 0.9-1.0 on 18% of all windows, which contain only
# broadband low-frequency noise, while the original model's probability on them is about zero).
# **Gate:** keep a head score only where the original ConvNeXT probability for the same class
# exceeds `g`; the head then re-ranks candidate windows and cannot create detections from nothing.
#
# Everything is evaluated at equal false-alarm rate as before (thresholds from held-out Murie days,
# computed on the *gated* scores). Heads for the held-out site are leave-one-site-out
# (`m_loso_<site>_raw`, trained on mixtures and negatives from the other sites); Murie evaluations use
# the Murie-only head `v2`. **Caveat:** the gate value was chosen while looking at Butler and Taylor,
# so those two sites are not a clean test of it. The sweep over `g` below shows the sensitivity.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_processing.paths import INTERMEDIATE, KALEIDOSCOPE
from training.rescorer import (
    CODES,
    DAYS,
    Rescorer,
    call_scores,
    gate,
    load_region,
    scores,
    threshold_at,
)

plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
G = 0.001
LEVELS = np.geomspace(0.25, 30, 12)
KIND = "hard_mlp"
murie = load_region("murie")
mt = np.isin(murie["day"], DAYS["test"])
sites = {s: load_region(s) for s in ("taylor", "butler", "riverview")}


def head(tag, d):
    h = Rescorer.load(INTERMEDIATE / "rescorer" / tag / f"{KIND}.pt")
    return h.predict(d["emb"], site_mean=d["emb"].astype(np.float32).mean(0) if h.site_norm else None)


def variants(tag, d, dm):
    """Scores for baseline, ungated head and gated heads on d, and matching Murie held-out-day scores on dm."""
    out = {"baseline": (scores("prob", d, None), scores("prob", dm, None)), "head": (head(tag, d), head(tag, dm))}
    for g in (0.0005, G, 0.01):
        out[f"head, gate p>{g}"] = (gate(out["head"][0], d["tlogit"], g), gate(out["head"][1], dm["tlogit"], g))
    return out


mm = {k: murie[k][mt] for k in ("emb", "tlogit", "maxnon", "day")}
STY = {"baseline": ("k", "-"), "head": ("tab:red", "--"), "head, gate p>0.0005": ("tab:blue", ":"),
       f"head, gate p>{G}": ("tab:blue", "-"), "head, gate p>0.01": ("tab:green", "-")}

# %% [markdown]
# ## 1. Real goshawk calls at Taylor (leave-one-site-out head)

# %%
tk = pd.read_csv(next(KALEIDOSCOPE.glob("Taylor*/cluster.csv")))
tk = tk[tk["MANUAL ID"].astype(str).str.startswith("NOGO")]
gos = [(f, o, du, 2) for f, o, du in zip(tk["IN FILE"], tk["OFFSET"], tk["DURATION"]) if f in set(sites["taylor"]["file"])]
V = variants("m_loso_taylor_raw", sites["taylor"], mm)
fig, ax = plt.subplots(figsize=(6.5, 4))
for name, (S, Sm) in V.items():
    sc = call_scores(S, sites["taylor"], gos)
    c, ls = STY[name]
    ax.plot(LEVELS, [(sc > threshold_at(Sm[:, 2], lv)).mean() for lv in LEVELS], color=c, ls=ls, label=name)
ax.set_xscale("log"); ax.set_xlabel("flagged windows / hour (held-out Murie)"); ax.set_ylabel(f"recall on {len(gos)} goshawk calls")
ax.legend(fontsize=7); ax.set_title("Goshawk at Taylor (head never saw the site)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Real barred-owl calls at Murie (held-out days, Murie-only head)

# %%
kc = pd.read_csv(next(KALEIDOSCOPE.glob("Murie*/cluster.csv")))
bar = kc[kc["MANUAL ID"] == "Barred Owl"]
bar_h = [(f, o, du, 3) for f, o, du in zip(bar["IN FILE"], bar["OFFSET"], bar["DURATION"]) if f[:4] in DAYS["val"] + DAYS["test"]]
near = np.zeros(len(murie["day"]), bool)
for f, o, du, _ in [(f, o, du, 3) for f, o, du in zip(bar["IN FILE"], bar["OFFSET"], bar["DURATION"])]:
    near |= (murie["file"] == f) & (murie["start_s"] >= o - 9) & (murie["start_s"] <= o + du + 4)
V = variants("v2", murie, mm)
nb = ~near[mt]
fig, ax = plt.subplots(figsize=(6.5, 4))
for name, (S, Sm) in V.items():
    sc = call_scores(S, murie, bar_h)
    c, ls = STY[name]
    ax.plot(LEVELS, [(sc > threshold_at(Sm[nb, 3], lv)).mean() for lv in LEVELS], color=c, ls=ls, label=name)
ax.set_xscale("log"); ax.set_xlabel("flagged windows / hour (held-out Murie)"); ax.set_ylabel(f"recall on {len(bar_h)} barred calls")
ax.legend(fontsize=7); ax.set_title("Barred owl at Murie, held-out days")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Flagged windows per hour at the held-out sites (2 / hour on Murie)

# %%
rows = []
for s, tag in (("taylor", "m_loso_taylor_raw"), ("butler", "m_loso_butler_raw"), ("riverview", "m_loso_riverview_raw")):
    for name, (S, Sm) in variants(tag, sites[s], mm).items():
        for k, code in enumerate(CODES):
            rows.append(dict(site=s, species=code, method=name, flagged_per_h=(S[:, k] > threshold_at(Sm[:, k], 2)).mean() * 3600))
xs = pd.DataFrame(rows)
fig, axs = plt.subplots(1, 3, figsize=(17, 3.8))
for ax, s in zip(axs, ("taylor", "butler", "riverview")):
    xs[xs.site == s].pivot(index="species", columns="method", values="flagged_per_h")[list(STY)].plot.bar(ax=ax, legend=False, width=0.85)
    ax.set_yscale("symlog", linthresh=1); ax.set_title(s); ax.set_ylabel("flagged windows / h")
axs[0].legend(fontsize=6)
plt.tight_layout()
plt.show()
xs.pivot_table(index=["site", "species"], columns="method", values="flagged_per_h").round(1)

# %% [markdown]
# ## 4. Synthetic mixtures (unseen sources), gated versus ungated

# %%
mx = dict(np.load(INTERMEDIATE / "mixtures/test.npz"))
hm = Rescorer.load(INTERMEDIATE / "rescorer/v2" / f"{KIND}.pt")
Sx = hm.predict(mx["emb"])
Sxb = 1 / (1 + np.exp(-mx["tlogit"]))
Smh = head("v2", mm)
fig, axs = plt.subplots(1, 5, figsize=(20, 3.5), sharey=True)
for k, (ax, code) in enumerate(zip(axs, CODES)):
    pos = mx["species"] == k
    curves = {"baseline": (Sxb[pos, k], scores("prob", mm, None)[:, k]), "head": (Sx[pos, k], Smh[:, k])}
    for g in (G, 0.01):
        curves[f"head, gate p>{g}"] = (gate(Sx, mx["tlogit"], g)[pos, k], gate(Smh, mm["tlogit"], g)[:, k])
    for name, (p, neg) in curves.items():
        c, ls = STY[name]
        ax.plot(LEVELS, [(p > threshold_at(neg, lv)).mean() for lv in LEVELS], color=c, ls=ls, label=name)
    ax.set_xscale("log"); ax.set_title(code); ax.set_xlabel("flagged windows / hour")
axs[0].set_ylabel("recall on mixtures"); axs[0].legend(fontsize=6)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Reading the figures
# * If the gated curves in sections 1, 2 and 4 sit close to the ungated head while section 3 falls back
#   to about the baseline's flag rate, the gate keeps the gains and removes the shortcut.
# * The price of a gate is that windows where the original model has almost no evidence for a class can
#   never be recovered; the sweep over `g` shows how much recall that costs.
