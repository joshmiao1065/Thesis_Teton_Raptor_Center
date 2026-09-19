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
# # 03 Is benchmark performance retained? (synthetic clips, held-out source recordings)
#
# The synthetic benchmark has 60 s clips of each target species in forest or Gaussian noise at SNRs
# from -40 to +40 dB, plus noise-only clips. The heads were trained only on calls from *training*
# source recordings, so we evaluate on the clips whose source recording is held out (a small set:
# one source recording per species, two for barred owl). Clip score = maximum over the clip's
# windows. **AUROC** is computed per species: that species' clips at a given SNR range versus all
# clips of the other species and the noise-only clips, so it also checks cross-species confusion.
# The operating point for the fixed-threshold table comes from held-out Murie days (2 flagged
# windows per hour), as elsewhere.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_processing.mixtures import load_calls, split_sources
from data_processing.paths import INTERMEDIATE
from training.rescorer import CODES, DAYS, Rescorer, gate, load_region, scores, threshold_at

plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
FOLDER = {"01-agos": "norgos", "02-ggow": "grgowl", "03-flow": "flaowl", "04-boow": "borowl", "05-baow": "brdowl"}
test_sources = {sp: {sid for sid, _ in split_sources(items)["test"]} for sp, items in load_calls().items()}
print("held-out sources:", {k: sorted(v) for k, v in test_sources.items()})

rows, embs = [], []
for f in sorted((INTERMEDIATE / "features/synthetic").glob("*.npz")):
    name = f.stem
    if "__generated__" in name:
        folder, rest = name.split("__generated__")
        sp = FOLDER[folder[:7]]
        src = rest.split("_generated_")[0].replace("call-", "")
        bg = "forest" if "forest" in rest else "gaussian"
        snr = int(rest.rsplit("snr", 1)[1])
    elif name.startswith("base__"):
        sp, src, bg, snr = "none", name, name.split("__")[1].split("_1min")[0], np.nan
    else:
        continue  # cropped call clips and source recordings
    z = np.load(f)
    rows.append(dict(name=name, species=sp, source=src, bg=bg, snr=snr,
                     heldout=(sp == "none") or (src in test_sources[sp])))
    embs.append({k: z[k] for k in ("emb", "tlogit", "maxnon")})
meta = pd.DataFrame(rows)
print(meta.groupby("species").agg(clips=("name", "size"), heldout=("heldout", "sum")))

# %%
heads = {f"{tag}/{k}": Rescorer.load(INTERMEDIATE / "rescorer" / tag / f"{k}.pt")
         for tag in ("v2", "all_raw", "all_norm", "m_all_raw", "m_all_norm") if (INTERMEDIATE / "rescorer" / tag).exists() for k in ("hard_mlp", "hard_lin")}
D = {k: np.concatenate([e[k] for e in embs]) for k in ("emb", "tlogit", "maxnon")}
n_win = np.array([len(e["emb"]) for e in embs])
starts = np.r_[0, np.cumsum(n_win)[:-1]]
site_mean = D["emb"].astype(np.float32).mean(0)  # unlabeled mean of the benchmark itself, for site-normalized heads
S = {m: scores(m, D, None) for m in ("prob", "rule", "margin")}
S.update({k: h.predict(D["emb"], site_mean=site_mean if h.site_norm else None) for k, h in heads.items()})
GATE = 0.001
for k in [k for k in S if k.startswith("m_all_raw")]:  # gated versions of the multi-site heads
    S[k + "+gate"] = gate(S[k], D["tlogit"], GATE)
clip = {m: np.stack([np.maximum.reduceat(S[m][:, k], starts) for k in range(5)], 1) for m in S}

murie = load_region("murie")
mtest = np.isin(murie["day"], DAYS["test"])
mean_m = murie["emb"].astype(np.float32).mean(0)
thr = {}
for m in S:
    base_m = m.removesuffix("+gate")
    if base_m in heads:
        h = heads[base_m]
        neg = h.predict(murie["emb"][mtest], site_mean=mean_m if h.site_norm else None)
        if m.endswith("+gate"):
            neg = gate(neg, murie["tlogit"][mtest], GATE)
    else:
        neg = scores(m, {k: murie[k][mtest] for k in ("tlogit", "maxnon")}, None)
    thr[m] = [threshold_at(neg[:, k], 2) for k in range(5)]


def auroc(pos, neg):
    r = pd.Series(np.r_[pos, neg]).rank().to_numpy()
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


sp_ = meta.species.to_numpy()
snr = meta.snr.to_numpy()
held = meta.heldout.to_numpy()
BINS = {"low SNR (<= -20 dB)": (-99, -20), "mid SNR (-10..0 dB)": (-10, 0), "high SNR (>= 5 dB)": (5, 99)}
res = []
for m in S:
    for k, code in enumerate(CODES):
        for bname, (lo, hi) in BINS.items():
            in_bin = (snr >= lo) & (snr <= hi)
            pos = clip[m][held & (sp_ == code) & in_bin, k]
            neg = clip[m][held & (sp_ != code) & ((sp_ == "none") | in_bin), k]
            if len(pos) and len(neg):
                res.append(dict(method=m, species=code, snr_bin=bname, auroc=auroc(pos, neg), n_pos=len(pos)))
res = pd.DataFrame(res)

# %% [markdown]
# ## AUROC per species and SNR range

# %%
fig, axs = plt.subplots(1, 3, figsize=(17, 3.8), sharey=True)
for ax, b in zip(axs, BINS):
    g = res[res.snr_bin == b].pivot(index="species", columns="method", values="auroc")[list(S)]
    g.plot.bar(ax=ax, legend=False, width=0.85)
    ax.set_title(b)
    ax.set_ylim(0.4, 1.02)
axs[0].set_ylabel("AUROC (held-out sources)")
axs[2].legend(fontsize=6, ncol=2)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Detection at the Murie-calibrated operating point (2 flagged windows / hour)

# %%
det = []
ge = snr >= -10
for m in S:
    for k, code in enumerate(CODES):
        pos = clip[m][held & (sp_ == code) & ge, k]
        oth = clip[m][held & (sp_ != code) & (sp_ != "none") & ge, k]
        noise = clip[m][held & (sp_ == "none"), k]
        det.append(dict(method=m, species=code, detected=(pos > thr[m][k]).mean(),
                        other_species_flagged=(oth > thr[m][k]).mean(), noise_only_flagged=(noise > thr[m][k]).mean()))
det = pd.DataFrame(det)
fig, axs = plt.subplots(1, 3, figsize=(17, 3.6), sharey=True)
for ax, col in zip(axs, ("detected", "other_species_flagged", "noise_only_flagged")):
    det.pivot(index="species", columns="method", values=col)[list(S)].plot.bar(ax=ax, legend=False, width=0.85)
    ax.set_title(col.replace("_", " ") + " (clips, SNR >= -10 dB)")
plt.tight_layout()
plt.show()
det.pivot_table(index="species", columns="method", values="detected").round(2)

# %% [markdown]
# ## Key methods, numbers
# Baseline probability, the Murie-only MLP head, the multi-site MLP head and the same head gated by
# the baseline probability (p > 0.001).

# %%
KEY = ["prob", "v2/hard_mlp", "m_all_raw/hard_mlp", "m_all_raw/hard_mlp+gate"]
for col in ("detected", "other_species_flagged", "noise_only_flagged"):
    print(col)
    print(det[det.method.isin(KEY)].pivot(index="species", columns="method", values=col)[KEY].round(2), "\n")
low = res[(res.snr_bin == "low SNR (<= -20 dB)") & res.method.isin(KEY)]
print("AUROC at low SNR (<= -20 dB)")
print(low.pivot(index="species", columns="method", values="auroc")[KEY].round(2))

# %% [markdown]
# ## Reading the figures
# A method retains benchmark performance if its AUROC and detection rate at the transferred
# operating point are not below the baseline's on the held-out source clips, with no rise in
# cross-species or noise-only flags. The held-out set is small (one source recording per species),
# so differences of a few percent are noise.
