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
# # 01 Rescorer evaluation: fewer false alarms at equal recall?
#
# **Setup.** ConvNeXT gives a 1024-d embedding and 9,736 class probabilities per 5 s window. A
# *rescorer* is a small head (linear or one-hidden-layer MLP) on the frozen embedding that outputs
# the five target probabilities. It is trained with (a) real Murie windows from days 0402-0409 as
# negatives and (b) calls from unseen-source recordings mixed into real Murie background at random
# SNR as positives (`training/build_mixtures.py`, `training/train_rescorer.py`).
# Variants: *clean* drops windows the baseline flagged from the negatives; *hard* keeps them as
# negatives (they are the suspected false alarms).
#
# **Fair comparison.** Scores are compared at the same false-alarm rate: the threshold of each
# method is set on held-out Murie days (0413-0415) so that it flags a given number of windows per
# hour (windows overlap at a 1 s hop, so this is flagged seconds per hour). Recall is then measured on
# (1) mixtures made from held-out *source recordings* and held-out *days*, (2) the manually
# confirmed barred-owl calls on the held-out days, (3) the manually confirmed goshawk calls at
# another site. **Caveat:** the Murie "false alarms" are windows without a baseline detection or
# with an unverified one; without human labels they are pseudo-negatives (assumption A7).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_processing.paths import INTERMEDIATE, KALEIDOSCOPE
from training.rescorer import CODES, DAYS, Rescorer, call_scores, load_region, scores, threshold_at

plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
TAG = "v2"
LEVELS = np.geomspace(0.25, 100, 14)
d = load_region("murie")
mix = dict(np.load(INTERMEDIATE / "mixtures/test.npz"))
heads = {p.stem: Rescorer.load(p) for p in sorted((INTERMEDIATE / "rescorer" / TAG).glob("*.pt"))}
other = {r: load_region(r) for r in ("riverview", "taylor", "butler", "singing")
         if (INTERMEDIATE / "features" / r).exists() and any((INTERMEDIATE / "features" / r).glob("*.npz"))}
print(len(d["start_s"]), "Murie windows; heads:", list(heads), "; other sites:", {r: len(o["start_s"]) for r, o in other.items()})


def all_scores(dd):
    s = {k: scores(k, dd, None) for k in ("prob", "rule", "margin")}
    s.update({k: h.predict(dd["emb"]) for k, h in heads.items()})
    return s


SM = all_scores(d)
SO = {r: all_scores(o) for r, o in other.items()}
SX = all_scores({"emb": mix["emb"], "tlogit": mix["tlogit"], "maxnon": mix["maxnon"]})
METHODS = list(SM)
STYLE = {"prob": ("k", "-"), "rule": ("0.5", "--"), "margin": ("0.5", ":"),
         "clean_lin": ("tab:blue", "--"), "clean_mlp": ("tab:blue", "-"),
         "hard_lin": ("tab:red", "--"), "hard_mlp": ("tab:red", "-")}

# %% [markdown]
# ## Reference sets

# %%
kc = pd.read_csv(next(KALEIDOSCOPE.glob("Murie*/cluster.csv")))
bar = kc[kc["MANUAL ID"] == "Barred Owl"]
bar_calls = [(f, o, du, 3) for f, o, du in zip(bar["IN FILE"], bar["OFFSET"], bar["DURATION"])]
bar_heldout = [c for c in bar_calls if c[0][:4] in DAYS["val"] + DAYS["test"]]
gos = []
if "taylor" in other:
    tk = pd.read_csv(next(KALEIDOSCOPE.glob("Taylor*/cluster.csv")))
    tk = tk[tk["MANUAL ID"].astype(str).str.startswith("NOGO")]
    have = set(other["taylor"]["file"])
    gos = [(f, o, du, 2) for f, o, du in zip(tk["IN FILE"], tk["OFFSET"], tk["DURATION"]) if f in have]
print(f"held-out Murie barred calls: {len(bar_heldout)}; goshawk calls at Taylor with audio: {len(gos)}")

test_rows = np.where(np.isin(d["day"], DAYS["test"]))[0]
near_bar = np.zeros(len(d["day"]), bool)  # windows around confirmed barred calls are not negatives for barred
for f, o, du, _ in bar_calls:
    near_bar |= (d["file"] == f) & (d["start_s"] >= o - 9) & (d["start_s"] <= o + du + 4)


def neg_scores(m, k):
    x = SM[m][test_rows, k]
    return x[~near_bar[test_rows]] if CODES[k] == "brdowl" else x


def recall_curve(m, k, pos):
    neg = neg_scores(m, k)
    return np.array([(pos > threshold_at(neg, lv)).mean() for lv in LEVELS])


# %% [markdown]
# ## 1. Synthetic mixtures (unseen sources, unseen days)

# %%
fig, axs = plt.subplots(1, 5, figsize=(20, 3.6), sharey=True)
for k, (ax, code) in enumerate(zip(axs, CODES)):
    pos_idx = mix["species"] == k
    for m in METHODS:
        c, ls = STYLE[m]
        ax.plot(LEVELS, recall_curve(m, k, SX[m][pos_idx, k]), color=c, ls=ls, label=m)
    ax.set_xscale("log"); ax.set_title(code); ax.set_xlabel("flagged windows / hour (held-out Murie)")
axs[0].set_ylabel("recall on mixtures"); axs[0].legend(fontsize=7)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Specificity: does a head fire on the *wrong* species?
# Each row: mixtures of one species (held-out sources). Each column: the fraction flagged as that
# class at the threshold giving 2 flagged windows/hour on Murie. A species-specific method is
# diagonal; a method that only learned "a call was added" lights up whole rows.

# %%
fig, axs = plt.subplots(1, 4, figsize=(17, 3.8))
for ax, m in zip(axs, ["prob", "rule", "clean_mlp", "hard_mlp"]):
    M = np.zeros((5, 5))
    for k in range(5):
        tau = threshold_at(neg_scores(m, k), 2)
        for j in range(5):
            M[j, k] = (SX[m][mix["species"] == j, k] > tau).mean()
    ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(5)); ax.set_xticklabels(CODES, rotation=45); ax.set_yticks(range(5)); ax.set_yticklabels(CODES)
    ax.set_title(m); ax.grid(False)
    for j in range(5):
        for k in range(5):
            ax.text(k, j, f"{M[j, k]:.2f}", ha="center", va="center", color="w" if M[j, k] < 0.6 else "k", fontsize=8)
axs[0].set_ylabel("true species (mixture)"); axs[0].set_xlabel("flagged as")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Real, independently confirmed calls
# Barred owl at Murie (held-out days) and goshawk at Taylor (a different site; thresholds still
# come from Murie negatives, so this also tests whether the operating point transfers).

# %%
real = {"brdowl (Murie, held-out days)": (3, [(m, call_scores(SM[m], d, bar_heldout)) for m in METHODS])}
if gos:
    real["norgos (Taylor)"] = (2, [(m, call_scores(SO["taylor"][m], other["taylor"], gos)) for m in METHODS])
fig, axs = plt.subplots(1, len(real), figsize=(6.5 * len(real), 3.8), squeeze=False)
for ax, (title, (k, items)) in zip(axs[0], real.items()):
    for m, sc in items:
        c, ls = STYLE[m]
        ax.plot(LEVELS, [(sc > threshold_at(neg_scores(m, k), lv)).mean() for lv in LEVELS], color=c, ls=ls, label=m)
    ax.set_xscale("log"); ax.set_title(f"{title}: n={len(items[0][1])} calls"); ax.set_xlabel("flagged windows / hour (held-out Murie)")
axs[0][0].set_ylabel("recall on confirmed calls"); axs[0][0].legend(fontsize=7)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Other sites: flagged windows per hour at the same Murie-calibrated thresholds
# There are no target labels at these sites beyond the few confirmed calls, so this shows how much
# each method flags, not whether the flags are right. A method that flags nothing anywhere is not
# useful, and one that flags far more is not more accurate.

# %%
LV = 2
rows = []
for r, So in SO.items():
    for m in METHODS:
        for k, code in enumerate(CODES):
            tau = threshold_at(neg_scores(m, k), LV)
            x = So[m][:, k]
            rows.append(dict(site=r, method=m, species=code, flagged_per_h=(x > tau).mean() * 3600))
xs = pd.DataFrame(rows)
if len(xs):
    fig, axs = plt.subplots(1, len(SO), figsize=(5 * len(SO), 3.6), sharey=True, squeeze=False)
    for ax, (r, g) in zip(axs[0], xs.groupby("site")):
        g.pivot(index="species", columns="method", values="flagged_per_h")[METHODS].plot.bar(ax=ax, legend=False, width=0.85)
        ax.set_title(f"{r} ({len(other[r]['start_s']) / 3600:.0f} h)"); ax.set_ylabel("flagged windows / h")
    axs[0][0].legend(fontsize=6)
    plt.tight_layout()
    plt.show()
    print(f"Murie held-out days flag {LV}/h by construction.")

# %% [markdown]
# ## Reading the figures
# See the text summary in the living context. Two cautions: the mixture test uses very few source
# recordings (held-out: 1 to 2 per species), and the Murie negatives are unlabeled.
