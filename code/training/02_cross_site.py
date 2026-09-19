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
# # 02 Cross-site behaviour of the rescorers (leave-one-site-out)
#
# A rescorer trained only on Murie background can misfire on a new site (the great-gray head flags a
# large share of Butler North windows). Here each held-out site is scored by heads that never saw it:
#
# * `raw`: trained on Murie (training days) plus the other sites' negatives.
# * `site-norm`: the same, but every site's embeddings are centred on that site's own mean
#   (needs only unlabeled audio from the new site).
# * `murie-only`: the Murie-only head from notebook 01 (the failing case).
# * `multi-mix`: as above, and the synthetic call mixtures are also built on the other sites'
#   backgrounds (the held-out site's background is never used).
#
# Thresholds always come from held-out Murie days (2 flagged windows / hour), so a head that flags
# far more than 2/h elsewhere is not transferring its operating point. There are almost no labels
# at these sites (65 goshawk calls at Taylor), so the flag rate says how much a method fires, not
# whether it is right.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_processing.paths import INTERMEDIATE, KALEIDOSCOPE
from training.rescorer import CODES, DAYS, Rescorer, call_scores, load_region, scores, threshold_at

plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
SITES = [s for s in ("riverview", "taylor", "butler", "singing")
         if (INTERMEDIATE / "rescorer" / f"loso_{s}").exists()]
KIND = "hard_mlp"
LV = 2
murie = load_region("murie")
sites = {s: load_region(s) for s in ["murie"] + SITES}
means = {s: d["emb"].astype(np.float32).mean(0) for s, d in sites.items()}
print({s: f"{len(d['start_s']) / 3600:.0f} h" for s, d in sites.items()}, "| held-out sites with heads:", SITES)


def head_scores(tag, kind, site):
    h = Rescorer.load(INTERMEDIATE / "rescorer" / tag / f"{kind}.pt")
    return h.predict(sites[site]["emb"], site_mean=means[site] if h.site_norm else None)


def murie_negatives(tag, kind):
    S = head_scores(tag, kind, "murie")
    return S[np.isin(murie["day"], DAYS["test"])]


def flags(neg_S, S, lv=LV):
    return np.array([(S[:, k] > threshold_at(neg_S[:, k], lv)).mean() * 3600 for k in range(5)])


# %% [markdown]
# ## Flagged windows per hour at the held-out site (Murie-calibrated thresholds)

# %%
rows = []
for s in SITES:
    base = scores("prob", sites[s], None)
    base_neg = scores("prob", murie, None)[np.isin(murie["day"], DAYS["test"])]
    for k, code in enumerate(CODES):
        rows.append(dict(site=s, species=code, method="baseline prob", flagged_per_h=flags(base_neg, base)[k]))
    for name, tag in (("murie-only", "v2"), ("raw (LOSO)", f"loso_{s}_raw"), ("site-norm (LOSO)", f"loso_{s}"),
                      ("multi-mix raw (LOSO)", f"m_loso_{s}_raw"), ("multi-mix site-norm (LOSO)", f"m_loso_{s}")):
        if not (INTERMEDIATE / "rescorer" / tag / f"{KIND}.pt").exists():
            continue
        f = flags(murie_negatives(tag, KIND), head_scores(tag, KIND, s))
        for k, code in enumerate(CODES):
            rows.append(dict(site=s, species=code, method=name, flagged_per_h=f[k]))
res = pd.DataFrame(rows)
fig, axs = plt.subplots(1, len(SITES), figsize=(5.5 * len(SITES), 3.8), sharey=False, squeeze=False)
for ax, s in zip(axs[0], SITES):
    g = res[res.site == s].pivot(index="species", columns="method", values="flagged_per_h")
    g.plot.bar(ax=ax, width=0.85)
    ax.set_yscale("symlog", linthresh=1); ax.set_title(f"{s} ({len(sites[s]['start_s']) / 3600:.0f} h): head {KIND}")
    ax.set_ylabel("flagged windows / h"); ax.axhline(LV, color="k", lw=0.6, ls=":")
plt.tight_layout()
plt.show()
res.pivot_table(index=["site", "species"], columns="method", values="flagged_per_h").round(1)

# %% [markdown]
# ## Real goshawk calls at Taylor when Taylor is held out

# %%
if "taylor" in SITES:
    tk = pd.read_csv(next(KALEIDOSCOPE.glob("Taylor*/cluster.csv")))
    tk = tk[tk["MANUAL ID"].astype(str).str.startswith("NOGO")]
    have = set(sites["taylor"]["file"])
    gos = [(f, o, du, 2) for f, o, du in zip(tk["IN FILE"], tk["OFFSET"], tk["DURATION"]) if f in have]
    lvls = np.geomspace(0.25, 100, 14)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    variants = [("baseline prob", None, "k", "-"), ("murie-only", "v2", "tab:red", "--"),
                ("raw (LOSO)", "loso_taylor_raw", "tab:orange", "-"), ("site-norm (LOSO)", "loso_taylor", "tab:blue", "-"),
                ("multi-mix raw (LOSO)", "m_loso_taylor_raw", "tab:green", "-"),
                ("multi-mix site-norm (LOSO)", "m_loso_taylor", "tab:purple", "-")]
    for name, tag, c, ls in variants:
        if tag is None:
            S, neg = scores("prob", sites["taylor"], None), scores("prob", murie, None)[np.isin(murie["day"], DAYS["test"])]
        else:
            if not (INTERMEDIATE / "rescorer" / tag / f"{KIND}.pt").exists():
                continue
            S, neg = head_scores(tag, KIND, "taylor"), murie_negatives(tag, KIND)
        sc = call_scores(S, sites["taylor"], gos)
        ax.plot(lvls, [(sc > threshold_at(neg[:, 2], lv)).mean() for lv in lvls], color=c, ls=ls, label=name)
    ax.set_xscale("log"); ax.set_xlabel("flagged windows / hour (held-out Murie)"); ax.set_ylabel(f"recall on {len(gos)} goshawk calls")
    ax.legend(); ax.set_title("Goshawk at a site the head never saw")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Reading the figures
# See the living context for the summary; the honest limits are the small number of confirmed calls
# and that flag rates at sites without labels cannot distinguish missed calls from false alarms.
