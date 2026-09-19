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
# # 01 Murie 2 overview: what does ConvNeXT say about the recordings?
#
# Inputs: per-window scores from `evaluation/run_convnext.py` (every 5 s window, 1 s hop, all 198
# recordings). The baseline rule keeps a window when the top-1 class over all 9,736 classes is one
# of the five targets and its probability exceeds 0.1; consecutive windows are merged into events.
#
# Questions: (1) what does the model output on this site, (2) where and when do target events
# occur, (3) how does it score the manually confirmed barred-owl calls (an independent reference),
# (4) what do the events look like.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoConfig

from analysis.plots import event_gallery
from data_processing.paths import INTERMEDIATE, KALEIDOSCOPE, RESULTS
from evaluation.events import TARGET_CODES, baseline_detections, load_scores, merge_events
from models.convnext import MODEL_ID

plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
OUT = RESULTS / "murie_overview"
OUT.mkdir(parents=True, exist_ok=True)
COLORS = dict(zip(TARGET_CODES, ["tab:purple", "tab:green", "tab:orange", "tab:red", "tab:blue"]))


class _Labels:  # only the label map is needed, not the model
    config = AutoConfig.from_pretrained(MODEL_ID)


w = load_scores(INTERMEDIATE / "convnext_murie", _Labels)
w["hour"] = w.file.str[5:7].astype(int)
det = baseline_detections(w)
ev = merge_events(det)
print(f"{len(w):,} windows in {w.file.nunique()} files ({len(w) / 3600:.0f} h of windows at 1 s hop)")
ev.to_csv(OUT / "baseline_events.csv", index=False)
pd.DataFrame({"windows": det.species.value_counts(), "events": ev.species.value_counts(),
              "median_p_max": ev.groupby("species").p_max.median().round(2)})

# %% [markdown]
# ## 1. What wins the top-1 slot?
# If the model were confident about what it hears, the top-1 probability would be high and the
# class plausible for a Wyoming forest. Instead a single class takes over half of all windows.

# %%
top = w.top1.value_counts(normalize=True).head(20)
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(top.index, top.values, color=["tab:red" if c in TARGET_CODES else "0.6" for c in top.index])
ax.set_ylabel("share of all windows")
ax.set_title("Top-1 class over all Murie windows (red = one of the five targets)")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.show()
print(f"Windows whose top-1 probability is below 0.1: {(w.top1_p < 0.1).mean():.1%}")

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].hist(w.top1_p, bins=50, color="0.5")
ax[0].axvline(0.1, color="r", ls="--", label="baseline threshold 0.1")
ax[0].set_xlabel("top-1 probability"); ax[0].set_ylabel("windows"); ax[0].set_yscale("log"); ax[0].legend()
ax[0].set_title("All windows")
tgt = w[w.top1.isin(TARGET_CODES)]
for c in TARGET_CODES:
    ax[1].hist(tgt[tgt.top1 == c].top1_p, bins=np.linspace(0, 1, 41), alpha=0.6, label=c, color=COLORS[c])
ax[1].axvline(0.1, color="r", ls="--"); ax[1].set_yscale("log"); ax[1].legend(fontsize=7)
ax[1].set_xlabel("top-1 probability"); ax[1].set_title("Windows whose top-1 is a target")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. When and where do target events occur?

# %%
ev["hour"] = ev.file.str[5:7].astype(int)
hourly = ev.groupby(["hour", "species"]).size().unstack(fill_value=0).reindex(range(24), fill_value=0)
fig, ax = plt.subplots(figsize=(10, 3.5))
hourly.plot.bar(stacked=True, ax=ax, color=[COLORS[c] for c in hourly.columns])
ax.set_xlabel("hour of day (file start)"); ax.set_ylabel("events")
ax.set_title("Baseline events by hour")
plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
for sp in ("borowl", "brdowl", "norgos"):
    c = ev[ev.species == sp].file.value_counts().to_numpy()
    ax[0].plot(np.arange(1, len(c) + 1), np.cumsum(c) / c.sum(), label=f"{sp} ({c.sum()} events, {len(c)} files)", color=COLORS[sp])
ax[0].set_xscale("log"); ax[0].set_xlabel("files (most events first)"); ax[0].set_ylabel("cumulative share of events")
ax[0].legend(fontsize=7); ax[0].set_title("Concentration of events in files")
for sp in ("borowl", "brdowl", "norgos", "flaowl"):
    ax[1].hist(ev[ev.species == sp].p_max, bins=np.linspace(0.1, 1, 19), alpha=0.6, label=sp, color=COLORS[sp])
ax[1].set_xlabel("event confidence (max probability)"); ax[1].set_ylabel("events"); ax[1].legend(fontsize=7)
ax[1].set_title("Event confidence")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Which other classes fire together with each target?
# For windows where a target has probability above 0.1, count the other classes that also exceed 0.1
# (from the stored top-10), and compare with the background rate over a random sample of all windows.

# %%
import collections

id2 = _Labels.config.id2label
co = {t: collections.Counter() for t in TARGET_CODES}
n_t = dict.fromkeys(TARGET_CODES, 0)
base = collections.Counter()
n_base = 0
rng = np.random.default_rng(0)
for f in sorted((INTERMEDIATE / "convnext_murie").glob("*.npz")):
    z = np.load(f)
    ti, tv, tp = z["top_idx"], z["top_p"], z["target_p"]
    for k, t in enumerate(TARGET_CODES):
        for i in np.where(tp[:, k] > 0.1)[0]:
            n_t[t] += 1
            co[t].update(id2[int(c)] for c, v in zip(ti[i], tv[i]) if v > 0.1 and id2[int(c)] != t)
    for i in rng.choice(len(ti), min(300, len(ti)), replace=False):
        n_base += 1
        base.update(id2[int(c)] for c, v in zip(ti[i], tv[i]) if v > 0.1)

fig, axs = plt.subplots(1, 3, figsize=(14, 3.5))
for ax, t in zip(axs, ["borowl", "brdowl", "norgos"]):
    items = co[t].most_common(8)
    ax.barh([c for c, _ in items][::-1], [n / n_t[t] for _, n in items][::-1], color=COLORS[t])
    ax.set_title(f"{t}: co-active classes ({n_t[t]} windows)")
    ax.set_xlabel("share of the target's windows")
plt.tight_layout()
plt.show()
print("background rate of the most common classes:", [(c, round(n / n_base, 3)) for c, n in base.most_common(6)])

# %% [markdown]
# ## 4. Independent reference: manually confirmed barred-owl calls
# Kaleidoscope (the field team's own detector) has 74 barred-owl rows confirmed by a human. For each
# call we take the windows overlapping it and look at the barred and boreal probabilities.

# %%
kc = pd.read_csv(next(KALEIDOSCOPE.glob("Murie*/cluster.csv")))
bar = kc[kc["MANUAL ID"] == "Barred Owl"]
rows = []
for fname, off, dur in zip(bar["IN FILE"], bar["OFFSET"], bar["DURATION"]):
    seg = w[(w.file == fname) & (w.start_s >= int(np.floor(off)) - 4) & (w.start_s <= int(np.ceil(off + dur)))]
    if len(seg):
        rows.append(dict(file=fname, offset=off, dur=dur, max_barred=seg.p_brdowl.max(), max_boreal=seg.p_borowl.max(),
                         top1=seg.top1.mode().iloc[0]))
ref = pd.DataFrame(rows)
ref.to_csv(OUT / "manual_barred_calls_scored.csv", index=False)
fig, ax = plt.subplots(1, 3, figsize=(14, 3.3))
ax[0].hist(ref.max_barred, bins=np.linspace(0, 1, 21), color=COLORS["brdowl"])
ax[0].set_title(f"max barred probability per call (n={len(ref)})"); ax[0].set_xlabel("probability")
ax[1].hist(ref.max_boreal, bins=np.linspace(0, 1, 21), color=COLORS["borowl"])
ax[1].set_title("max boreal probability on the same calls"); ax[1].set_xlabel("probability")
ref.top1.value_counts().head(6).plot.bar(ax=ax[2], color="0.5"); ax[2].set_title("most common top-1 class in the call windows")
plt.tight_layout()
plt.show()
print(f"calls with barred p > 0.1 in an overlapping window: {(ref.max_barred > 0.1).mean():.1%}; "
      f"p > 0.5: {(ref.max_barred > 0.5).mean():.1%}")

# %% [markdown]
# ## 5. What do the events look like?
# Spectrograms (0-4 kHz) of random events. Cyan lines mark the flagged interval, with 2 s of context.
# Barred events first, then boreal at high and at low confidence.

# %%
event_gallery(ev[(ev.species == "brdowl") & (ev.p_max >= 0.9)], title="Barred owl events, p >= 0.9")
plt.show()

# %%
event_gallery(ev[(ev.species == "borowl") & (ev.p_max >= 0.85)], title="Boreal owl events, p >= 0.85")
plt.show()

# %%
event_gallery(ev[(ev.species == "borowl") & (ev.p_max < 0.25)], title="Boreal owl events, p < 0.25")
plt.show()

# %% [markdown]
# ## Reading the figures
# * The top-1 class is dominated by one implausible class and more than half of the windows have no
#   class above 0.1, so the output on this site is mostly a noise-sink rather than a species call.
# * The barred owl is the well-behaved class: high confidence on confirmed calls, nocturnal timing.
# * Boreal (and goshawk) events cluster at dawn and dusk, are mostly low confidence, and often show
#   no obvious call in the spectrogram. This is a visual impression, not a label: the audit sheet
#   (`analysis/make_audit_sheet.py`) is the way to turn it into measurements.
