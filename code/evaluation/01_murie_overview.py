# %% [markdown]
# # 01 Murie 2 overview: what does ConvNeXT say about the recordings?
# Inputs: per-window scores from `evaluation.run_convnext` (`intermediate/convnext_murie`).
# Questions: (1) does the baseline reproduce, (2) what wins top-1 when it is not a target,
# (3) how do boreal and barred detections relate, (4) how does the model score the
# manually confirmed barred-owl calls from the independent detector.

# %%

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoConfig

from evaluation.events import TARGET_CODES, baseline_detections, load_scores, merge_events
from models.convnext import MODEL_ID

from data_processing.paths import INTERMEDIATE, KALEIDOSCOPE, RESULTS

OUT = RESULTS / "murie_overview"
OUT.mkdir(parents=True, exist_ok=True)


class _M:  # only the label map is needed
    config = AutoConfig.from_pretrained(MODEL_ID)


w = load_scores(INTERMEDIATE / "convnext_murie", _M)
w["hour"] = w.file.str[5:7].astype(int)
w["day"] = w.file.str[:4]
print(len(w), "windows,", w.file.nunique(), "files")

# %% [markdown]
# ## 1. Baseline detections and events

# %%
det = baseline_detections(w)
ev = merge_events(det)
print("windows kept:", det.species.value_counts().to_dict())
print("events:", ev.species.value_counts().to_dict())
print(ev.groupby("species").p_max.describe().round(2))

# %% [markdown]
# ## 2. What wins top-1 across all windows?

# %%
top = w.top1.value_counts()
print((top.head(15) / len(w)).round(4))
print("share of windows whose top-1 probability is below 0.1:", (w.top1_p < 0.1).mean().round(3))
hi = w[w.top1_p >= 0.5].top1.value_counts().head(15)
print("top-1 species with p >= 0.5:\n", hi)

# %% [markdown]
# ## 3. Target probabilities: how often is a target the runner-up, and by hour of day

# %%
P = w[[f"p_{c}" for c in TARGET_CODES]].to_numpy()
w["best_target"] = np.array(TARGET_CODES)[P.argmax(1)]
w["best_target_p"] = P.max(1)
tab = pd.DataFrame({c: [(w[f"p_{c}"] > t).sum() for t in (0.05, 0.1, 0.3, 0.5, 0.9)] for c in TARGET_CODES},
                   index=["p>0.05", "p>0.1", "p>0.3", "p>0.5", "p>0.9"])
print("windows with target probability above threshold (any rank):\n", tab)
hourly = det.assign(hour=det.file.str[5:7].astype(int)).groupby(["hour", "species"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(8, 3.5))
hourly.plot.bar(stacked=True, ax=ax)
ax.set_ylabel("detected windows")
fig.tight_layout()
fig.savefig(OUT / "detections_by_hour.png", dpi=120)

# %% [markdown]
# ## 4. Boreal versus barred: co-occurrence and confidence

# %%
bo = w[(w.p_borowl > 0.1)]
near = w.set_index(["file", "start_s"]).p_brdowl
print("boreal>0.1 windows:", len(bo), " with barred prob > 0.05 within +-10 s:")
idx = w.set_index(["file", "start_s"])
hit = 0
for (f, s) in zip(bo.file, bo.start_s):
    seg = idx.loc[f].loc[max(s - 10, 0) : s + 10, "p_brdowl"]
    hit += bool((seg > 0.05).any())
print(hit, f"({hit / max(len(bo), 1):.2%})")

# %% [markdown]
# ## 5. Independent reference: manually confirmed barred-owl calls

# %%
kc = pd.read_csv(next(KALEIDOSCOPE.glob("Murie*/cluster.csv")))
bar = kc[kc["MANUAL ID"] == "Barred Owl"].copy()
rows = []
for fname, off, dur_ in zip(bar["IN FILE"], bar["OFFSET"], bar["DURATION"]):
    lo, hi = int(np.floor(off)) - 4, int(np.ceil(off + dur_))  # windows overlapping the call
    seg = w[(w.file == fname) & (w.start_s >= lo) & (w.start_s <= hi)]
    if seg.empty:
        rows.append(dict(file=fname, offset=off, n=0)); continue
    rows.append(dict(file=fname, offset=off, dur=dur_, n=len(seg), max_barred=seg.p_brdowl.max(),
                     max_boreal=seg.p_borowl.max(), any_target_top1=seg.top1.isin(TARGET_CODES).any(),
                     top1_mode=seg.top1.mode().iloc[0]))
ref = pd.DataFrame(rows)
print(ref.describe(include="all").T[["count", "mean", "50%"]].round(3))
print("calls with barred p>0.1 in an overlapping window:", (ref.max_barred > 0.1).mean().round(3),
      "| p>0.5:", (ref.max_barred > 0.5).mean().round(3))
print(ref.top1_mode.value_counts().head(8))
ref.to_csv(OUT / "manual_barred_calls_scored.csv", index=False)
ev.to_csv(OUT / "baseline_events.csv", index=False)
