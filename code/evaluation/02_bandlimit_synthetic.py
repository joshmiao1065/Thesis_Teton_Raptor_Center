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
# # 02 Does the 8 kHz band limit change ConvNeXT's scores? (controlled test)
#
# The field files are 8 kHz, but the model was trained on full-band audio. If the missing
# 4-16 kHz content mattered, the same synthetic clip should score differently before and after
# band limiting. Synthetic clips (60 s, known species, forest or Gaussian background at several
# SNRs) were scored as generated ("full") and after a 32 -> 8 -> 32 kHz round trip ("8khz"),
# with `evaluation/score_bandlimit.py`. Noise-only clips give the false-positive side.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from data_processing.paths import RESULTS

plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
df = pd.read_csv(RESULTS / "bandlimit" / "clip_scores.csv")
sp = df[df.species != "none"]
print(len(sp), "synthetic clips with a target species;", (df.species == "none").sum() // 2, "noise-only clips")

# %% [markdown]
# ## Detection of the true species, clip level
# A clip counts as detected when some window has the true species as top-1 with p > 0.1.

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
d = sp.groupby(["species", "cond"]).det_true.mean().unstack()
d.plot.bar(ax=ax[0], color=["tab:orange", "tab:blue"]); ax[0].set_ylabel("share of clips detected"); ax[0].set_title("By species")
s = sp.groupby(["snr", "cond"]).det_true.mean().unstack()
for c, col in zip(s.columns, ["tab:orange", "tab:blue"]):
    ax[1].plot(s.index, s[c], marker="o", label=c, color=col)
ax[1].set_xlabel("SNR (dB)"); ax[1].set_ylabel("share of clips detected"); ax[1].legend(); ax[1].set_title("By SNR")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Score of the true species

# %%
piv = sp.pivot_table(index=["src", "bg", "snr", "species"], columns="cond", values="max_true").reset_index()
fig, ax = plt.subplots(figsize=(4.8, 4.5))
ax.scatter(piv["full"], piv["8khz"], s=6, alpha=0.5)
ax.plot([0, 1], [0, 1], "r--")
ax.set_xlabel("max probability, full band"); ax.set_ylabel("max probability, 8 kHz round trip")
ax.set_title("Same clip, two conditions")
plt.tight_layout()
plt.show()
print("mean absolute change in max probability:", (piv["8khz"] - piv["full"]).abs().mean().round(3))
print("mean change:", (piv["8khz"] - piv["full"]).mean().round(3))

# %% [markdown]
# ## False alarms
# Windows where a wrong target species was reported, and detections on the noise-only clips.

# %%
print("wrong-target detections per clip:")
print(sp.groupby(["cond", "species"]).det_other.mean().unstack(0).round(3))
print("\nnoise-only clips (windows flagged out of 56):")
print(df[df.species == "none"].groupby(["src", "cond"])[["det_any_windows", "n_windows"]].mean())

# %% [markdown]
# ## Reading the figures
# The two conditions are nearly identical: the band limit does not create false positives and does
# not explain the field results. False alarms on this synthetic background are zero, so the field
# false alarms come from real field sounds that the synthetic backgrounds do not contain.
