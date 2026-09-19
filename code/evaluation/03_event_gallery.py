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
# # 03 Event gallery: goshawk and flammulated events
# Companion to `01_murie_overview` (which shows barred and boreal events). Spectrograms of random
# baseline events; the cyan lines mark the flagged interval, with 2 s of context.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from analysis.plots import event_gallery
from data_processing.paths import RESULTS

ev = pd.read_csv(RESULTS / "murie_overview" / "baseline_events.csv")

# %%
event_gallery(ev[(ev.species == "norgos") & (ev.p_max >= 0.5)], title="Goshawk events, p >= 0.5")
plt.show()

# %%
event_gallery(ev[(ev.species == "norgos") & (ev.p_max < 0.2)], title="Goshawk events, p < 0.2")
plt.show()

# %%
event_gallery(ev[ev.species == "flaowl"], n=9, title="Flammulated owl events (all confidences)")
plt.show()
