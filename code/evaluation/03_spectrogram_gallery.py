# %% [markdown]
# # 03 Spectrogram gallery of detected events (8 kHz audio, log-frequency STFT)
# Usage: python evaluation/03_spectrogram_gallery.py <species> <pmin> <pmax> <n> <out.png>

# %%
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import spectrogram

from data_processing.paths import MURIE_AUDIO as FOREST, RESULTS


def find(name):
    for p in (FOREST / name, FOREST / "Night Recordings" / name):
        if p.exists():
            return p


def panel(ax, f, start, dur, title):
    sr = sf.info(f).samplerate
    y, _ = sf.read(f, start=int(max(start - 2, 0) * sr), frames=int((dur + 4) * sr), dtype="float32")
    fr, t, S = spectrogram(y, sr, nperseg=512, noverlap=384)
    S = 10 * np.log10(S + 1e-12)
    ax.imshow(S, origin="lower", aspect="auto", extent=[t[0], t[-1], fr[0], fr[-1]],
              vmin=np.percentile(S, 40), vmax=np.percentile(S, 99.8), cmap="magma")
    ax.axvline(min(start, 2), color="c", lw=0.6); ax.axvline(min(start, 2) + dur, color="c", lw=0.6)
    ax.set_ylim(0, 4000); ax.set_title(title, fontsize=7); ax.tick_params(labelsize=6)


if __name__ == "__main__":
    sp, pmin, pmax, n, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    ev = pd.read_csv(RESULTS / "murie_overview/baseline_events.csv")
    e = ev[(ev.species == sp) & (ev.p_max >= pmin) & (ev.p_max < pmax)]
    e = e.sample(min(n, len(e)), random_state=1)
    rows = int(np.ceil(len(e) / 3))
    fig, axs = plt.subplots(rows, 3, figsize=(15, 2.4 * rows))
    for ax, r in zip(np.ravel(axs), e.itertuples()):
        panel(ax, find(r.file), r.start_s, r.end_s - r.start_s, f"{r.file} @{r.start_s}s p={r.p_max:.2f}")
    fig.tight_layout(); fig.savefig(out, dpi=90)
