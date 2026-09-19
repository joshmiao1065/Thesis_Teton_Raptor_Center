"""Plot helpers shared by the notebooks (spectrogram panels and event galleries)."""
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram

from data_processing.paths import MURIE_AUDIO


def find_audio(name, root=MURIE_AUDIO):
    for p in (root / name, root / "Night Recordings" / name):
        if p.exists():
            return p


def spectrogram_panel(ax, path, start, dur, title="", pad=2, fmax=4000):
    """Log-power spectrogram of [start - pad, start + dur + pad] seconds; cyan lines mark the interval."""
    sr = sf.info(path).samplerate
    a = max(start - pad, 0)
    y, _ = sf.read(path, start=int(a * sr), frames=int((dur + 2 * pad) * sr), dtype="float32")
    y = y if y.ndim == 1 else y[:, 0]
    fr, t, S = spectrogram(y, sr, nperseg=512, noverlap=384)
    S = 10 * np.log10(S + 1e-12)
    ax.imshow(S, origin="lower", aspect="auto", extent=[t[0], t[-1], fr[0], fr[-1]],
              vmin=np.percentile(S, 40), vmax=np.percentile(S, 99.8), cmap="magma")
    ax.axvline(start - a, color="c", lw=0.6)
    ax.axvline(start - a + dur, color="c", lw=0.6)
    ax.set_ylim(0, fmax)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


def event_gallery(events, n=9, cols=3, seed=1, title=None):
    """Grid of spectrograms for a random sample of events (columns file, start_s, end_s, p_max)."""
    e = events.sample(min(n, len(events)), random_state=seed)
    rows = int(np.ceil(len(e) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 2.4 * rows), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, r in zip(axs.ravel(), e.itertuples()):
        ax.axis("on")
        spectrogram_panel(ax, find_audio(r.file), r.start_s, r.end_s - r.start_s,
                          f"{r.file} @{r.start_s}s  p={r.p_max:.2f}")
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return fig
