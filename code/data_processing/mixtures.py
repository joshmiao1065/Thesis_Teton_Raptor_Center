"""Build labelled 5 s training/evaluation windows by mixing a target call into field background.

Positives are short clips of the five target species (band-limited to 8 kHz to match the field
audio). Backgrounds are 5 s windows read from the field recordings at 8 kHz, restricted to
places where the baseline model reported no target probability above a floor.
"""
import glob
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from data_processing import paths

SR = 8000
WIN = 5 * SR
FOLDERS = {"norgos": "01-agos-american-northern-goshawk", "grgowl": "02-ggow-great-gray-owl",
           "flaowl": "03-flow-flammulated-owl", "borowl": "04-boow-boreal-owl",
           "brdowl": "05-baow-bdow-barred-owl"}
ORDER = ["flaowl", "grgowl", "norgos", "brdowl", "borowl"]  # column order of target outputs


def load_calls():
    """{species: [(source_id, 8 kHz float32 array), ...]} from the cropped clips."""
    out = {}
    for sp, folder in FOLDERS.items():
        items = []
        for f in sorted(glob.glob(str(paths.SYNTHETIC / folder / "cropped" / "*.wav"))):
            y, sr = sf.read(f, dtype="float32", always_2d=True)
            y = torch.from_numpy(y[:, 0].copy())
            y = AF.resample(y, sr, SR).numpy() if sr != SR else y.numpy()
            items.append((Path(f).stem.replace("call-", ""), y[: WIN]))
        out[sp] = items
    return out


def split_sources(items, n_val=1, n_test=1):
    """Hold out whole source recordings so that no call appears in more than one split."""
    return {"train": items[: len(items) - n_val - n_test], "val": items[len(items) - n_val - n_test : len(items) - n_test],
            "test": items[len(items) - n_test :]}


def active_rms(y, frame=400):
    n = len(y) // frame
    e = np.sqrt((y[: n * frame].reshape(n, frame) ** 2).mean(1)) + 1e-9
    act = e > e.max() * 0.1  # frames within 20 dB of the loudest
    return float(np.sqrt((e[act] ** 2).mean()))


def mix(bg, call, snr_db, rng):
    """Place `call` at a random position in the 5 s background `bg` at the requested SNR."""
    c = call[: WIN]
    pos = int(rng.integers(0, WIN - len(c) + 1))
    g = 10 ** (snr_db / 20) * (np.sqrt((bg ** 2).mean()) + 1e-9) / active_rms(c)
    m = bg.copy()
    m[pos : pos + len(c)] += g * c
    return np.clip(m, -1, 1).astype(np.float32)


def read_bg(path, start_s):
    y, _ = sf.read(str(path), start=int(start_s * SR), frames=WIN, dtype="float32")
    return y if y.ndim == 1 else y[:, 0]
