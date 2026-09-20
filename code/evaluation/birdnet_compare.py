"""Shared helpers to compare BirdNET and ConvNeXT scores at equal amounts of flagged audio.

Windows differ in length and hop between the models, so rates are compared as seconds of flagged
audio per hour (the union of the flagged windows' time spans), and "recall" of a known call is
whether any flagged second overlaps it.
"""
import numpy as np
import pandas as pd

from data_processing.paths import INTERMEDIATE, KALEIDOSCOPE
from training.rescorer import CODES


def load_birdnet(tag):
    """BirdNET windows of a scored folder (intermediate/<tag>): file, start_s and p_<code> columns."""
    files, starts, P = [], [], []
    for f in sorted((INTERMEDIATE / tag).glob("*.npz")):
        z = np.load(f)
        files += [f.stem.split("__")[-1] + ".wav"] * len(z["start_s"])
        starts.append(z["start_s"]); P.append(z["target_p"])
    df = pd.DataFrame(np.concatenate(P), columns=[f"p_{c}" for c in CODES])
    df.insert(0, "start_s", np.concatenate(starts))
    df.insert(0, "file", files)
    return df


def manual_calls(folder_pattern, prefix):
    """(file, offset, duration) of Kaleidoscope rows whose manual ID starts with prefix."""
    kc = pd.read_csv(next(KALEIDOSCOPE.glob(folder_pattern + "/cluster.csv")))
    col = next(c for c in kc.columns if c.startswith("MANUAL"))
    kc = kc[kc[col].astype(str).str.startswith(prefix)]
    return list(zip(kc["IN FILE"], kc["OFFSET"], kc["DURATION"]))


class Timeline:
    """Concatenated seconds of a set of files; masks mark the seconds covered by flagged windows."""

    def __init__(self, lengths, files):
        self.files = [f for f in files if f in lengths]
        sec = np.ceil([lengths[f] for f in self.files]).astype(int)
        self.base = dict(zip(self.files, np.r_[0, np.cumsum(sec)[:-1]].astype(int)))
        self.len = dict(zip(self.files, sec))
        self.total = int(sec.sum())

    def mask(self, f_arr, s_arr, dur, flag):
        d = np.zeros(self.total + 2, np.int32)
        for f, s in zip(np.asarray(f_arr)[flag], np.asarray(s_arr)[flag]):
            b = self.base.get(f)
            if b is None:
                continue
            d[b + int(s)] += 1
            d[b + min(int(np.ceil(s + dur)), self.len[f])] -= 1
        return np.cumsum(d)[: self.total] > 0

    def per_hour(self, mask):
        return mask.sum() / (self.total / 3600)

    def recall(self, mask, calls):
        calls = [c for c in calls if c[0] in self.base]
        hit = sum(mask[self.base[f] + int(o): self.base[f] + min(int(np.ceil(o + d)) + 1, self.len[f])].any() for f, o, d in calls)
        return hit / len(calls) if calls else np.nan

    def threshold(self, f_arr, s_arr, dur, score, rate, iters=22):
        """Score threshold at which the flagged audio is `rate` seconds per hour on this timeline."""
        lo, hi = 0.0, 1.0
        for _ in range(iters):
            mid = (lo + hi) / 2
            r = self.per_hour(self.mask(f_arr, s_arr, dur, score > mid))
            lo, hi = (mid, hi) if r > rate else (lo, mid)
        return hi


def overlap_max(src_files, src_starts, src_dur, tgt_files, tgt_starts, tgt_dur, tgt_p):
    """For each source window, the maximum of tgt_p over the target windows overlapping it (0 if none)."""
    tgt_files, tgt_starts, tgt_p = np.asarray(tgt_files), np.asarray(tgt_starts), np.asarray(tgt_p)
    by = {f: (tgt_starts[tgt_files == f], tgt_p[tgt_files == f]) for f in set(src_files)}
    out = np.zeros(len(src_files), np.float32)
    for i, (f, s) in enumerate(zip(src_files, src_starts)):
        ts, tp = by[f]
        sel = (ts < s + src_dur) & (ts + tgt_dur > s)
        out[i] = tp[sel].max() if sel.any() else 0
    return out
