"""Turn per-window ConvNeXT scores (npz from run_convnext) into detections and events."""
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_CODES = ["flaowl", "grgowl", "norgos", "brdowl", "borowl"]  # order of target_p columns


def load_scores(npz_dir, model):
    """Concatenate all per-file npz into one window-level frame (one row per window)."""
    id2label = model.config.id2label
    rows = []
    for f in sorted(Path(npz_dir).glob("*.npz")):
        z = np.load(f)
        df = pd.DataFrame(z["target_p"], columns=[f"p_{c}" for c in TARGET_CODES])
        df.insert(0, "start_s", z["start_s"])
        df.insert(0, "file", f.stem.split("__")[-1] + ".wav")
        df["night"] = "Night Recordings" in f.stem
        df["top1_idx"], df["top1_p"] = z["top_idx"][:, 0], z["top_p"][:, 0]
        df["top1"] = [id2label[i] for i in df["top1_idx"]]
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def baseline_detections(w, thr=0.1):
    """Baseline rule: top-1 over all classes, kept if it is a target and p > thr."""
    d = w[w["top1"].isin(TARGET_CODES) & (w["top1_p"] > thr)]
    return d[["file", "start_s", "top1", "top1_p"]].rename(columns={"top1": "species", "top1_p": "p"})


def merge_events(det, win=5):
    """Merge consecutive 1 s-offset windows of the same file and species into events."""
    det = det.sort_values(["file", "species", "start_s"])
    new = (
        (det["file"] != det["file"].shift())
        | (det["species"] != det["species"].shift())
        | (det["start_s"] != det["start_s"].shift() + 1)
    )
    g = new.cumsum()
    ev = det.groupby(g).agg(
        file=("file", "first"), species=("species", "first"),
        start_s=("start_s", "first"), last_s=("start_s", "last"),
        p_max=("p", "max"), n_windows=("p", "size"),
    )
    ev["end_s"] = ev["last_s"] + win
    return ev.reset_index(drop=True)
