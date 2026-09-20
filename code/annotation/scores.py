"""Per-clip model scores for the annotation app, computed from the stored per-window outputs.

    python -m annotation.scores --dir annotation

For every manifest item that comes from a field recording, `scores.json` holds the ConvNeXT windows
(5 s, 1 s hop) and BirdNET windows (3 s) overlapping the shaded region: the five target logits and the
top classes. ConvNeXT target logits are stored directly; BirdNET's are the inverse sigmoid of the saved
probabilities (equal to the raw logits to about 1e-4). The app only shows them on request.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from data_processing.paths import INTERMEDIATE

CODES = ["flaowl", "grgowl", "norgos", "brdowl", "borowl"]  # column order of the stored target arrays


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def item_scores(it, cx_labels, bn_labels):
    site, stem = it["site"], it["rel"].replace("/", "__").removesuffix(".wav")
    c0 = it["clip"][0]
    a, b = c0 + it["focal"][0], c0 + it["focal"][1]
    out = dict(cx=[], bn=[])
    f = INTERMEDIATE / "features" / site / f"{stem}.npz"
    if f.exists():
        z = np.load(f)
        for i in np.where((z["start_s"] < b) & (z["start_s"] + 5 > a))[0]:
            top1 = float(logit(z["top1_p"][i]))
            out["cx"].append(dict(t=float(z["start_s"][i] - c0), logits=dict(zip(CODES, z["tlogit"][i].round(2).tolist())),
                                  top=[[cx_labels[str(int(z["top1_idx"][i]))], round(top1, 2)]]))
    g = INTERMEDIATE / f"birdnet_{site}" / f"{stem}.npz"
    if g.exists():
        z = np.load(g)
        for i in np.where((z["start_s"] < b) & (z["start_s"] + 3 > a))[0]:
            lg = logit(z["target_p"][i]).round(2).tolist()
            top = [[bn_labels[int(k)].split("_")[-1], round(float(logit(p)), 2)] for k, p in zip(z["top_idx"][i][:3], z["top_p"][i][:3])]
            out["bn"].append(dict(t=float(z["start_s"][i] - c0), logits=dict(zip(CODES, lg)), top=top))
    return out


def compute_scores(folder):
    from transformers import AutoConfig

    from models.convnext import MODEL_ID

    folder = Path(folder)
    man = json.loads((folder / "manifest.json").read_text())
    cx_labels = {str(k): v for k, v in AutoConfig.from_pretrained(MODEL_ID).id2label.items()}
    bn_labels = json.loads((folder / "birdnet_labels.json").read_text())
    scores = {it["id"]: item_scores(it, cx_labels, bn_labels)
              for grp in ("queue", "reference", "practice", "examples") for it in man.get(grp, []) if "rel" in it}
    (folder / "scores.json").write_text(json.dumps(scores))
    return scores


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    n = len(compute_scores(ap.parse_args().dir))
    print(n, "items with scores")
