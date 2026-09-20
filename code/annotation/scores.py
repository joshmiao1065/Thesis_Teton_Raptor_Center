"""Per-clip model scores for the annotation app.

    python -m annotation.scores --dir annotation

For every manifest item that comes from a field recording, `scores.json` holds the ConvNeXT windows
(5 s, 1 s hop) and BirdNET windows (3 s) overlapping the shaded region: the five target logits and the
top-5 classes over all classes with their logits. ConvNeXT is re-run on just those windows (same 32 kHz
GPU resampling and preprocessing as the feature extraction; the stored target logits agree with the
re-run ones, see the printed check); BirdNET's come from its saved per-window outputs as the inverse
sigmoid of the probabilities (equal to the raw logits to about 1e-4). The app only shows them on request.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from data_processing.audio import WINDOW, gpu_resample
from data_processing.paths import INTERMEDIATE
from evaluation.extract_features import (
    extract_windows,  # noqa: F401  (same preprocessing path as the features)
)

CODES = ["flaowl", "grgowl", "norgos", "brdowl", "borowl"]  # column order of the stored target arrays
TOP = 5


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def audio_path(site, rel):
    from annotation.sampling import audio_root

    return audio_root(site) / rel


@torch.no_grad()
def convnext_windows(model, prep, tids, path, a, b, labels):
    """ConvNeXT logits of the 5 s windows (integer start seconds) overlapping [a, b] of a recording."""
    info = sf.info(path)
    dur = info.frames / info.samplerate
    starts = [s for s in range(max(int(np.floor(a - 5)) + 1, 0), int(np.ceil(b))) if s + 5 <= dur]
    if not starts:
        return [], None
    m0 = max(starts[0] - 2, 0)  # margin so that resampling edge effects stay outside the windows
    m1 = min(starts[-1] + 5 + 2, dur)
    y, _ = sf.read(str(path), start=int(m0 * info.samplerate), frames=int((m1 - m0) * info.samplerate), dtype="float32", always_2d=True)
    y32 = gpu_resample(torch.from_numpy(y[:, 0].copy()), info.samplerate, 32000)
    w = torch.stack([y32[int((s - m0) * 32000): int((s - m0) * 32000) + WINDOW] for s in starts])
    logits = model.classifier(model.convnext(prep(w)).pooler_output)
    top = torch.topk(logits, TOP, dim=1)
    out = []
    for i, s in enumerate(starts):
        out.append(dict(start=s, logits=dict(zip(CODES, logits[i, tids].cpu().numpy().round(2).tolist())),
                        top=[[labels[str(int(k))], round(float(v), 2)] for k, v in zip(top.indices[i].cpu(), top.values[i].cpu())]))
    return out, logits[:, tids].cpu().numpy()


def bn_windows(site, stem, a, b, bn_labels):
    g = INTERMEDIATE / f"birdnet_{site}" / f"{stem}.npz"
    if not g.exists():
        return []
    z = np.load(g)
    out = []
    for i in np.where((z["start_s"] < b) & (z["start_s"] + 3 > a))[0]:
        out.append(dict(start=float(z["start_s"][i]), logits=dict(zip(CODES, logit(z["target_p"][i]).round(2).tolist())),
                        top=[[bn_labels[int(k)].split("_")[-1], round(float(logit(p)), 2)] for k, p in zip(z["top_idx"][i][:TOP], z["top_p"][i][:TOP])]))
    return out


def compute_scores(folder):
    from transformers import AutoConfig

    from models.convnext import MODEL_ID, Preprocessor, load_model, target_ids

    folder = Path(folder)
    man = json.loads((folder / "manifest.json").read_text())
    labels = {str(k): v for k, v in AutoConfig.from_pretrained(MODEL_ID).id2label.items()}
    bn_labels = json.loads((folder / "birdnet_labels.json").read_text())
    model, prep = load_model(), Preprocessor().cuda()
    tids = target_ids(model)
    scores, diffs = {}, []
    for grp in ("queue", "reference", "practice", "examples"):
        for it in man.get(grp, []):
            if "rel" not in it:
                continue
            c0 = it["clip"][0]
            a, b = c0 + it["focal"][0], c0 + it["focal"][1]
            stem = it["rel"].replace("/", "__").removesuffix(".wav")
            cx, lg = convnext_windows(model, prep, tids, audio_path(it["site"], it["rel"]), a, b, labels)
            f = INTERMEDIATE / "features" / it["site"] / f"{stem}.npz"
            if f.exists() and cx:
                z = np.load(f)
                diffs.append(np.abs(lg - z["tlogit"][[w["start"] for w in cx]]).max())
            for w in cx:
                w["t"] = w.pop("start") - c0
            bn = bn_windows(it["site"], stem, a, b, bn_labels)
            for w in bn:
                w["t"] = w.pop("start") - c0
            scores[it["id"]] = dict(cx=cx, bn=bn)
    (folder / "scores.json").write_text(json.dumps(scores))
    print(f"re-run vs stored ConvNeXT target logits: max abs difference {max(diffs):.3f}, mean of maxima {np.mean(diffs):.4f} over {len(diffs)} clips")
    return scores


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    print(len(compute_scores(ap.parse_args().dir)), "items with scores")
