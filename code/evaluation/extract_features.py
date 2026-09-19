"""Extract ConvNeXT features for every 5 s window (1 s hop) of a folder of 8 kHz recordings.

Per file (npz, resumable): pooled embedding (N, 1024, float16), target-class logits (N, 5),
top-1 class/probability over all classes, largest probability among non-target classes,
sum of all class probabilities. Audio is read with soundfile and resampled to 32 kHz on the GPU.

    python -m evaluation.extract_features --root <dir> --out <dir>
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from data_processing.audio import gpu_resample, read_wav, windows
from models.convnext import Preprocessor, load_model, target_ids


def extract(model, prep, tids, y32, batch=32):
    """Features for every 5 s window (1 s hop) of a long 32 kHz signal."""
    return extract_windows(model, prep, tids, windows(y32), batch)


@torch.no_grad()
def extract_windows(model, prep, tids, w, batch=32):
    """Features for a (N, 160000) matrix of independent 32 kHz windows."""
    n = w.shape[0]
    mask = torch.ones(model.config.num_labels, dtype=torch.bool, device=w.device)
    mask[tids] = False
    out = dict(emb=np.zeros((n, 1024), np.float16), tlogit=np.zeros((n, len(tids)), np.float32),
               top1_idx=np.zeros(n, np.int16), top1_p=np.zeros(n, np.float32),
               maxnon=np.zeros(n, np.float32), psum=np.zeros(n, np.float32))
    for i in range(0, n, batch):
        x = prep(w[i:i + batch])
        emb = model.convnext(x).pooler_output
        logits = model.classifier(emb)
        p = torch.sigmoid(logits)
        v, ix = p.max(1)
        sl = slice(i, i + len(x))
        out["emb"][sl] = emb.half().cpu().numpy()
        out["tlogit"][sl] = logits[:, tids].cpu().numpy()
        out["top1_idx"][sl] = ix.cpu().numpy()
        out["top1_p"][sl] = v.cpu().numpy()
        out["maxnon"][sl] = p[:, mask].max(1).values.cpu().numpy()
        out["psum"][sl] = p.sum(1).cpu().numpy()
    out["start_s"] = np.arange(n, dtype=np.int32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    key = lambda f: f.relative_to(root).with_suffix("").as_posix().replace("/", "__")
    todo = [f for f in sorted(root.rglob("*.wav")) if not (out / (key(f) + ".npz")).exists()]
    print(f"{len(todo)} files to do", flush=True)
    model, prep = load_model(), Preprocessor().cuda()
    tids = target_ids(model)
    for k, f in enumerate(todo):
        t0 = time.time()
        y, sr = read_wav(f)
        y = gpu_resample(y, sr, 32000)
        res = extract(model, prep, tids, y)
        np.savez(out / (key(f) + ".npz"), **res)
        print(f"[{k + 1}/{len(todo)}] {key(f)} {len(res['start_s'])} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
