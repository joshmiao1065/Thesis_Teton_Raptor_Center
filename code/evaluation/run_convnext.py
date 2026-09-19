"""Score every 5 s window (1 s hop) of a set of recordings with ConvNeXT and save per-file npz.

Saves, per window: the five target-class probabilities and the top-10 classes/probabilities
(sigmoid over all 9,736 classes). Resumable: files with an existing output are skipped.

    python -m evaluation.run_convnext --root <dir with wavs> --out <dir> [--limit N]
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from data_processing.audio import load_32k, windows
from models.convnext import Preprocessor, load_model, target_ids


def score_file(model, prep, tids, y, batch=32, device="cuda"):
    w = windows(y)
    n = w.shape[0]
    tp = np.zeros((n, len(tids)), np.float32)
    ti = np.zeros((n, 10), np.int16)
    tv = np.zeros((n, 10), np.float32)
    for i in range(0, n, batch):
        x = prep(w[i : i + batch].to(device))
        with torch.no_grad():
            p = torch.sigmoid(model(x).logits)
        v, ix = torch.topk(p, 10)
        tp[i : i + batch] = p[:, tids].cpu().numpy()
        ti[i : i + batch] = ix.cpu().numpy()
        tv[i : i + batch] = v.cpu().numpy()
    return dict(start_s=np.arange(n, dtype=np.int32), target_p=tp, top_idx=ti, top_p=tv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-quantize", action="store_true")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(root.rglob("*.wav"))
    todo = [f for f in files if not (out / (f.relative_to(root).with_suffix("").as_posix().replace("/", "__") + ".npz")).exists()]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(files)} wavs, {len(todo)} to do", flush=True)

    model, prep = load_model(), Preprocessor().to("cuda")
    tids = target_ids(model)
    load = lambda f: load_32k(str(f), quantize16=not a.no_quantize)
    with ThreadPoolExecutor(1) as ex:
        nxt = ex.submit(load, todo[0]) if todo else None
        for k, f in enumerate(todo):
            t0 = time.time()
            y = nxt.result()
            nxt = ex.submit(load, todo[k + 1]) if k + 1 < len(todo) else None
            res = score_file(model, prep, tids, y)
            name = f.relative_to(root).with_suffix("").as_posix().replace("/", "__")
            np.savez_compressed(out / (name + ".npz"), **res)
            print(f"[{k + 1}/{len(todo)}] {name} {len(res['start_s'])} windows {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
