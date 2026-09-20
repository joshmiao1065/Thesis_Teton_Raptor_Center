"""Score every 3 s window of a set of recordings with BirdNET v2.4 and save per-file npz.

Saves, per window: the five target-class probabilities (same column order as ConvNeXT's
`TARGET_CODES`), and the top-5 classes/probabilities over all 6,522 BirdNET classes (sigmoid,
sensitivity 1.0). Resumable: files with an existing output are skipped. Run with `.venv-birdnet`
(needs the CUDA libraries on LD_LIBRARY_PATH for the GPU, see the environment notes).

    python -m evaluation.run_birdnet --root <dir with wavs> --out <dir> [--limit N] [--overlap 2]
"""
import argparse
import json
import time
from pathlib import Path

import birdnet
import numpy as np

TARGET_NAMES = [  # order = evaluation.events.TARGET_CODES (flaowl, grgowl, norgos, brdowl, borowl)
    "Psiloscops flammeolus_Flammulated Owl",
    "Strix nebulosa_Great Gray Owl",
    "Accipiter gentilis_Northern Goshawk",
    "Strix varia_Barred Owl",
    "Aegolius funereus_Boreal Owl",
]


def out_name(f, root):
    return f.relative_to(root).with_suffix("").as_posix().replace("/", "__") + ".npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--overlap", type=float, default=0.0, help="seconds of window overlap (hop = 3 - overlap)")
    ap.add_argument("--per-call", type=int, default=4, help="files per predict call")
    ap.add_argument("--device", default="GPU:0")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(root.rglob("*.wav"))
    todo = [f for f in files if not (out / out_name(f, root)).exists()]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(files)} wavs, {len(todo)} to do", flush=True)

    model = birdnet.load("acoustic", "2.4", "pb")
    labels = list(model.species_list)
    tidx = np.array([labels.index(n) for n in TARGET_NAMES])
    (out / "_targets.json").write_text(json.dumps({"names": TARGET_NAMES, "class_idx": tidx.tolist()}))
    t_all = time.time()
    for i in range(0, len(todo), a.per_call):
        chunk = todo[i : i + a.per_call]
        t0 = time.time()
        r = model.predict(chunk, top_k=None, n_workers=1, batch_size=32, overlap_duration_s=a.overlap,
                          default_confidence_threshold=None, device=a.device)
        probs, ids = np.asarray(r.species_probs), np.asarray(r.species_ids)  # (files, windows, classes)
        hop = r.hop_duration_s
        order = {Path(str(x)).resolve(): j for j, x in enumerate(r.inputs)}  # results may not follow the input order
        for f in chunk:
            k = order[f.resolve()]
            # the class order differs per window: species_ids says which class each column is
            p = np.empty(probs[k].shape, np.float32)
            np.put_along_axis(p, ids[k].astype(np.int64), probs[k].astype(np.float32), axis=1)
            n_valid = int(np.ceil(float(r.input_durations[k]) / hop))
            p = p[:n_valid]  # multi-file calls pad shorter files
            top = np.argsort(-p, axis=1)[:, :5]
            np.savez_compressed(
                out / out_name(f, root),
                start_s=(np.arange(len(p)) * hop).astype(np.float32),
                target_p=p[:, tidx],
                top_idx=top.astype(np.int16),
                top_p=np.take_along_axis(p, top, 1),
                hop=np.float32(hop),
            )
        print(f"[{min(i + a.per_call, len(todo))}/{len(todo)}] {len(chunk)} files {time.time() - t0:.1f}s "
              f"(total {time.time() - t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
