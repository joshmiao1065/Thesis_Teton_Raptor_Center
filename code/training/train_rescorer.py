"""Train the rescorer heads and save them.

    python -m training.train_rescorer --tag v2                       # Murie negatives only
    python -m training.train_rescorer --tag multi --sites murie,riverview,taylor,butler [--site-norm]
    python -m training.train_rescorer --tag loso_butler --sites murie,riverview,taylor --site-norm

Heads: {clean, hard} negatives x {linear, MLP}. Positives are the call-in-background mixtures
(Murie backgrounds); mixture embeddings are centered with the Murie mean when --site-norm is set.
"""
import argparse
import time

import numpy as np
import pandas as pd

from data_processing.paths import INTERMEDIATE, KALEIDOSCOPE
from training.rescorer import load_region, train_head, training_set_multi


def confirmed_goshawk(d):
    tk = pd.read_csv(next(KALEIDOSCOPE.glob("Taylor*/cluster.csv")))
    tk = tk[tk["MANUAL ID"].astype(str).str.startswith("NOGO")]
    have = set(d["file"])
    return [(f, o, du, 2) for f, o, du in zip(tk["IN FILE"], tk["OFFSET"], tk["DURATION"]) if f in have]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--sites", default="murie")
    ap.add_argument("--site-norm", action="store_true")
    a = ap.parse_args()
    regions = {s: load_region(s) for s in a.sites.split(",")}
    excl = {"taylor": confirmed_goshawk(regions["taylor"])} if "taylor" in regions else None
    mix = np.load(INTERMEDIATE / "mixtures/train.npz")
    Xp, yp = mix["emb"].astype(np.float32), mix["species"].astype(int)
    out = INTERMEDIATE / "rescorer" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    for variant in ("clean", "hard"):
        Xn, mask, means = training_set_multi(regions, variant, site_norm=a.site_norm, exclude_calls=excl)
        Xpp = Xp - means["murie"] if a.site_norm else Xp
        for hidden, kind in ((0, "lin"), (256, "mlp")):
            t0 = time.time()
            head = train_head(Xn, mask, Xpp, yp, hidden=hidden, steps=a.steps, seed=a.seed, site_norm=a.site_norm)
            head.save(out / f"{variant}_{kind}.pt")
            print(f"{a.tag} {variant}_{kind}: {len(Xn)} negatives from {list(regions)}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
