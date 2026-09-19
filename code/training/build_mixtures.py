"""Generate call-in-background mixtures and extract their ConvNeXT features.

Writes intermediate/mixtures/<split>.npz with embeddings, target logits, top-1 info, species
index (-1 for pure background), snr and source id. Splits: Murie days (files) and call sources
are held out separately.

    python -m training.build_mixtures --n-train 3000 --n-eval 500
"""
import argparse

import numpy as np
import torch

from data_processing import paths
from data_processing.audio import gpu_resample
from data_processing.mixtures import ORDER, SR, load_calls, mix, read_bg, split_sources
from evaluation.extract_features import extract_windows
from models.convnext import Preprocessor, load_model, target_ids

FOREST = paths.MURIE_AUDIO
DAYS = {"train": [f"04{d:02d}" for d in range(2, 10)], "val": ["0410", "0411", "0412"], "test": ["0413", "0414", "0415"]}


def allowed_backgrounds(split):
    """(path, second) pairs whose baseline scores show no target above 0.05 in the window."""
    out = []
    for z in sorted((paths.INTERMEDIATE / "convnext_murie").glob("*.npz")):
        name = z.stem.split("__")[-1] + ".wav"
        if name[:4] not in DAYS[split]:
            continue
        p = FOREST / name if (FOREST / name).exists() else FOREST / "Night Recordings" / name
        ok = np.load(z)["target_p"].max(1) < 0.05
        out += [(p, int(s)) for s in np.where(ok)[0][::5]]
    return out


def site_backgrounds(site, every=5):
    """(path, second) pairs at another site whose baseline scores show no target above 0.05.
    Windows around manually confirmed calls (goshawk at Taylor) are excluded."""
    root = paths.REGION_AUDIO[site]
    files = {p.name: p for p in root.rglob("*.wav")}
    bad = {}
    if site == "taylor":
        import pandas as pd

        tk = pd.read_csv(next(paths.KALEIDOSCOPE.glob("Taylor*/cluster.csv")))
        for f, o, du in zip(tk["IN FILE"], tk["OFFSET"], tk["DURATION"]):
            if str(tk.loc[tk["IN FILE"] == f, "MANUAL ID"].iloc[0]).startswith("NOGO"):
                bad.setdefault(f, []).append((o - 9, o + du + 4))
    out = []
    for z in sorted((paths.INTERMEDIATE / "features" / site).glob("*.npz")):
        name = z.stem.split("__")[-1] + ".wav"
        p = np.load(z)["tlogit"]
        ok = (1 / (1 + np.exp(-p))).max(1) < 0.05
        for s in np.where(ok)[0][::every]:
            if any(lo <= s <= hi for lo, hi in bad.get(name, [])):
                continue
            out.append((files[name], int(s)))
    return out


def build_multi(n_per_species, snr, calls, model, prep, tids):
    """Training-only mixtures on other sites' backgrounds (species x site), saved with a site label."""
    rng = np.random.default_rng(7)
    rows, sites = [], []
    for site in ("riverview", "taylor", "butler"):
        bgs = site_backgrounds(site)
        print(site, len(bgs), "background windows", flush=True)
        for sp in ORDER:
            srcs = split_sources(calls[sp])["train"]
            for _ in range(n_per_species):
                sid, c = srcs[int(rng.integers(len(srcs)))]
                p, s = bgs[int(rng.integers(len(bgs)))]
                sn = float(rng.uniform(*snr))
                rows.append((ORDER.index(sp), sid, sn, mix(read_bg(p, s), c, sn, rng)))
                sites.append(site)
    wav = torch.from_numpy(np.stack([r[3] for r in rows]))
    parts = [extract_windows(model, prep, tids, gpu_resample(wav[i : i + 512], SR, 32000)) for i in range(0, len(wav), 512)]
    f = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    f.pop("start_s")
    np.savez(paths.INTERMEDIATE / "mixtures" / "train_multi.npz", species=np.array([r[0] for r in rows], np.int8),
             source=np.array([r[1] for r in rows]), snr=np.array([r[2] for r in rows], np.float32),
             site=np.array(sites), **f)
    print("multi-site mixtures saved:", len(rows), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--snr", type=float, nargs=2, default=[-20, 5])
    ap.add_argument("--multi", type=int, default=0, help="only build N mixtures per species per other site")
    a = ap.parse_args()
    calls = load_calls()
    model, prep = load_model(), Preprocessor().cuda()
    tids = target_ids(model)
    if a.multi:
        build_multi(a.multi, a.snr, calls, model, prep, tids)
        return
    out_dir = paths.INTERMEDIATE / "mixtures"; out_dir.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", a.n_train), ("val", a.n_eval), ("test", a.n_eval)):
        rng = np.random.default_rng({'train': 1, 'val': 2, 'test': 3}[split])
        bgs = allowed_backgrounds(split)
        print(split, len(bgs), "background windows", flush=True)
        rows = []
        for sp in ORDER:
            srcs = split_sources(calls[sp])[split]
            for _ in range(n):
                sid, c = srcs[int(rng.integers(len(srcs)))]
                p, s = bgs[int(rng.integers(len(bgs)))]
                snr = float(rng.uniform(*a.snr))
                rows.append((ORDER.index(sp), sid, snr, mix(read_bg(p, s), c, snr, rng)))
        wav = torch.from_numpy(np.stack([r[3] for r in rows]))
        y32 = torch.cat([gpu_resample(wav[i : i + 64], SR, 32000) for i in range(0, len(wav), 64)])
        f = extract_windows(model, prep, tids, y32)
        f.pop("start_s")
        np.savez(out_dir / f"{split}.npz", species=np.array([r[0] for r in rows], np.int8),
                 source=np.array([r[1] for r in rows]), snr=np.array([r[2] for r in rows], np.float32), **f)
        print(split, "saved", len(rows), flush=True)


if __name__ == "__main__":
    main()
