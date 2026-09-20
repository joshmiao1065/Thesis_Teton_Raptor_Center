"""Score the synthetic clips at full band and after a 32 -> 8 -> 32 kHz round trip (H4 test).

Writes results/bandlimit/clip_scores.csv; the notebook 02_bandlimit_synthetic plots it.
"""
import re

import numpy as np
import pandas as pd
import torch

from data_processing.audio import STEP, WINDOW, gpu_resample, read_wav
from data_processing.paths import RESULTS
from data_processing.paths import SYNTHETIC as SYN
from models.convnext import TARGETS, Preprocessor, load_model, target_ids

OUT = RESULTS / "bandlimit"; OUT.mkdir(parents=True, exist_ok=True)
FOLDER2CODE = {"01-agos-american-northern-goshawk": "norgos", "02-ggow-great-gray-owl": "grgowl",
               "03-flow-flammulated-owl": "flaowl", "04-boow-boreal-owl": "borowl",
               "05-baow-bdow-barred-owl": "brdowl"}
CODES = list(TARGETS)  # column order of target_p

model, prep = load_model(), Preprocessor().cuda()
tids = target_ids(model)
id2label = model.config.id2label


@torch.no_grad()
def score(y32, batch=32):
    w = y32.unfold(0, WINDOW, STEP)
    tp, top1, top1p = [], [], []
    for i in range(0, len(w), batch):
        p = torch.sigmoid(model(prep(w[i:i + batch])).logits)
        v, ix = p.max(1)
        tp.append(p[:, tids].cpu()); top1.append(ix.cpu()); top1p.append(v.cpu())
    return torch.cat(tp).numpy(), torch.cat(top1).numpy(), torch.cat(top1p).numpy()


def band_limit(y32):
    return gpu_resample(gpu_resample(y32, 32000, 8000), 8000, 32000)


clips = []
for folder, code in FOLDER2CODE.items():
    for f in sorted((SYN / folder / "generated").glob("*.wav")):
        m = re.search(r"_generated_(forest|gaussian_noise)_1min_snr(-?\d+)", f.name)
        clips.append(dict(path=f, species=code, bg=m.group(1), snr=int(m.group(2)), src=f.name.split("_generated")[0]))
for f in (SYN / "base").glob("*.wav"):
    clips.append(dict(path=f, species="none", bg=f.stem.split("_1min")[0], snr=None, src=f.stem))
print(len(clips), "clips")

rows, per_window = [], []
for k, c in enumerate(clips):
    y, sr = read_wav(c["path"])
    y = gpu_resample(y, sr, 32000) if sr != 32000 else y.cuda()
    for cond, yy in (("full", y), ("8khz", band_limit(y))):
        tp, t1, t1p = score(yy)
        lab = np.array([id2label[i] for i in t1])
        det = np.isin(lab, CODES) & (t1p > 0.1)
        true_col = CODES.index(c["species"]) if c["species"] in CODES else None
        rows.append(dict(**{k_: v for k_, v in c.items() if k_ != "path"}, cond=cond,
                         max_true=(tp[:, true_col].max() if true_col is not None else np.nan),
                         det_true=(bool((det & (lab == c["species"])).any()) if true_col is not None else False),
                         det_other=int((det & (lab != c["species"])).sum()),
                         det_any_windows=int(det.sum()), n_windows=len(det),
                         top1_nontarget_frac=float((~np.isin(lab, CODES)).mean()),
                         **{f"max_{cd}": tp[:, i].max() for i, cd in enumerate(CODES)}))
    if k % 50 == 0:
        print(k, flush=True)
df = pd.DataFrame(rows)
df.to_csv(OUT / "clip_scores.csv", index=False)
