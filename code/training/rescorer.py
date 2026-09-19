"""Low-capacity rescorers on frozen ConvNeXT embeddings, and matched-false-alarm evaluation.

A rescorer maps the 1024-d pooled embedding of a 5 s window to five target probabilities.
Training data: real field windows as negatives (split by day) and call-in-background mixtures
as positives (split by source recording). Evaluation compares scores at equal false-alarm rates,
measured as flagged windows per hour of held-out field audio.
"""

import numpy as np
import torch
import torch.nn as nn

from data_processing.paths import INTERMEDIATE
CODES = ["flaowl", "grgowl", "norgos", "brdowl", "borowl"]
DAYS = {"train": [f"04{d:02d}" for d in range(2, 10)], "val": ["0410", "0411", "0412"], "test": ["0413", "0414", "0415"]}


def load_region(name):
    """Concatenate per-file feature npz -> dict of arrays plus file names and day strings."""
    parts, files = [], []
    for f in sorted((INTERMEDIATE / "features" / name).glob("*.npz")):
        z = np.load(f)
        parts.append({k: z[k] for k in z.files})
        files.append(f.stem.split("__")[-1] + ".wav")
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out["file"] = np.concatenate([[fn] * len(p["start_s"]) for fn, p in zip(files, parts)])
    out["day"] = np.array([f[:4] for f in out["file"]])
    return out


def scores(kind, d, top1_targets):
    """Baseline scores (n, 5) from stored logits: prob, top-1 rule, margin over best non-target."""
    p = 1 / (1 + np.exp(-d["tlogit"]))
    if kind == "prob":
        return p
    if kind == "margin":
        return p - d["maxnon"][:, None]
    if kind == "rule":  # baseline: keep the probability only if this class is the top-1 class
        return np.where(p >= d["maxnon"][:, None], p, 0.0)
    raise ValueError(kind)


class Head(nn.Module):
    def __init__(self, hidden=0, drop=0.2):
        super().__init__()
        self.net = nn.Linear(1024, 5) if not hidden else nn.Sequential(
            nn.Dropout(drop), nn.Linear(1024, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, 5))

    def forward(self, x):
        return self.net(x)


class Rescorer:
    """A trained head plus the feature standardization it needs; predicts (n, 5) probabilities."""

    def __init__(self, net, mu, sd, hidden, device="cuda", site_norm=False):
        self.net, self.mu, self.sd, self.hidden, self.device = net.eval(), mu, sd, hidden, device
        self.site_norm = site_norm

    def predict(self, X, chunk=100000, site_mean=None):
        """site_mean: mean embedding of the recording site (required if the head was trained site-normalized)."""
        if self.site_norm:
            assert site_mean is not None, "this head needs the mean embedding of the site"
        out = []
        for i in range(0, len(X), chunk):
            x = np.asarray(X[i : i + chunk], dtype=np.float32)
            if self.site_norm:
                x = x - site_mean
            x = (torch.as_tensor(x) - torch.as_tensor(self.mu)) / torch.as_tensor(self.sd)
            with torch.no_grad():
                out.append(torch.sigmoid(self.net(x.to(self.device))).cpu().numpy())
        return np.concatenate(out)

    def save(self, path):
        torch.save({"state": self.net.state_dict(), "mu": self.mu, "sd": self.sd, "hidden": self.hidden,
                    "site_norm": self.site_norm}, path)

    @classmethod
    def load(cls, path, device="cuda"):
        c = torch.load(path, weights_only=False)
        net = Head(c["hidden"]).to(device)
        net.load_state_dict(c["state"])
        return cls(net, c["mu"], c["sd"], c["hidden"], device, c.get("site_norm", False))


def train_head(Xn, Mn, Xp, yp, hidden=0, steps=3000, lr=1e-3, wd=1e-3, seed=0, device="cuda", drop=0.2, batch=256,
               site_norm=False):
    """Balanced batches: `batch` negatives (target 0 on the classes they are valid negatives for)
    and `batch` positives (target 1 on their species, 0 elsewhere)."""
    torch.manual_seed(seed)
    mu, sd = Xn.mean(0), Xn.std(0) + 1e-6
    f = lambda a: ((torch.as_tensor(a, dtype=torch.float32) - torch.as_tensor(mu)) / torch.as_tensor(sd)).to(device)
    Xn_t, Xp_t = f(Xn), f(Xp)
    Mn_t = torch.as_tensor(Mn, dtype=torch.float32, device=device)
    yp_t = torch.as_tensor(yp, dtype=torch.long, device=device)
    net = Head(hidden, drop).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for _ in range(steps):
        i = torch.randint(len(Xn_t), (batch,), device=device)
        j = torch.randint(len(Xp_t), (batch,), device=device)
        x = torch.cat([Xn_t[i], Xp_t[j]])
        y = torch.cat([torch.zeros(batch, 5, device=device), nn.functional.one_hot(yp_t[j], 5).float()])
        w = torch.cat([Mn_t[i], torch.ones(batch, 5, device=device)])
        loss = (bce(net(x), y) * w).sum() / w.sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return Rescorer(net, mu, sd, hidden, device, site_norm)


def training_set(d, variant, every=3):
    """Negatives from the training days (every `every`-th window) with a per-class validity mask.

    variant 'clean': drop every window on which the baseline reports any target probability above 0.05.
    variant 'hard': keep them as negatives (they are the suspected false alarms), except windows the
    baseline scores as barred above 0.1, which are not used as barred negatives.
    """
    tr = np.where(np.isin(d["day"], DAYS["train"]))[0][::every]
    p = 1 / (1 + np.exp(-d["tlogit"][tr]))
    mask = np.ones((len(tr), 5), np.float32)
    if variant == "clean":
        mask[(p > 0.05).any(1)] = 0
    elif variant == "hard":
        mask[p[:, 3] > 0.1, 3] = 0
    else:
        raise ValueError(variant)
    return d["emb"][tr].astype(np.float32), mask


def threshold_at(neg, per_hour):
    """Score threshold with `per_hour` flagged windows per hour of audio (1 s hop = 3600 windows/h)."""
    k = max(int(round(len(neg) * per_hour / 3600)), 1)
    return np.partition(neg, -k)[-k]


def call_scores(S, d, calls, pad_lo=4):
    """Max score per manual call over the windows that overlap it. calls: (file, offset_s, dur_s, col)."""
    idx = {}
    for i, (f, s) in enumerate(zip(d["file"], d["start_s"])):
        idx.setdefault(f, {})[int(s)] = i
    out = []
    for f, off, dur, col in calls:
        w = [idx[f][s] for s in range(int(np.floor(off)) - pad_lo, int(np.ceil(off + dur)) + 1) if f in idx and s in idx[f]]
        if w:
            out.append(S[w, col].max())
    return np.array(out)


def near_calls(d, calls, before=9, after=4):
    """Boolean mask of windows around manually confirmed calls (file, offset_s, dur_s, col)."""
    m = np.zeros(len(d["start_s"]), bool)
    for f, o, du, _ in calls:
        m |= (d["file"] == f) & (d["start_s"] >= o - before) & (d["start_s"] <= o + du + after)
    return m


def training_set_multi(regions, variant, every=3, site_norm=False, exclude_calls=None):
    """Negatives pooled over regions. regions: {name: feature dict}; 'murie' is restricted to training days.

    site_norm subtracts each region's own mean embedding (unlabeled audio only).
    exclude_calls: {name: [(file, offset, dur, col), ...]} windows around confirmed calls are not negatives
    for that call's class.
    """
    Xs, Ms = [], []
    means = {}
    for name, d in regions.items():
        idx = np.arange(len(d["start_s"]))
        means[name] = d["emb"].astype(np.float32).mean(0) if site_norm else None
        sel = idx[np.isin(d["day"], DAYS["train"])] if name == "murie" else idx
        sel = sel[::every]
        p = 1 / (1 + np.exp(-d["tlogit"][sel]))
        mask = np.ones((len(sel), 5), np.float32)
        if variant == "clean":
            mask[(p > 0.05).any(1)] = 0
        elif variant == "hard":
            mask[p[:, 3] > 0.1, 3] = 0
        if exclude_calls and name in exclude_calls:
            for col in {c[3] for c in exclude_calls[name]}:
                nc = near_calls(d, [c for c in exclude_calls[name] if c[3] == col])[sel]
                mask[nc, col] = 0
        x = d["emb"][sel].astype(np.float32)
        Xs.append(x - means[name] if site_norm else x)
        Ms.append(mask)
    return np.concatenate(Xs), np.concatenate(Ms), means


def gate(S, tlogit, g):
    """Gated rescorer: keep a head score only where the original model's probability for the same
    class exceeds g (the head re-ranks candidates; it cannot create detections from nothing)."""
    return S * (1 / (1 + np.exp(-tlogit)) > g)
