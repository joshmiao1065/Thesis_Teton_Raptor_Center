"""Six-class softmax heads on ConvNeXT pooled embeddings: the five targets plus one "other" class.

ConvNeXT-Base-BirdSet-XCL ends in one linear layer (pooled 1024-d embedding -> 9,736 logits) read
through a per-class sigmoid. These heads replace that read-out with a softmax over six classes, so the
targets compete with an explicit "other" class trained on verified false positives and background.

- `zero_shot_probs`: no training; softmax over all 9,736 pretrained logits, non-targets pooled into "other".
- `train_head`: a linear (or one-hidden-layer) softmax head. Options: per-sample weights, label smoothing,
  embedding mixup, and a warm start from the pretrained target rows with an L2-SP penalty that keeps the
  weights near them. With the "other" logit at 0, softmax([z, 0]) = sigmoid(z), so the warm-started head
  begins at the pretrained per-class probabilities.
"""
import numpy as np
import torch
from torch import nn

N_CLASSES = 6
OTHER = 5


def zero_shot_probs(X, W, b, target_idx, chunk=20000, device="cpu"):
    """(n, 6): softmax over all pretrained classes; columns = targets in `target_idx` order, then "other"."""
    target_idx = np.asarray(target_idx)
    rest = np.setdiff1d(np.arange(W.shape[0]), target_idx)
    Wt = torch.as_tensor(W, dtype=torch.float32, device=device)
    bt = torch.as_tensor(b, dtype=torch.float32, device=device)
    out = []
    for i in range(0, len(X), chunk):
        z = torch.as_tensor(np.asarray(X[i : i + chunk], np.float32), device=device) @ Wt.T + bt
        lse = torch.logsumexp(z, 1, keepdim=True)
        p_t = torch.exp(z[:, target_idx] - lse)
        p_o = torch.exp(torch.logsumexp(z[:, rest], 1, keepdim=True) - lse)
        out.append(torch.cat([p_t, p_o], 1).cpu().numpy())
    return np.concatenate(out)


class SoftmaxHead(nn.Module):
    def __init__(self, dim=1024, hidden=0, drop=0.2):
        super().__init__()
        self.hidden = hidden
        if hidden:
            self.body = nn.Sequential(nn.Dropout(drop), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop))
            self.out = nn.Linear(hidden, N_CLASSES)
        else:
            self.body = nn.Identity()
            self.out = nn.Linear(dim, N_CLASSES)

    def forward(self, x):
        return self.out(self.body(x))


class Head:
    """A trained head plus the input standardization it needs; `predict_proba` gives (n, 6)."""

    def __init__(self, net, mu, sd):
        self.net, self.mu, self.sd = net.eval(), mu, sd

    def logits(self, X, chunk=50000):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), chunk):
                x = (torch.as_tensor(np.asarray(X[i : i + chunk], np.float32)) - self.mu) / self.sd
                out.append(self.net(x).numpy())
        return np.concatenate(out)

    def predict_proba(self, X):
        z = self.logits(X)
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)


def train_head(X, y, sample_weight=None, hidden=0, epochs=60, batch=512, lr=1e-3, wd=1e-4, label_smoothing=0.0,
               mixup=0.0, init=None, l2sp=0.0, standardize=True, seed=0, threads=None, device="cpu"):
    """Fit a six-class softmax head.

    X (n, d) embeddings, y (n,) in 0..5. `sample_weight` (n,) are sampling probabilities (normalized here),
    so batches follow the weighting. `init=(W5, b5)`: pretrained rows for the five targets (d must be the
    embedding dimension, linear head, no standardization); "other" starts at zero. `l2sp` penalizes the
    squared distance of the output layer from its initial weights. `mixup` = Beta(a, a) parameter (0 = off).
    """
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = np.asarray(X, np.float32)
    if init is not None:
        assert hidden == 0 and not standardize, "a warm start needs the raw embedding and a linear head"
    mu = torch.as_tensor(X.mean(0) if standardize else np.zeros(X.shape[1], np.float32))
    sd = torch.as_tensor(X.std(0) + 1e-6 if standardize else np.ones(X.shape[1], np.float32))
    Xt = ((torch.as_tensor(X) - mu) / sd).to(device)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.long, device=device)
    net = SoftmaxHead(X.shape[1], hidden).to(device)
    if init is not None:
        with torch.no_grad():
            net.out.weight.zero_(); net.out.bias.zero_()
            net.out.weight[:OTHER] = torch.as_tensor(init[0], dtype=torch.float32)
            net.out.bias[:OTHER] = torch.as_tensor(init[1], dtype=torch.float32)
    w0 = [p.detach().clone() for p in net.out.parameters()]
    p = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, np.float64)
    p = torch.as_tensor(p / p.sum(), dtype=torch.float32, device=device)
    steps = epochs * max(1, len(X) // batch)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0 if l2sp else wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    net.train()
    for _ in range(steps):
        i = torch.multinomial(p, batch, replacement=True)
        x, t = Xt[i], nn.functional.one_hot(yt[i], N_CLASSES).float()
        if mixup:
            lam = float(rng.beta(mixup, mixup))
            j = torch.randperm(batch, device=device)
            x, t = lam * x + (1 - lam) * x[j], lam * t + (1 - lam) * t[j]
        loss = ce(net(x), t)
        if l2sp:
            loss = loss + l2sp * sum(((a - a0) ** 2).sum() for a, a0 in zip(net.out.parameters(), w0))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return Head(net.cpu(), mu, sd)
