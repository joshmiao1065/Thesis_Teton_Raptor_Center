
"""Six-class softmax heads on ConvNeXT pooled embeddings: the five targets plus one "other" class.

ConvNeXT-Base-BirdSet-XCL ends in one linear layer (pooled 1024-d embedding -> 9,736 logits) read
through a per-class sigmoid. These heads replace that read-out with a softmax over six classes, so the
targets compete with an explicit "other" class trained on verified false positives and background.

- `zero_shot_probs`: no training; softmax over all 9,736 pretrained logits, non-targets pooled into "other".
- `train_head`: a linear (or one-hidden-layer) softmax head. Options: per-sample weights, label smoothing,
  embedding mixup, and a warm start from the pretrained target rows with an L2-SP penalty that keeps the
  weights near them. With the "other" logit at 0, softmax([z, 0]) = sigmoid(z), so the warm-started head
  begins at the pretrained per-class probabilities.
- `PooledHead` / `train_pooled_head`: the whole pretrained layer kept, read through the same six-way softmax as
  `zero_shot_probs` (target logits; "other" = logsumexp of every non-target logit), then fine-tuned: only the
  five target rows (`mode="rows"`, the non-target rows frozen) or the whole layer (`mode="full"`), with L2-SP
  to the pretrained weights. Before training it equals `zero_shot_probs`. `objective="sigmoid"` trains the same
  layer with the original per-class sigmoid loss instead (binary cross-entropy on the five target logits; "other"
  = all five negative), the control for whether the softmax itself matters. `PooledHead.rest_max` gives the
  largest non-target logit, so a fine-tuned layer can also be read through the original top-1 rule.
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


class PooledHead(nn.Module):
    """Pretrained layer (W, b) -> six logits: the targets' (in `target_idx` order), then logsumexp of the rest."""

    def __init__(self, W, b, target_idx, mode="rows"):
        super().__init__()
        assert mode in ("rows", "full")
        tid = np.asarray(target_idx)
        rest = np.setdiff1d(np.arange(W.shape[0]), tid)
        W, b = torch.as_tensor(W, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32)
        self.mode = mode
        self.register_buffer("tid", torch.as_tensor(tid))
        self.register_buffer("rest", torch.as_tensor(rest))
        if mode == "rows":
            self.wt, self.bt = nn.Parameter(W[tid].clone()), nn.Parameter(b[tid].clone())
            self.register_buffer("wr", W[rest].clone())
            self.register_buffer("br", b[rest].clone())
        else:
            self.w, self.b = nn.Parameter(W.clone()), nn.Parameter(b.clone())

    def forward(self, x):
        if self.mode == "rows":
            zt = x @ self.wt.T + self.bt
            zo = torch.logsumexp(x @ self.wr.T + self.br, 1, keepdim=True)
        else:
            z = x @ self.w.T + self.b
            zt, zo = z[:, self.tid], torch.logsumexp(z[:, self.rest], 1, keepdim=True)
        return torch.cat([zt, zo], 1)

    def rest_max(self, x):
        """Largest non-target logit (n,): with the target logits it gives the original top-1 rule."""
        if self.mode == "rows":
            return (x @ self.wr.T + self.br).max(1).values
        return (x @ self.w.T + self.b)[:, self.rest].max(1).values


def _fit(net, Xt, yt, sample_weight, epochs, batch, lr, wd, label_smoothing, mixup, l2sp, rng, device,
         objective="softmax"):
    """Shared loop: weighted sampling with replacement, AdamW + cosine, optional mixup and L2-SP (squared distance
    of every trainable weight from its value at the start). Loss: six-way cross-entropy (`objective="softmax"`) or
    binary cross-entropy on the first five logits with one-hot targets ("sigmoid"; "other" = all negative)."""
    params = [q for q in net.parameters() if q.requires_grad]
    w0 = [q.detach().clone() for q in params]
    p = np.ones(len(Xt)) if sample_weight is None else np.asarray(sample_weight, np.float64)
    p = torch.as_tensor(p / p.sum(), dtype=torch.float32, device=device)
    steps = epochs * max(1, len(Xt) // batch)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0 if l2sp else wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps, 1))
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    net.train()
    for _ in range(steps):
        i = torch.multinomial(p, batch, replacement=True)
        x, t = Xt[i], nn.functional.one_hot(yt[i], N_CLASSES).float()
        if mixup:
            lam = float(rng.beta(mixup, mixup))
            j = torch.randperm(batch, device=device)
            x, t = lam * x + (1 - lam) * x[j], lam * t + (1 - lam) * t[j]
        z = net(x)
        loss = ce(z, t) if objective == "softmax" else \
            nn.functional.binary_cross_entropy_with_logits(z[:, :OTHER], t[:, :OTHER])
        if l2sp:
            loss = loss + l2sp * sum(((a - a0) ** 2).sum() for a, a0 in zip(params, w0))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net


def train_pooled_head(X, y, W, b, target_idx, mode="rows", sample_weight=None, epochs=15, batch=512, lr=3e-4,
                      l2sp=1e-3, objective="softmax", seed=0, device="cpu"):
    """Fine-tune the pretrained layer under the six-way pooled softmax (see `PooledHead`), or under the original
    per-class sigmoid loss (`objective="sigmoid"`, target rows only); raw embeddings in."""
    assert objective in ("softmax", "sigmoid")
    assert objective == "softmax" or mode == "rows", "the sigmoid loss only reaches the target rows"
    torch.manual_seed(seed)
    X = np.asarray(X, np.float32)
    Xt = torch.as_tensor(X, device=device)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.long, device=device)
    net = PooledHead(W, b, target_idx, mode).to(device)
    _fit(net, Xt, yt, sample_weight, epochs, batch, lr, 0.0, 0.0, 0.0, l2sp, np.random.default_rng(seed), device,
         objective)
    d = X.shape[1]
    return Head(net.cpu(), torch.zeros(d), torch.ones(d))


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
    assert not (l2sp and hidden), "L2-SP is defined for the linear head (all its weights are the output layer)"
    _fit(net, Xt, yt, sample_weight, epochs, batch, lr, wd, label_smoothing, mixup, l2sp, rng, device)
    return Head(net.cpu(), mu, sd)
