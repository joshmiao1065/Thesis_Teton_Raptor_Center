import numpy as np
import torch

from training.softmax_head import OTHER, train_head, train_pooled_head, zero_shot_probs


def test_zero_shot_matches_full_softmax():
    rng = np.random.default_rng(0)
    W, b, X = rng.normal(size=(50, 8)), rng.normal(size=50), rng.normal(size=(7, 8))
    tid = [3, 10, 0, 42, 7]
    p = zero_shot_probs(X, W, b, tid)
    z = X @ W.T + b
    full = np.exp(z - z.max(1, keepdims=True))
    full /= full.sum(1, keepdims=True)
    assert np.allclose(p[:, :5], full[:, tid], atol=1e-5)
    assert np.allclose(p.sum(1), 1, atol=1e-5)


def test_warm_start_reproduces_sigmoid_before_training():
    rng = np.random.default_rng(1)
    W5, b5 = rng.normal(size=(5, 8)) * 0.1, rng.normal(size=5) - 4
    X = rng.normal(size=(20, 8))
    head = train_head(X, np.full(20, OTHER), init=(W5, b5), standardize=False, epochs=0)
    z = X @ W5.T + b5
    p = head.predict_proba(X)
    # with the other logit at 0, softmax([z_k, 0, ...]) = sigmoid(z_k) when the remaining targets are negligible
    k = z.argmax(1)
    sig = 1 / (1 + np.exp(-z[np.arange(20), k]))
    assert np.allclose(p[np.arange(20), k], sig, atol=0.05)


def test_learns_separable_classes():
    rng = np.random.default_rng(2)
    centers = rng.normal(size=(6, 16)) * 4
    y = rng.integers(0, 6, 600)
    X = centers[y] + rng.normal(size=(600, 16))
    for hidden in (0, 32):
        head = train_head(X, y, hidden=hidden, epochs=20, batch=64, lr=3e-3)
        assert (head.predict_proba(X).argmax(1) == y).mean() > 0.95


def test_pooled_head_starts_at_zero_shot():
    rng = np.random.default_rng(3)
    W, b, X = rng.normal(size=(40, 8)), rng.normal(size=40), rng.normal(size=(9, 8))
    tid = [5, 1, 30, 12, 0]
    for mode in ("rows", "full"):
        head = train_pooled_head(X, np.full(9, OTHER), W, b, tid, mode=mode, epochs=0)
        assert np.allclose(head.predict_proba(X), zero_shot_probs(X, W, b, tid), atol=1e-5)


def test_pooled_rows_mode_keeps_non_target_rows():
    rng = np.random.default_rng(4)
    W, b = rng.normal(size=(30, 8)), rng.normal(size=30)
    tid = [0, 1, 2, 3, 4]
    y = rng.integers(0, 6, 300)
    X = rng.normal(size=(300, 8)) + np.eye(8)[np.minimum(y, 7)] * 3
    head = train_pooled_head(X, y, W, b, tid, mode="rows", epochs=5, batch=32, lr=1e-2)
    net = head.net
    assert np.allclose(net.wr.numpy(), W[5:]) and np.allclose(net.br.numpy(), b[5:])
    assert not np.allclose(net.wt.detach().numpy(), W[:5])


def test_pooled_full_mode_learns():
    rng = np.random.default_rng(5)
    W, b = rng.normal(size=(30, 16)) * 0.1, np.zeros(30)
    centers = rng.normal(size=(6, 16)) * 4
    y = rng.integers(0, 6, 600)
    X = centers[y] + rng.normal(size=(600, 16))
    head = train_pooled_head(X, y, W, b, [0, 1, 2, 3, 4], mode="full", epochs=20, batch=64, lr=3e-2, l2sp=0.0)
    assert (head.predict_proba(X).argmax(1) == y).mean() > 0.9


def test_pooled_sigmoid_objective_moves_only_target_rows():
    rng = np.random.default_rng(6)
    W, b = rng.normal(size=(30, 8)) * 0.1, rng.normal(size=30) - 3
    tid = [0, 1, 2, 3, 4]
    y = rng.integers(0, 6, 400)
    X = rng.normal(size=(400, 8)) + np.eye(8)[np.minimum(y, 7)] * 3
    head = train_pooled_head(X, y, W, b, tid, mode="rows", objective="sigmoid", epochs=10, batch=64, lr=3e-2,
                             l2sp=0.0)
    net = head.net
    assert np.allclose(net.wr.numpy(), W[5:])
    zt = (X @ net.wt.detach().numpy().T + net.bt.detach().numpy())
    pos = y < 5
    assert (zt[pos].argmax(1) == y[pos]).mean() > 0.8
    rm = net.rest_max(torch.as_tensor(X, dtype=torch.float32)).numpy()
    assert np.allclose(rm, (X @ W[5:].T + b[5:]).max(1), atol=1e-5)
