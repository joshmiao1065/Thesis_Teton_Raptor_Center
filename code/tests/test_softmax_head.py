import numpy as np

from training.softmax_head import OTHER, train_head, zero_shot_probs


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
