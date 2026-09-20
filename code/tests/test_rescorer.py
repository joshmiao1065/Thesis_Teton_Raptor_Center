import numpy as np

from training.rescorer import gate, threshold_at


def test_threshold_at_flags_the_requested_rate():
    rng = np.random.default_rng(0)
    neg = rng.random(36000)  # 10 hours at 1 s hop
    thr = threshold_at(neg, 2)  # 2 flagged windows per hour = 20 windows
    assert (neg >= thr).sum() == 20


def test_gate_zeroes_scores_where_the_original_probability_is_tiny():
    S = np.array([[0.9, 0.9], [0.9, 0.9]])
    tlogit = np.array([[-10.0, 0.0], [0.0, -10.0]])  # p = 4.5e-5 and 0.5
    G = gate(S, tlogit, 0.001)
    assert G.tolist() == [[0.0, 0.9], [0.9, 0.0]]
