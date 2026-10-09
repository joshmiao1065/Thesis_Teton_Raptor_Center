"""Cross-validated discriminant-direction gate on frozen embeddings.

A "gate" here is a single direction in embedding space, built with plain vector arithmetic (no
gradient training): `direction = mean(real calls) [- mean(false positives)]`. Scoring a new
embedding is a dot product with that direction. This is deliberately low-capacity compared to a
trained head (`training/rescorer.py`): the only free choice is a decision threshold, so it is much
harder to overfit to the specific false positives used to build it, and cross-validation checks that
directly by rebuilding the direction from a different subset each fold and scoring only what that
fold's direction never saw.
"""
import numpy as np
import pandas as pd


def build_direction(real_vecs, false_vecs=None):
    """A unit-norm direction: `mean(real_vecs)`, or `mean(real_vecs) - mean(false_vecs)` if
    `false_vecs` is given. Without `false_vecs` this is a plain nearest-prototype direction (how
    similar something is to typical real calls); with it, a discriminant direction that also points
    away from typical false positives, which generally separates the two groups better (see
    `training/01_rescorer_evaluation.py`'s sibling `training/06_discriminant_cv.py` notebook)."""
    d = real_vecs.mean(0)
    if false_vecs is not None:
        d = d - false_vecs.mean(0)
    return d / (np.linalg.norm(d) + 1e-12)


def score(vecs, direction):
    """Dot product of each embedding with a (unit-norm) `direction`. Embeddings are used as given,
    not separately normalized, so an embedding's own magnitude also contributes to its score."""
    return vecs @ direction


def fold_id(n, k_folds, seed):
    """A length-`n` array of fold ids in [0, k_folds), one random assignment per `seed`."""
    rng = np.random.default_rng(seed)
    return rng.permutation(n) % k_folds


def tp_bounds(keep, ver_correct, ver_total):
    """Bounds on how many of the detections `keep` marks as kept are actually true positives, given
    only a verification sheet's reported correct/total counts per group (not per-detection labels).

    `keep`, `ver_correct`, `ver_total` are same-length arrays: `ver_correct[i]`/`ver_total[i]` are the
    correct/total counts of whichever sheet row detection `i` belongs to (several detections typically
    share a row, hence a group-by). For each group with `C` correct of `N` total, and `n` of them
    actually tested: even in the worst case (every one of the sheet's `N-C` globally-incorrect
    detections happened to be tested and kept), at least `kk - (N-C)` of the `kk` kept detections must
    be correct (`lo`); at most `min(kk, C)` can be (`hi`, can't exceed what was kept or the sheet's
    global correct count). `avail_lo`/`avail_hi` apply the same bound to every tested detection
    (`n`) regardless of `keep`, the ceiling `lo`/`hi` could ever reach when the tested group is only
    part of the sheet's full population (`n < N`).
    """
    lo = hi = avail_lo = avail_hi = 0
    df = pd.DataFrame({"k": np.asarray(keep, bool), "C": ver_correct, "N": ver_total})
    for (C, N), g in df.groupby(["C", "N"]):
        C, N, n, kk = int(C), int(N), len(g), int(g.k.sum())
        lo += max(0, kk - (N - C))
        hi += min(kk, C)
        avail_lo += max(0, n - (N - C))
        avail_hi += min(n, C)
    return lo, hi, avail_lo, avail_hi


def cross_validate_direction(real_vecs, false_vecs, k_folds=5, n_seeds=10, pct_thresholds=(35, 50, 65, 80),
                             external_vecs=None, external_ver_correct=None, external_ver_total=None):
    """How well does a direction built from (most of) `real_vecs` and (most of) `false_vecs` separate
    held-out false positives from real calls, averaged over `k_folds` x `n_seeds` independent splits?

    The FALSE side is always split and held out: a direction is built from `k_folds - 1` folds of
    `false_vecs` and scored only on the remaining fold, so no reported number ever comes from a
    direction that saw the false positives it is being judged on.

    The REAL side is evaluated one of two ways:
      - if `external_vecs` is given (a recall-check set independent of `real_vecs`, e.g. confirmed
        calls or detections from a site/sheet row not used to build the direction), the full
        `real_vecs` pool builds the direction every fold and recall is measured on `external_vecs`,
        bounded via `tp_bounds` if `external_ver_correct`/`external_ver_total` are given (else every
        row of `external_vecs` is assumed a true positive and recall is exact, not bounded).
      - otherwise, `real_vecs` is ALSO split and held out each fold the same way as `false_vecs`, and
        recall is the fraction of that fold's held-out real slice scoring above threshold -- weaker,
        since it only checks the direction is not overfitting to the `real_vecs` it was shown, not
        that `real_vecs` is itself a trustworthy stand-in for real-world positives.

    Returns a long DataFrame, one row per (seed, fold, pct threshold): `fp_removed_pct`, and either
    `real_kept_pct` (exact) or `real_kept_lo_pct`/`real_kept_hi_pct` (bounded).
    """
    use_external = external_vecs is not None
    rows = []
    for seed in range(n_seeds):
        for fold in range(k_folds):
            f_test_mask = fold_id(len(false_vecs), k_folds, seed) == fold
            f_train, f_test = false_vecs[~f_test_mask], false_vecs[f_test_mask]
            if use_external:
                r_train, r_test = real_vecs, None
            else:
                r_test_mask = fold_id(len(real_vecs), k_folds, seed) == fold
                r_train, r_test = real_vecs[~r_test_mask], real_vecs[r_test_mask]

            direction = build_direction(r_train, f_train)
            s_false = score(f_test, direction)
            s_pos = score(external_vecs, direction) if use_external else score(r_test, direction)
            combined = np.concatenate([s_false, s_pos])
            for pct in pct_thresholds:
                thr = np.percentile(combined, pct)
                row = dict(seed=seed, fold=fold, pct=pct, n_false_test=len(s_false),
                          fp_removed_pct=float((s_false < thr).mean() * 100))
                if use_external and external_ver_correct is not None:
                    keep = s_pos >= thr
                    lo, hi, _avail_lo, avail_hi = tp_bounds(keep, external_ver_correct, external_ver_total)
                    tot = int(np.asarray(external_ver_correct)[0])
                    row.update(real_kept_lo_pct=100 * lo / tot, real_kept_hi_pct=100 * hi / tot,
                              real_kept_lo_pct_of_achievable=100 * lo / max(avail_hi, 1),
                              real_kept_hi_pct_of_achievable=100 * hi / max(avail_hi, 1))
                else:
                    row.update(real_kept_pct=float((s_pos >= thr).mean() * 100), n_real_test=len(s_pos))
                rows.append(row)
    return pd.DataFrame(rows)
