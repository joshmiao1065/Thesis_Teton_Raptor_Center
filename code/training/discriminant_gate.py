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
from sklearn.model_selection import KFold, StratifiedKFold


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


def _split_ids(n, k, seed, groups=None):
    """Fold assignment for `n` items into `k` folds. Stratified by `groups` (e.g. site, sheet-row
    id) when given, so every fold keeps a proportional mix of sources instead of risking a fold
    that is accidentally all-one-site; falls back to plain random folding otherwise. Stratified
    splitting needs every stratum to have at least `k` members, so a stratum smaller than that is
    merged into an "other" bucket rather than raising."""
    ids = np.empty(n, dtype=int)
    if groups is None:
        splitter, x, y = KFold(n_splits=k, shuffle=True, random_state=seed), np.zeros(n), None
    else:
        groups = np.asarray(groups)
        counts = pd.Series(groups).value_counts()
        rare = counts[counts < k].index
        y = np.where(np.isin(groups, rare), "_other", groups) if len(rare) else groups
        splitter, x = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed), np.zeros(n)
    for fold, (_, test_idx) in enumerate(splitter.split(x, y)):
        ids[test_idx] = fold
    return ids


def _criterion_value(criterion, fp_removed_frac, kept_frac):
    """How good is a threshold, given its false-positive-removal rate and real-call-kept rate on
    held-out inner-validation data? `balanced_accuracy` and `youden_j` rank thresholds identically
    (`youden_j = 2 * balanced_accuracy - 1`); `f1` is the harmonic mean instead of the arithmetic
    mean, so it penalizes a threshold that is great on one axis and poor on the other more harshly."""
    if criterion == "balanced_accuracy":
        return (fp_removed_frac + kept_frac) / 2
    if criterion == "youden_j":
        return fp_removed_frac + kept_frac - 1
    if criterion == "f1":
        denom = fp_removed_frac + kept_frac
        return 0.0 if denom == 0 else 2 * fp_removed_frac * kept_frac / denom
    raise ValueError(f"unknown criterion {criterion!r}")


def _total_correct(ver_correct, ver_total):
    """Sum of a verification sheet's per-group `C` (correct count), counting each (C, N) sheet row
    once even though many detections share it -- the same de-duplication `tp_bounds` does, needed
    here only to turn a bounded count into a percentage of the sheet's true total."""
    df = pd.DataFrame({"C": ver_correct, "N": ver_total}).drop_duplicates()
    return int(df.C.sum())


def nested_cross_validate_direction(real_vecs, false_vecs, real_groups=None, false_groups=None,
                                    k_outer=5, k_inner=4, n_seeds=10,
                                    pct_grid=(20, 30, 40, 50, 60, 70, 80, 90),
                                    criterion="balanced_accuracy",
                                    external_vecs=None, external_groups=None,
                                    external_ver_correct=None, external_ver_total=None):
    """Nested cross-validation: an INNER loop picks a decision threshold using only training-fold
    data, then an OUTER loop scores that frozen (direction, threshold) pair on a held-out fold
    neither step has seen. This is stricter than `cross_validate_direction`, which only reports
    metrics at fixed thresholds and never actually selects one -- here the selection step itself is
    cross-validated, so the final number is not optimistic about whichever threshold happened to
    look best on the data used to build the direction.

    `real_groups`/`false_groups`/`external_groups` (e.g. site name or sheet-row id) make every fold,
    inner and outer, keep a proportional mix of sources (`StratifiedKFold`) instead of plain random
    folding -- important when a pool is a blend of several sites with different background noise.

    Evaluation of the "positive" (real-call) side mirrors `cross_validate_direction`: if
    `external_vecs` is given, the direction is always built from the FULL `real_vecs` pool (it is a
    reference library, not something to hold out) and the positive side being split and tested is
    `external_vecs`, bounded via `tp_bounds` when `external_ver_correct`/`external_ver_total` are
    given; otherwise `real_vecs` itself is split the same way as `false_vecs`.

    Returns a DataFrame, one row per (seed, outer_fold): `chosen_pct` (the inner-selected
    threshold), `fp_removed_pct`, and either `real_kept_pct` or the bounded columns, all measured
    ONLY on the outer-test fold. The bounded columns come in two normalizations, and they are NOT
    interchangeable: `real_kept_lo_pct`/`real_kept_hi_pct` are a fraction of the sheet's GLOBAL
    correct count, so even a perfect fold caps out near `1/k_outer` of that (only `1/k_outer` of the
    external set was in this fold's test split) -- useful to compare a single fold's contribution,
    but DO NOT sum them across a seed's folds to estimate a sheet-wide total: `tp_bounds`'s worst
    case (every globally-incorrect detection happened to land in the kept set) is independently
    assumed fresh in each fold, so summing double-counts that slack and can legitimately exceed
    100%. (A seed-level total needs `tp_bounds` called ONCE on the whole reassembled external set,
    not derived from the per-fold numbers here.) `real_kept_lo_pct_of_achievable`/`_hi_pct_of_
    achievable` are instead a fraction of what THIS fold's own test subset could possibly show --
    a proper rate comparable across folds of different sizes, and the right column to average
    across folds/seeds for a recall-rate estimate with a meaningful mean/std.
    """
    use_external = external_vecs is not None
    pos_vecs = external_vecs if use_external else real_vecs
    pos_groups = external_groups if use_external else real_groups
    pos_ver_correct = external_ver_correct if use_external else None
    pos_ver_total = external_ver_total if use_external else None
    bounded = use_external and pos_ver_correct is not None

    rows = []
    for seed in range(n_seeds):
        outer_f_ids = _split_ids(len(false_vecs), k_outer, seed, false_groups)
        outer_p_ids = _split_ids(len(pos_vecs), k_outer, seed + 10_000, pos_groups)

        for outer_fold in range(k_outer):
            f_test_mask = outer_f_ids == outer_fold
            f_train, f_test = false_vecs[~f_test_mask], false_vecs[f_test_mask]
            f_train_groups = None if false_groups is None else np.asarray(false_groups)[~f_test_mask]

            p_test_mask = outer_p_ids == outer_fold
            p_train, p_test = pos_vecs[~p_test_mask], pos_vecs[p_test_mask]
            p_train_groups = None if pos_groups is None else np.asarray(pos_groups)[~p_test_mask]
            if bounded:
                c_train, n_train = pos_ver_correct[~p_test_mask], pos_ver_total[~p_test_mask]
                c_test, n_test = pos_ver_correct[p_test_mask], pos_ver_total[p_test_mask]

            # INNER LOOP: pick a threshold percentile using only this outer fold's training data.
            inner_f_ids = _split_ids(len(f_train), k_inner, seed + outer_fold, f_train_groups)
            inner_p_ids = _split_ids(len(p_train), k_inner, seed + outer_fold + 20_000, p_train_groups)
            inner_scores = {pct: [] for pct in pct_grid}
            for inner_fold in range(k_inner):
                f_in_train = f_train[inner_f_ids != inner_fold]
                f_in_val = f_train[inner_f_ids == inner_fold]
                p_in_train = real_vecs if use_external else p_train[inner_p_ids != inner_fold]
                p_in_val = p_train[inner_p_ids == inner_fold]

                direction = build_direction(p_in_train, f_in_train)
                s_f_val, s_p_val = score(f_in_val, direction), score(p_in_val, direction)
                combined = np.concatenate([s_f_val, s_p_val])
                for pct in pct_grid:
                    thr = np.percentile(combined, pct)
                    fp_removed_frac = (s_f_val < thr).mean()
                    if bounded:
                        keep = s_p_val >= thr
                        c_val = c_train[inner_p_ids == inner_fold]
                        n_val = n_train[inner_p_ids == inner_fold]
                        lo, hi, _, avail_hi = tp_bounds(keep, c_val, n_val)
                        kept_frac = (lo + hi) / 2 / max(avail_hi, 1)
                    else:
                        kept_frac = (s_p_val >= thr).mean()
                    inner_scores[pct].append(_criterion_value(criterion, fp_removed_frac, kept_frac))
            chosen_pct = max(pct_grid, key=lambda pct: np.mean(inner_scores[pct]))

            # Refit on the FULL outer-train data at the chosen threshold, then score the untouched
            # outer-test fold exactly once.
            direction = build_direction(real_vecs if use_external else p_train, f_train)
            s_f_train, s_p_train = score(f_train, direction), score(p_train, direction)
            thr = np.percentile(np.concatenate([s_f_train, s_p_train]), chosen_pct)

            s_f_test, s_p_test = score(f_test, direction), score(p_test, direction)
            row = dict(seed=seed, outer_fold=outer_fold, chosen_pct=chosen_pct,
                      n_false_test=len(f_test), n_pos_test=len(p_test),
                      fp_removed_pct=float((s_f_test < thr).mean() * 100))
            if bounded:
                keep = s_p_test >= thr
                lo, hi, _avail_lo, avail_hi = tp_bounds(keep, c_test, n_test)
                tot = _total_correct(c_test, n_test)
                row.update(real_kept_lo_pct=100 * lo / tot, real_kept_hi_pct=100 * hi / tot,
                          real_kept_lo_pct_of_achievable=100 * lo / max(avail_hi, 1),
                          real_kept_hi_pct_of_achievable=100 * hi / max(avail_hi, 1))
            else:
                row["real_kept_pct"] = float((s_p_test >= thr).mean() * 100)
            rows.append(row)
    return pd.DataFrame(rows)
