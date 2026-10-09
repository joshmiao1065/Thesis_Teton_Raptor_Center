import numpy as np

from training.discriminant_gate import (
    build_direction,
    cross_validate_direction,
    fold_id,
    score,
    tp_bounds,
)


def test_fold_id_covers_every_index_and_is_balanced():
    ids = fold_id(103, k_folds=5, seed=0)
    assert set(ids) == set(range(5))
    assert len(ids) == 103
    counts = np.bincount(ids)
    assert counts.max() - counts.min() <= 1  # near-even split


def test_fold_id_is_deterministic_per_seed():
    a = fold_id(50, 5, seed=1)
    b = fold_id(50, 5, seed=1)
    c = fold_id(50, 5, seed=2)
    assert (a == b).all()
    assert not (a == c).all()


def test_build_direction_prototype_vs_discriminant():
    real = np.array([[1.0, 0.0], [1.0, 0.0]])  # mean exactly on +x
    proto = build_direction(real)
    np.testing.assert_allclose(proto, [1.0, 0.0])

    false = np.array([[0.0, 1.0], [0.0, 1.0]])  # mean exactly on +y
    disc = build_direction(real, false)
    # subtracting the false mean should pull the direction away from +y, toward -y
    assert disc[1] < proto[1]
    np.testing.assert_allclose(disc, [1 / np.sqrt(2), -1 / np.sqrt(2)], atol=1e-9)


def test_score_is_dot_product_not_cosine():
    direction = np.array([1.0, 0.0])
    vecs = np.array([[3.0, 0.0], [1.0, 0.0]])  # same direction, different magnitude
    s = score(vecs, direction)
    assert s[0] > s[1]  # magnitude matters: this is a plain dot product, not cosine similarity


def test_tp_bounds_matches_known_sheet_counts():
    # a single sheet row: 8 correct of 10 total, only 6 of the 10 were tested (heldout subset)
    ver_correct = np.full(6, 8)
    ver_total = np.full(6, 10)
    keep_all = np.ones(6, bool)
    lo, hi, avail_lo, avail_hi = tp_bounds(keep_all, ver_correct, ver_total)
    # at most 2 of the sheet's global incorrect detections could be among these 6 kept, so >= 4 must be correct
    assert lo == 4
    assert hi == 6  # can't exceed what was kept
    assert avail_hi == 6  # ceiling on true positives achievable in a 6-detection tested subset
    assert avail_lo == 4

    keep_none = np.zeros(6, bool)
    lo0, hi0, _, _ = tp_bounds(keep_none, ver_correct, ver_total)
    assert lo0 == 0 and hi0 == 0


def test_cross_validate_direction_separates_well_separated_clusters():
    rng = np.random.default_rng(0)
    real = rng.normal(loc=[5, 0], scale=0.3, size=(80, 2))
    false = rng.normal(loc=[0, 5], scale=0.3, size=(80, 2))
    cv = cross_validate_direction(real, false, k_folds=5, n_seeds=3, pct_thresholds=(50,))
    assert len(cv) == 5 * 3  # one row per (seed, fold) at a single threshold
    assert cv.fp_removed_pct.mean() > 90
    assert cv.real_kept_pct.mean() > 90


def test_cross_validate_direction_struggles_on_overlapping_clusters():
    rng = np.random.default_rng(0)
    real = rng.normal(loc=[0, 0], scale=1.0, size=(80, 2))
    false = rng.normal(loc=[0.2, 0.2], scale=1.0, size=(80, 2))  # heavily overlapping
    cv = cross_validate_direction(real, false, k_folds=5, n_seeds=3, pct_thresholds=(50,))
    # at the 50th percentile of a near-even mix, removal should sit near chance, not near-complete
    assert cv.fp_removed_pct.mean() < 80


def test_cross_validate_direction_with_external_recall_check_and_bounds():
    rng = np.random.default_rng(0)
    real = rng.normal(loc=[5, 0], scale=0.3, size=(80, 2))
    false = rng.normal(loc=[0, 5], scale=0.3, size=(80, 2))
    external = rng.normal(loc=[5, 0], scale=0.3, size=(20, 2))  # same cluster as `real`, independent set
    ver_correct = np.full(20, 18)  # the sheet says 18 of 20 are correct, globally
    ver_total = np.full(20, 20)
    cv = cross_validate_direction(real, false, k_folds=5, n_seeds=2, pct_thresholds=(50,),
                                  external_vecs=external, external_ver_correct=ver_correct, external_ver_total=ver_total)
    assert "real_kept_lo_pct" in cv.columns and "real_kept_hi_pct" in cv.columns
    assert (cv.real_kept_lo_pct <= cv.real_kept_hi_pct + 1e-9).all()  # bounds must not invert
    assert cv.real_kept_hi_pct.mean() > 70  # well-separated clusters: most achievable recall kept
