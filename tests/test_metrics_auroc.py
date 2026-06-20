#!/usr/bin/env python
"""Unit tests for metrics.auroc (tie-averaged Mann-Whitney) and bootstrap_ci.

Ground truth is a brute-force O(n^2) pairwise AUROC (count pos>neg, +0.5 per
tie) which is the textbook definition and handles ties explicitly, so the test
is self-contained (no sklearn dependency on this host). An optional sklearn
cross-check runs only if sklearn is importable.

Pins the ORIENTATION contract (A2): the Exp-3 positive class is the incorrect
step, so downstream code passes score = 1 - conf; this test asserts the flip
identity AUROC(1 - s) == 1 - AUROC(s) holds exactly under tie-averaging.

Run directly: `python tests/test_metrics_auroc.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asym_rewards.metrics import auroc, bootstrap_ci  # noqa: E402


def _brute_auroc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    wins = 0.0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1.0
            elif a == b:
                wins += 0.5
    return wins / (pos.size * neg.size)


def test_no_tie_matches_brute_force():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=200)
    labels = (rng.random(200) < 0.4).astype(int)
    assert abs(auroc(scores, labels) - _brute_auroc(scores, labels)) < 1e-12


def test_tie_heavy_matches_brute_force():
    # discrete vote-fraction confidence over K=8 -> only 9 distinct values, many ties
    rng = np.random.default_rng(1)
    scores = rng.integers(0, 9, size=300) / 8.0
    labels = (rng.random(300) < 0.5).astype(int)
    assert abs(auroc(scores, labels) - _brute_auroc(scores, labels)) < 1e-12


def test_hand_checked_example():
    # pos scores {0.9, 0.6}, neg scores {0.6, 0.1}
    # pairs: 0.9>0.6 win, 0.9>0.1 win, 0.6==0.6 tie(0.5), 0.6>0.1 win => 3.5/4
    scores = np.array([0.9, 0.6, 0.6, 0.1])
    labels = np.array([1, 1, 0, 0])
    assert abs(auroc(scores, labels) - 0.875) < 1e-12


def test_perfect_and_inverted():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert abs(auroc(scores, labels) - 1.0) < 1e-12
    assert abs(auroc(-scores, labels) - 0.0) < 1e-12


def test_orientation_flip_identity():
    # A2: detecting the opposite class via 1 - score must map AUROC -> 1 - AUROC,
    # exactly, even with ties (tie-averaged Mann-Whitney guarantees this).
    rng = np.random.default_rng(2)
    scores = rng.integers(0, 9, size=400) / 8.0
    labels = (rng.random(400) < 0.45).astype(int)
    a = auroc(scores, labels)
    a_flip = auroc(1.0 - scores, labels)
    assert abs(a + a_flip - 1.0) < 1e-12


def test_single_class_is_nan():
    assert np.isnan(auroc(np.array([0.1, 0.2, 0.3]), np.array([1, 1, 1])))
    assert np.isnan(auroc(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 0])))


def test_nan_score_is_nan_not_inflated():
    # np.argsort sorts NaN to max rank; without a guard a NaN positive would
    # silently beat all negatives and inflate AUROC. Must return nan instead.
    scores = np.array([0.2, np.nan, 0.1, 0.3])
    labels = np.array([0, 1, 0, 1])
    assert np.isnan(auroc(scores, labels))


def test_optional_sklearn_crosscheck():
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("  [skip] sklearn not installed; brute-force check already covers it")
        return
    rng = np.random.default_rng(3)
    scores = rng.normal(size=500)
    labels = (rng.random(500) < 0.5).astype(int)
    assert abs(auroc(scores, labels) - roc_auc_score(labels, scores)) < 1e-9


def test_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(4)
    scores = rng.normal(size=300)
    labels = (rng.random(300) < 0.4).astype(int)
    point = auroc(scores, labels)
    lo, hi = bootstrap_ci(auroc, scores, labels, n_boot=500, seed=7)
    assert lo <= hi
    assert lo <= point <= hi


def test_bootstrap_ci_empty_is_nan():
    lo, hi = bootstrap_ci(auroc, np.array([]), np.array([]), n_boot=100)
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_ci_nan_point_estimate_is_nan():
    # auroc is nan on NaN scores; bootstrap must NOT manufacture a finite CI
    # from the ~37% of replicates that omit the bad row.
    scores = np.array([0.2, np.nan, 0.1, 0.3, 0.4, 0.5])
    labels = np.array([0, 1, 0, 1, 0, 1])
    lo, hi = bootstrap_ci(auroc, scores, labels, n_boot=1000, seed=0)
    assert np.isnan(lo) and np.isnan(hi)


def test_nonbinary_labels_excluded_from_ranking():
    # neutral rows (label 2, or -1) must be dropped, not ranked: AUROC over the
    # binary subset, always within [0, 1], matching the explicit binary slice.
    scores = np.array([0.9, 0.1, 0.5, 0.5, 0.05])
    labels = np.array([1, 0, 2, -1, 0])  # two neutral rows
    got = auroc(scores, labels)
    want = auroc(np.array([0.9, 0.1, 0.05]), np.array([1, 0, 0]))
    assert 0.0 <= got <= 1.0
    assert abs(got - want) < 1e-12


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok: {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
