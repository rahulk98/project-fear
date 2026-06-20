"""Calibration and selective-prediction metrics for Experiment 1.

All inputs are numpy arrays of equal length (one entry per problem):
- `p_hat`: confidence in [0, 1], typically the empirical Pass@K probability
- `correct`: 0/1 (or bool) indicator of whether ANY of the K samples was correct
            (or per-sample correctness when scoring at K=1)

`records`-shaped helpers operate on a list of dicts shaped:
    {"problem_id": str, "samples": [{"correct": bool, "abstained": bool}, ...]}
"""
from __future__ import annotations

from math import comb
from typing import Callable

import numpy as np


def _sample_fraction(records: list[dict], field: str) -> float:
    n = sum(len(r["samples"]) for r in records)
    if n == 0:
        return 0.0
    return sum(s[field] for r in records for s in r["samples"]) / n


def accuracy(records: list[dict]) -> float:
    """Pass@1-style: fraction of (problem, sample) pairs that are correct."""
    return _sample_fraction(records, "correct")


def abstention_rate(records: list[dict]) -> float:
    """Fraction of (problem, sample) pairs with no extractable boxed answer."""
    return _sample_fraction(records, "abstained")


def expected_calibration_error(p_hat: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """ECE with equal-width bins on [0, 1]."""
    p_hat = np.asarray(p_hat, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if p_hat.size == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if hi == 1.0:
            mask = (p_hat >= lo) & (p_hat <= hi)
        else:
            mask = (p_hat >= lo) & (p_hat < hi)
        if not mask.any():
            continue
        ece += (mask.sum() / p_hat.size) * abs(p_hat[mask].mean() - correct[mask].mean())
    return float(ece)


def brier_score(p_hat: np.ndarray, correct: np.ndarray) -> float:
    """Mean squared error of confidence vs binary outcome."""
    p_hat = np.asarray(p_hat, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if p_hat.size == 0:
        return 0.0
    return float(np.mean((p_hat - correct) ** 2))


def coverage_risk_curve(p_hat: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by descending confidence; at each cumulative coverage level, report risk (1 - accuracy).

    Returns (coverages, risks), both shape (N,). coverages[i] = (i+1)/N.
    """
    p_hat = np.asarray(p_hat, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = p_hat.size
    if n == 0:
        return np.array([]), np.array([])
    order = np.argsort(-p_hat, kind="stable")
    counts = np.arange(1, n + 1, dtype=float)
    coverages = counts / n
    risks = 1.0 - np.cumsum(correct[order]) / counts
    return coverages, risks


def auacc(p_hat: np.ndarray, correct: np.ndarray) -> float:
    """Area under the accuracy-coverage curve (higher = better selective prediction)."""
    coverages, risks = coverage_risk_curve(p_hat, correct)
    if coverages.size == 0:
        return 0.0
    return float(np.trapz(1.0 - risks, coverages))


def selective_accuracy_at_coverage(
    p_hat: np.ndarray, correct: np.ndarray, target_coverage: float
) -> float:
    """Accuracy on the top-`target_coverage` fraction of most-confident predictions."""
    p_hat = np.asarray(p_hat, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = p_hat.size
    if n == 0 or target_coverage <= 0:
        return 0.0
    k = max(1, int(round(target_coverage * n)))
    order = np.argsort(-p_hat, kind="stable")
    return float(correct[order[:k]].mean())


def pass_at_k(n_samples: int, n_correct: int, k: int) -> float:
    """Unbiased estimator: probability that at least one of k uniform samples is correct.

    1 - C(n - c, k) / C(n, k), with the standard truncation for k > n - c.
    """
    if k <= 0 or n_samples <= 0:
        return 0.0
    if n_samples - n_correct < k:
        return 1.0
    return 1.0 - comb(n_samples - n_correct, k) / comb(n_samples, k)


def pass_at_k_corpus(records: list[dict], k: int) -> float:
    """Mean Pass@k over all problems in `records`."""
    if not records:
        return 0.0
    vals = [
        pass_at_k(len(r["samples"]), sum(s["correct"] for s in r["samples"]), k)
        for r in records
    ]
    return float(np.mean(vals))


# ----- Ranking metrics: AUROC + bootstrap CI --------------------------------

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve, ranking the positive class (labels == 1) above
    the negative class (labels == 0).

    Rank-based (Mann-Whitney U) with TIE-AVERAGED ranks. Tie handling is
    mandatory here: a discrete confidence score (e.g. vote-fraction over K
    samples takes only K+1 distinct values) produces many ties, and a plain
    argsort AUROC is biased on ties. Returns nan for single-class input (one of
    the two classes empty) so a degenerate set never silently reports 0.5.

    Orientation: a higher `score` should indicate the positive class. To detect
    the negative/incorrect class instead, pass `score = 1 - conf` (or negate),
    which maps AUROC -> 1 - AUROC.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    # Restrict to the two classes of interest. Rows with any other label (e.g. a
    # neutral 0-vs-rest convention, or an unmapped value) must be EXCLUDED from
    # ranking too -- leaving them in argsort would shift ranks and can push
    # AUROC outside [0, 1].
    binary = (labels == 0) | (labels == 1)
    if not bool(binary.all()):
        scores = scores[binary]
        labels = labels[binary]
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    if np.any(np.isnan(scores)):
        # np.argsort sorts NaN to the END (max rank), which would silently
        # inflate AUROC; refuse rather than misreport.
        return float("nan")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=float)
    i = 0
    while i < scores.size:
        j = i
        while j + 1 < scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        # average rank (1-based) for the tie run [i, j]
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[pos].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def bootstrap_ci(
    metric_fn: Callable[..., float],
    *arrays: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for `metric_fn(*arrays)`.

    Resamples row indices jointly across all arrays (paired), so metrics over
    (score, label) pairs stay aligned. NaN replicates (e.g. a single-class
    resample under `auroc`) are dropped. Returns (lo, hi) at the
    (alpha/2, 1-alpha/2) percentiles, or (nan, nan) if every replicate is NaN.
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    assert all(len(a) == n for a in arrays), "bootstrap_ci arrays must be equal length"
    if n == 0:
        return (float("nan"), float("nan"))
    # If the metric is undefined on the full data (e.g. auroc on NaN scores or a
    # single-class set), the CI is undefined too -- do NOT manufacture a finite
    # interval from the subset of replicates that happen to omit the bad rows.
    if np.isnan(metric_fn(*arrays)):
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        val = metric_fn(*[a[idx] for a in arrays])
        if not np.isnan(val):
            vals.append(val)
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [100 * alpha / 2.0, 100 * (1.0 - alpha / 2.0)])
    return (float(lo), float(hi))


# ----- Selective prediction: fear-vs-calibration diagnostics ----------------

def attempt_rate(records: list[dict]) -> float:
    """Fraction of samples with an extracted answer (1 - abstention_rate)."""
    return 1.0 - abstention_rate(records)


def accuracy_on_attempted(records: list[dict]) -> float:
    """Accuracy computed only over samples that emitted a \\boxed{} answer.

    This is the headline "did abstention buy quality?" number. A scared model
    abstains a lot but does not raise this; a calibrated model does.
    """
    n_attempt = sum(1 for r in records for s in r["samples"] if not s["abstained"])
    if n_attempt == 0:
        return 0.0
    n_correct = sum(
        1 for r in records for s in r["samples"]
        if not s["abstained"] and s["correct"]
    )
    return n_correct / n_attempt


def per_problem_attempt_accuracy(records: list[dict]) -> dict[str, float | None]:
    """For each problem, P(correct | attempted) from non-abstained sibling samples.

    Returns None for problems where every sample abstained (counterfactual unknown).
    Used as the cross-sample estimate of "would the model have been correct?".
    """
    out: dict[str, float | None] = {}
    for r in records:
        attempted = [s for s in r["samples"] if not s["abstained"]]
        if not attempted:
            out[r["problem_id"]] = None
        else:
            out[r["problem_id"]] = sum(s["correct"] for s in attempted) / len(attempted)
    return out


def selective_confusion(
    records: list[dict],
    counterfactual_correct_by_pid: dict[str, float | None] | None = None,
) -> dict:
    """Per-sample selective confusion matrix.

    Returns counts:
      - attempt_right:   sample attempted AND correct
      - attempt_wrong:   sample attempted AND wrong
      - abstain_on_right (expected): sum over abstained samples of P(would_be_correct)
      - abstain_on_wrong (expected): sum over abstained samples of (1 - P(would_be_correct))
      - abstain_unknown: abstained samples on problems with no sibling attempts
      - total

    When `counterfactual_correct_by_pid` is None, falls back to the per-problem
    sibling-sample estimate via `per_problem_attempt_accuracy`. Pass a probe-
    derived map (forced-decoding correctness per pid or per (pid, sample_idx))
    to use a stronger counterfactual.
    """
    if counterfactual_correct_by_pid is None:
        counterfactual_correct_by_pid = per_problem_attempt_accuracy(records)

    attempt_right = attempt_wrong = 0
    abstain_on_right = abstain_on_wrong = 0.0
    abstain_unknown = 0
    total = 0
    for r in records:
        p_correct = counterfactual_correct_by_pid.get(r["problem_id"])
        for s in r["samples"]:
            total += 1
            if not s["abstained"]:
                if s["correct"]:
                    attempt_right += 1
                else:
                    attempt_wrong += 1
            else:
                if p_correct is None:
                    abstain_unknown += 1
                else:
                    abstain_on_right += p_correct
                    abstain_on_wrong += 1.0 - p_correct
    return {
        "attempt_right": attempt_right,
        "attempt_wrong": attempt_wrong,
        "abstain_on_right_expected": float(abstain_on_right),
        "abstain_on_wrong_expected": float(abstain_on_wrong),
        "abstain_unknown": abstain_unknown,
        "total": total,
    }


def _outcome(sample: dict) -> str:
    if sample["abstained"]:
        return "abstain"
    return "correct" if sample["correct"] else "wrong"


def condition_transition(
    rollouts_a: list[dict],
    rollouts_b: list[dict],
) -> dict:
    """3x3 transition matrix between two conditions on matched (pid, sample_idx).

    Unlike `accuracy_on_attempted` / `selective_confusion`, this takes the FLAT
    per-sample rollouts (the JSONL row shape), not pid-grouped records, because
    the matched-pair join needs sample_idx. Pairing is exact on
    (problem_id, sample_idx) - meaningful because seed_for() is deterministic
    across conditions, so the same key gets the same sampling seed in A and B.

    Returns:
      matrix:  {from_outcome: {to_outcome: count}} for outcomes in {correct,wrong,abstain}
      n_pairs: number of matched samples
      named:   key transitions the project cares about (correct->abstain, wrong->abstain, ...)
    """
    outcomes = ("correct", "wrong", "abstain")
    by_key_a = {(r["problem_id"], r["sample_idx"]): r for r in rollouts_a}
    by_key_b = {(r["problem_id"], r["sample_idx"]): r for r in rollouts_b}
    shared = sorted(set(by_key_a) & set(by_key_b))

    matrix = {fo: {to: 0 for to in outcomes} for fo in outcomes}
    for key in shared:
        matrix[_outcome(by_key_a[key])][_outcome(by_key_b[key])] += 1

    named = {
        "correct_to_abstain": matrix["correct"]["abstain"],
        "wrong_to_abstain":   matrix["wrong"]["abstain"],
        "abstain_to_correct": matrix["abstain"]["correct"],
        "abstain_to_wrong":   matrix["abstain"]["wrong"],
        "correct_to_wrong":   matrix["correct"]["wrong"],
        "wrong_to_correct":   matrix["wrong"]["correct"],
        "stayed_correct":     matrix["correct"]["correct"],
        "stayed_wrong":       matrix["wrong"]["wrong"],
        "stayed_abstain":     matrix["abstain"]["abstain"],
    }
    return {"matrix": matrix, "n_pairs": len(shared), "named": named}
