"""Tests for judge.score_condition and judge.matched_coverage_report.

numpy-only; no torch, no vllm, no math_verify.

Run:
    python3 tests/test_judge_scoring.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asym_rewards.judge import matched_coverage_report, score_condition  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic row builders
# ---------------------------------------------------------------------------

def _row(gold: int, pred: int, conf_yes: float) -> dict:
    return {"gold": gold, "pred": pred, "conf_yes": conf_yes}


def _correct_rows(n: int = 10) -> list[dict]:
    """All correctly judged rows (pred == gold, never abstain)."""
    rows = []
    for i in range(n):
        gold = 1 if i % 2 == 0 else -1
        rows.append(_row(gold=gold, pred=gold, conf_yes=1.0 if gold == 1 else 0.0))
    return rows


def _all_abstain_rows(n: int = 10) -> list[dict]:
    return [_row(gold=1, pred=0, conf_yes=0.5) for _ in range(n)]


# ---------------------------------------------------------------------------
# score_condition tests
# ---------------------------------------------------------------------------

def test_score_condition_empty():
    """Empty rows list returns all-nan/zero result without error."""
    out = score_condition([])
    assert out["n_attempted"] == 0
    assert out["coverage"] == 0.0
    assert math.isnan(out["auroc_incorrect"])
    assert out["committed_indices"] == []


def test_score_condition_all_abstain():
    """All-abstain rows: n_attempted=0, auroc=nan."""
    out = score_condition(_all_abstain_rows(10))
    assert out["n_attempted"] == 0
    assert math.isnan(out["auroc_incorrect"])
    assert out["coverage"] == 0.0


def test_score_condition_all_committed():
    """No abstentions: n_attempted == n."""
    rows = _correct_rows(8)
    out = score_condition(rows)
    assert out["n_attempted"] == 8
    assert abs(out["coverage"] - 1.0) < 1e-9


def test_score_condition_committed_indices_correct():
    """committed_indices contains exactly the indices with pred != 0."""
    rows = [
        _row(gold=1, pred=1, conf_yes=0.9),   # idx 0 committed
        _row(gold=-1, pred=0, conf_yes=0.5),  # idx 1 abstain
        _row(gold=-1, pred=-1, conf_yes=0.1), # idx 2 committed
        _row(gold=1, pred=0, conf_yes=0.5),   # idx 3 abstain
    ]
    out = score_condition(rows)
    assert out["committed_indices"] == [0, 2]
    assert out["n_attempted"] == 2
    assert abs(out["coverage"] - 0.5) < 1e-9


def test_score_condition_positive_class_is_incorrect():
    """Positive class = gold==-1 (incorrect step). A perfect classifier for
    incorrect steps should yield AUROC >= 0.5 when it scores them higher."""
    # 4 committed rows: 2 incorrect (gold=-1), 2 correct (gold=1).
    # Perfect oracle: incorrect rows have score=1 (conf_yes=0), correct have score=0.
    rows = [
        _row(gold=-1, pred=-1, conf_yes=0.0),  # score = 1.0, label=1
        _row(gold=-1, pred=-1, conf_yes=0.0),  # score = 1.0, label=1
        _row(gold=1,  pred=1,  conf_yes=1.0),  # score = 0.0, label=0
        _row(gold=1,  pred=1,  conf_yes=1.0),  # score = 0.0, label=0
    ]
    out = score_condition(rows)
    assert not math.isnan(out["auroc_incorrect"]), "AUROC should not be nan for balanced rows"
    assert out["auroc_incorrect"] >= 0.5, f"Expected AUROC >= 0.5, got {out['auroc_incorrect']}"
    assert out["n_pos"] == 2  # incorrect committed steps
    assert out["n_neg"] == 2  # correct committed steps


def test_score_condition_degenerate_single_class_is_nan():
    """Only incorrect steps committed -> single-class labels -> AUROC=nan."""
    rows = [_row(gold=-1, pred=-1, conf_yes=0.1) for _ in range(5)]
    out = score_condition(rows)
    assert math.isnan(out["auroc_incorrect"]), "Single-class AUROC must be nan"


def test_score_condition_coverage_fraction():
    """Coverage = n_committed / n."""
    rows = [_row(gold=1, pred=1, conf_yes=0.9)] * 3 + \
           [_row(gold=1, pred=0, conf_yes=0.5)] * 7
    out = score_condition(rows)
    assert abs(out["coverage"] - 0.3) < 1e-9


# ---------------------------------------------------------------------------
# f1_incorrect tests
# ---------------------------------------------------------------------------

def test_f1_incorrect_perfect_classifier():
    """Perfect classifier: pred==-1 exactly when gold==-1 -> F1=1.0."""
    rows = [
        _row(gold=-1, pred=-1, conf_yes=0.0),  # TP
        _row(gold=-1, pred=-1, conf_yes=0.0),  # TP
        _row(gold=1,  pred=1,  conf_yes=1.0),  # TN
        _row(gold=1,  pred=1,  conf_yes=1.0),  # TN
    ]
    out = score_condition(rows)
    assert abs(out["f1_incorrect"] - 1.0) < 1e-9, (
        f"Perfect classifier should have f1_incorrect=1.0, got {out['f1_incorrect']}"
    )


def test_f1_incorrect_all_predict_negative():
    """Judge always predicts incorrect (pred==-1) on 2 incorrect + 2 correct rows.

    Confusion:
      TP=2 (pred=-1, gold=-1), FP=2 (pred=-1, gold=1), FN=0
      precision = 2/(2+2) = 0.5, recall = 2/(2+0) = 1.0
      F1 = 2*0.5*1.0 / (0.5+1.0) = 1.0/1.5 = 0.6667
    """
    rows = [
        _row(gold=-1, pred=-1, conf_yes=0.0),  # TP
        _row(gold=-1, pred=-1, conf_yes=0.0),  # TP
        _row(gold=1,  pred=-1, conf_yes=0.9),  # FP
        _row(gold=1,  pred=-1, conf_yes=0.9),  # FP
    ]
    out = score_condition(rows)
    expected_f1 = 2 * 0.5 * 1.0 / (0.5 + 1.0)
    assert abs(out["f1_incorrect"] - expected_f1) < 1e-9, (
        f"Expected f1_incorrect={expected_f1:.4f}, got {out['f1_incorrect']}"
    )


def test_f1_incorrect_no_positives_predicted():
    """Judge never predicts incorrect (all pred==1) -> precision=0, F1=0.0."""
    rows = [
        _row(gold=-1, pred=1, conf_yes=0.8),  # FN
        _row(gold=1,  pred=1, conf_yes=0.9),  # TN
    ]
    out = score_condition(rows)
    assert out["f1_incorrect"] == 0.0, (
        f"No positive predictions -> F1 should be 0.0, got {out['f1_incorrect']}"
    )


def test_f1_incorrect_all_abstain_is_zero():
    """All-abstain rows: n_attempted=0 -> f1_incorrect=0.0."""
    out = score_condition(_all_abstain_rows(10))
    assert out["f1_incorrect"] == 0.0, (
        f"All-abstain rows: f1_incorrect must be 0.0, got {out['f1_incorrect']}"
    )


def test_f1_incorrect_empty_rows_is_zero():
    """Empty rows: f1_incorrect must be 0.0."""
    out = score_condition([])
    assert out["f1_incorrect"] == 0.0, (
        f"Empty rows: f1_incorrect must be 0.0, got {out['f1_incorrect']}"
    )


def test_f1_incorrect_mixed_hand_checked():
    """Hand-checked confusion matrix.

    Committed rows (pred != 0):
      idx 0: gold=-1, pred=-1  -> TP
      idx 1: gold=-1, pred=1   -> FN
      idx 2: gold=1,  pred=-1  -> FP
      idx 3: gold=1,  pred=1   -> TN
    TP=1, FP=1, FN=1
    precision = 1/2 = 0.5, recall = 1/2 = 0.5
    F1 = 2*0.5*0.5/(0.5+0.5) = 0.5
    """
    rows = [
        _row(gold=-1, pred=-1, conf_yes=0.1),  # TP
        _row(gold=-1, pred=1,  conf_yes=0.9),  # FN
        _row(gold=1,  pred=-1, conf_yes=0.1),  # FP
        _row(gold=1,  pred=1,  conf_yes=0.9),  # TN
        _row(gold=1,  pred=0,  conf_yes=0.5),  # abstain (not committed)
    ]
    out = score_condition(rows)
    assert abs(out["f1_incorrect"] - 0.5) < 1e-9, (
        f"Expected f1_incorrect=0.5, got {out['f1_incorrect']}"
    )


# ---------------------------------------------------------------------------
# matched_coverage_report tests
# ---------------------------------------------------------------------------

def _make_rows_with_committed(n: int, committed_idxs: list[int], gold_pattern=None) -> list[dict]:
    """Build n rows; committed_idxs have pred != 0, rest abstain."""
    committed_set = set(committed_idxs)
    rows = []
    for i in range(n):
        if i in committed_set:
            g = 1 if (i % 2 == 0) else -1
            if gold_pattern:
                g = gold_pattern[i % len(gold_pattern)]
            rows.append(_row(gold=g, pred=g, conf_yes=1.0 if g == 1 else 0.0))
        else:
            rows.append(_row(gold=1, pred=0, conf_yes=0.5))
    return rows


def test_matched_coverage_fallback_tiny_n():
    """When matched set is below tiny_n, fallback=True and warning is set."""
    # 2 conditions each committed on 1 step, intersection = 1 < tiny_n=5
    pc = {"A": {0}, "B": {0, 1}}
    rows_a = _make_rows_with_committed(5, [0, 1])
    rows_b = _make_rows_with_committed(5, [0, 1])
    per_rows = {"A": rows_a, "B": rows_b}
    out = matched_coverage_report(pc, per_rows, attempt_floor=0.0, tiny_n=5)
    assert out["fallback"] is True
    assert out["warning"] is not None
    assert out["per_condition"] == {}


def test_matched_coverage_full_intersection():
    """When all conditions commit on all rows, matched set = all rows."""
    n = 10
    idxs = list(range(n))
    pc = {
        "S0": set(idxs),
        "S2": set(idxs),
    }
    rows = _make_rows_with_committed(n, idxs)
    per_rows = {"S0": rows, "S2": rows}
    out = matched_coverage_report(pc, per_rows, attempt_floor=0.0, tiny_n=5)
    assert out["fallback"] is False
    assert out["n_matched"] == n
    assert "S0" in out["per_condition"]
    assert "S2" in out["per_condition"]


def test_matched_coverage_excludes_degenerate():
    """A condition below attempt_floor is excluded from the matched set."""
    n = 20
    # S0: commits on 1/20 = 0.05 < floor=0.2 -> excluded
    # S2: commits on all 20 -> included
    pc = {
        "S0": {0},
        "S2": set(range(n)),
    }
    rows_s0 = _make_rows_with_committed(n, [0])
    rows_s2 = _make_rows_with_committed(n, list(range(n)))
    per_rows = {"S0": rows_s0, "S2": rows_s2}
    out = matched_coverage_report(pc, per_rows, attempt_floor=0.2, tiny_n=5)
    assert "S0" in out["conditions_excluded"]
    assert "S2" in out["conditions_included"]
    # matched set = S2's committed (only included condition)
    assert out["n_matched"] == n


def test_matched_coverage_empty_conditions():
    """Empty per_condition_committed produces n_matched=0, no exception."""
    out = matched_coverage_report({}, {}, attempt_floor=0.2, tiny_n=5)
    assert out["n_matched"] == 0
    assert out["per_condition"] == {}


def test_matched_coverage_per_condition_keys():
    """per_condition dict must contain auroc_incorrect, ci bounds, n."""
    n = 20
    idxs = list(range(n))
    pc = {"A": set(idxs)}
    rows = [_row(gold=(-1 if i % 2 == 0 else 1), pred=(-1 if i % 2 == 0 else 1),
                 conf_yes=(0.1 if i % 2 == 0 else 0.9)) for i in range(n)]
    per_rows = {"A": rows}
    out = matched_coverage_report(pc, per_rows, attempt_floor=0.0, tiny_n=5)
    if not out["fallback"]:
        for cond_data in out["per_condition"].values():
            assert "auroc_incorrect" in cond_data
            assert "auroc_ci_lo" in cond_data
            assert "auroc_ci_hi" in cond_data
            assert "n" in cond_data


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_all() -> None:
    tests = [
        test_score_condition_empty,
        test_score_condition_all_abstain,
        test_score_condition_all_committed,
        test_score_condition_committed_indices_correct,
        test_score_condition_positive_class_is_incorrect,
        test_score_condition_degenerate_single_class_is_nan,
        test_score_condition_coverage_fraction,
        test_f1_incorrect_perfect_classifier,
        test_f1_incorrect_all_predict_negative,
        test_f1_incorrect_no_positives_predicted,
        test_f1_incorrect_all_abstain_is_zero,
        test_f1_incorrect_empty_rows_is_zero,
        test_f1_incorrect_mixed_hand_checked,
        test_matched_coverage_fallback_tiny_n,
        test_matched_coverage_full_intersection,
        test_matched_coverage_excludes_degenerate,
        test_matched_coverage_empty_conditions,
        test_matched_coverage_per_condition_keys,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {t.__name__}: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
