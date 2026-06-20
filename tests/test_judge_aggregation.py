#!/usr/bin/env python
"""Unit tests for vote-aggregation and matched-coverage helpers in smoke_judge.

Pure-function tests only (numpy, no vllm/torch/math_verify).

Run directly: `python tests/test_judge_aggregation.py`
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# aggregate_votes and matched_subset now live in asym_rewards.judge (canonical).
# smoke_judge re-imports from there; both import paths are valid. Tests use the
# canonical location.
from asym_rewards.judge import aggregate_votes, matched_subset  # noqa: E402


# ---------------------------------------------------------------------------
# aggregate_votes tests
# ---------------------------------------------------------------------------

def test_aggregate_unanimous_yes():
    agg = aggregate_votes([1, 1, 1])
    assert agg["pred"] == 1
    assert agg["votes_yes"] == 3
    assert agg["votes_no"] == 0
    assert agg["votes_abstain"] == 0
    assert abs(agg["conf_yes"] - 1.0) < 1e-9


def test_aggregate_unanimous_no():
    agg = aggregate_votes([-1, -1, -1])
    assert agg["pred"] == -1
    assert agg["votes_no"] == 3
    assert abs(agg["conf_yes"]) < 1e-9


def test_aggregate_yes_wins():
    agg = aggregate_votes([1, 1, -1, 0])
    assert agg["pred"] == 1
    assert agg["votes_yes"] == 2
    assert agg["votes_no"] == 1
    assert agg["votes_abstain"] == 1
    assert abs(agg["conf_yes"] - 0.5) < 1e-9


def test_aggregate_no_wins():
    agg = aggregate_votes([-1, -1, 1, 0])
    assert agg["pred"] == -1
    assert agg["votes_no"] == 2
    assert agg["votes_yes"] == 1


def test_aggregate_tie_is_abstain():
    # Equal yes and no -> tie -> pred == 0.
    agg = aggregate_votes([1, -1])
    assert agg["pred"] == 0
    agg2 = aggregate_votes([1, 1, -1, -1])
    assert agg2["pred"] == 0


def test_aggregate_all_abstain():
    agg = aggregate_votes([0, 0, 0])
    assert agg["pred"] == 0
    assert agg["votes_abstain"] == 3
    assert abs(agg["conf_yes"]) < 1e-9


def test_aggregate_k1_greedy_yes():
    agg = aggregate_votes([1])
    assert agg["pred"] == 1
    assert abs(agg["conf_yes"] - 1.0) < 1e-9


def test_aggregate_k1_greedy_no():
    agg = aggregate_votes([-1])
    assert agg["pred"] == -1
    assert abs(agg["conf_yes"]) < 1e-9


def test_aggregate_k1_greedy_abstain():
    agg = aggregate_votes([0])
    assert agg["pred"] == 0
    assert abs(agg["conf_yes"]) < 1e-9


def test_aggregate_empty():
    agg = aggregate_votes([])
    assert agg["pred"] == 0
    assert agg["votes_yes"] == 0
    assert agg["votes_no"] == 0
    assert agg["votes_abstain"] == 0
    assert abs(agg["conf_yes"]) < 1e-9


def test_aggregate_conf_yes_fraction():
    # 3 yes out of 8 -> conf_yes == 0.375
    agg = aggregate_votes([1, 1, 1, -1, -1, -1, -1, 0])
    assert abs(agg["conf_yes"] - 3 / 8) < 1e-9
    assert agg["pred"] == -1  # no > yes


# ---------------------------------------------------------------------------
# matched_subset tests
# ---------------------------------------------------------------------------

def test_matched_subset_full_overlap():
    pc = {
        "S0": {0, 1, 2, 3},
        "S1": {0, 1, 2, 3},
        "C1": {0, 1, 2, 3},
    }
    result = matched_subset(pc, attempt_floor=0.0, n_total=4)
    assert result["n_matched"] == 4
    assert set(result["matched_indices"]) == {0, 1, 2, 3}
    assert set(result["conditions_included"]) == {"S0", "S1", "C1"}
    assert result["conditions_excluded"] == []


def test_matched_subset_partial_overlap():
    pc = {
        "S0": {0, 1, 2, 3, 4},
        "S1": {1, 2, 3, 4, 5},
        "C1": {2, 3, 4, 5, 6},
    }
    result = matched_subset(pc, attempt_floor=0.0, n_total=7)
    # Intersection of {0-4}, {1-5}, {2-6} = {2, 3, 4}
    assert set(result["matched_indices"]) == {2, 3, 4}
    assert result["n_matched"] == 3


def test_matched_subset_no_overlap():
    pc = {
        "S0": {0, 1},
        "S1": {2, 3},
    }
    result = matched_subset(pc, attempt_floor=0.0, n_total=4)
    assert result["n_matched"] == 0
    assert result["matched_indices"] == []


def test_matched_subset_degenerate_excluded_by_floor():
    # n_total=10; S0 commits on 1/10=0.1, below floor=0.2 -> excluded.
    pc = {
        "S0": {0},
        "S1": {0, 1, 2, 3, 4, 5},
        "C1": {0, 1, 2, 3, 4},
    }
    result = matched_subset(pc, attempt_floor=0.2, n_total=10)
    assert "S0" in result["conditions_excluded"]
    assert "S1" in result["conditions_included"]
    assert "C1" in result["conditions_included"]
    # Intersection of S1 and C1 only = {0,1,2,3,4}
    assert set(result["matched_indices"]) == {0, 1, 2, 3, 4}


def test_matched_subset_all_degenerate():
    pc = {
        "S0": {0},
        "S1": {1},
    }
    result = matched_subset(pc, attempt_floor=0.5, n_total=10)
    assert result["n_matched"] == 0
    assert set(result["conditions_excluded"]) == {"S0", "S1"}
    assert result["conditions_included"] == []


def test_matched_subset_empty_input():
    result = matched_subset({}, attempt_floor=0.2, n_total=10)
    assert result["n_matched"] == 0
    assert result["matched_indices"] == []
    assert result["conditions_included"] == []
    assert result["conditions_excluded"] == []


def test_matched_subset_n_total_zero_no_exclusion():
    # n_total=0 means we skip the floor check -> all conditions included.
    pc = {
        "S0": {0, 1},
        "S1": {1, 2},
    }
    result = matched_subset(pc, attempt_floor=0.9, n_total=0)
    # Both included despite floor because n_total==0 skips the coverage gate.
    assert set(result["conditions_included"]) == {"S0", "S1"}
    assert set(result["matched_indices"]) == {1}


def test_matched_subset_sorted_output():
    pc = {
        "A": {5, 3, 1, 0, 4},
        "B": {4, 3, 2, 1, 0},
    }
    result = matched_subset(pc, attempt_floor=0.0, n_total=6)
    assert result["matched_indices"] == sorted(result["matched_indices"])


def test_matched_subset_single_condition():
    pc = {"S0": {0, 2, 4}}
    result = matched_subset(pc, attempt_floor=0.0, n_total=5)
    assert set(result["matched_indices"]) == {0, 2, 4}
    assert result["n_matched"] == 3


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok: {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
