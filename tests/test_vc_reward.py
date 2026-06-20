"""Tests for the vc reward family: classify_vc and build_judge_dataset.

Runs with stdlib and numpy only - no torch, no math_verify, no reward.py
import at module level.

Why no reward.py import here: reward.py imports math_verify at module level,
which is not installed on the MacOS laptop. classify_vc and classify_vc's
underlying judge_label both live in judge.py (with deferred imports from
reward.py inside function bodies). judge_label's deferred import of
extract_boxed_answer from reward.py IS needed at call time, BUT since we
provide completions that always have a parseable or unparseable box, we
can avoid that import by testing classify_vc via a thin stub approach.

Concretely: judge_label does `from asym_rewards.reward import extract_boxed_answer`
inside its body. extract_boxed_answer uses only stdlib (no math_verify).
So calling judge_label locally on the laptop SUCCEEDS as long as asym_rewards
is on sys.path, because reward.py's module-level `from math_verify import ...`
only fires on IMPORT of reward.py - and we never import reward.py here.

The deferred `from asym_rewards.judge import classify_vc` inside _fn_vc (reward.py)
would similarly never fire during these tests because we import from judge.py
directly, not from reward.py.

Run:
    python3 tests/test_vc_reward.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import os
from pathlib import Path

# Ensure src/ is on the path so asym_rewards is importable without install.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Import ONLY from judge.py (no reward.py import at module level).
from asym_rewards.judge import (  # noqa: E402
    JUDGE_PROMPT_TEMPLATE,
    classify_vc,
    build_judge_dataset,
    judge_problem_ids,
    _problem_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_completion_yes() -> str:
    return "This step is fine. \\boxed{YES}"


def _make_completion_no() -> str:
    return "This step has an error. \\boxed{NO}"


def _make_completion_silent() -> str:
    return "I cannot decide."


def _make_completion_unparseable() -> str:
    return "See \\boxed{MAYBE}"


# ---------------------------------------------------------------------------
# classify_vc: correct/wrong/silent mapping for both gold signs
# ---------------------------------------------------------------------------

def test_classify_vc_yes_vote_gold_pos1_is_correct():
    """YES vote + gold +1 -> correct."""
    outcome = classify_vc(_make_completion_yes(), gold_rating=1)
    assert outcome == "correct", f"expected 'correct', got {outcome!r}"


def test_classify_vc_no_vote_gold_neg1_is_correct():
    """NO vote + gold -1 -> correct (judge agrees step is wrong)."""
    outcome = classify_vc(_make_completion_no(), gold_rating=-1)
    assert outcome == "correct", f"expected 'correct', got {outcome!r}"


def test_classify_vc_yes_vote_gold_neg1_is_wrong():
    """YES vote + gold -1 -> wrong (judge says correct, human says wrong)."""
    outcome = classify_vc(_make_completion_yes(), gold_rating=-1)
    assert outcome == "wrong", f"expected 'wrong', got {outcome!r}"


def test_classify_vc_no_vote_gold_pos1_is_wrong():
    """NO vote + gold +1 -> wrong (judge says wrong, human says correct)."""
    outcome = classify_vc(_make_completion_no(), gold_rating=1)
    assert outcome == "wrong", f"expected 'wrong', got {outcome!r}"


def test_classify_vc_silent_no_box_is_silent():
    """No parseable box -> silent regardless of gold."""
    for gold in (1, -1):
        outcome = classify_vc(_make_completion_silent(), gold_rating=gold)
        assert outcome == "silent", f"gold={gold}: expected 'silent', got {outcome!r}"


def test_classify_vc_unparseable_box_is_silent():
    """Unparseable box content (e.g. MAYBE) -> silent."""
    for gold in (1, -1):
        outcome = classify_vc(_make_completion_unparseable(), gold_rating=gold)
        assert outcome == "silent", f"gold={gold}: expected 'silent', got {outcome!r}"


def test_classify_vc_all_six_combinations():
    """Exhaustive check of all {YES,NO,abstain} x {+1,-1} combinations."""
    expected = {
        (_make_completion_yes(), 1):  "correct",
        (_make_completion_yes(), -1): "wrong",
        (_make_completion_no(),  1):  "wrong",
        (_make_completion_no(),  -1): "correct",
        (_make_completion_silent(), 1):  "silent",
        (_make_completion_silent(), -1): "silent",
    }
    for (completion, gold), want in expected.items():
        got = classify_vc(completion, gold_rating=gold)
        assert got == want, f"completion={completion!r}, gold={gold}: expected {want!r}, got {got!r}"


# ---------------------------------------------------------------------------
# build_judge_dataset: balanced sampling + forbid_ids leak guard
# ---------------------------------------------------------------------------

def _make_synthetic_jsonl(n_rows: int = 2, steps_per_row: int = 2, comps_per_step: int = 3) -> list[dict]:
    """Build synthetic PRM800K-style rows for testing."""
    rows = []
    for r in range(n_rows):
        steps = []
        for s in range(steps_per_row):
            comps = []
            for c in range(comps_per_step):
                # Alternate ratings: +1, -1, +1, -1, ...
                rating = 1 if c % 2 == 0 else -1
                comps.append({"rating": rating, "text": f"step text r{r} s{s} c{c}"})
            steps.append({"completions": comps})
        rows.append({"question": {"problem": f"problem {r}"}, "label": {"steps": steps}})
    return rows


def _write_jsonl(rows: list[dict], path: "str | Path") -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_build_judge_dataset_returns_list_of_dicts():
    rows = _make_synthetic_jsonl(n_rows=1, steps_per_row=1, comps_per_step=2)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = f.name
        _write_jsonl(rows, tmp_path)
    try:
        result = build_judge_dataset(tmp_path, max_pos_per_row=1, max_neg_per_row=1)
        assert isinstance(result, list), "result must be a list"
        for ex in result:
            assert "prompt" in ex, f"missing 'prompt' key: {ex!r}"
            assert "answer" in ex, f"missing 'answer' key: {ex!r}"
            assert "id" in ex, f"missing 'id' key: {ex!r}"
    finally:
        os.unlink(tmp_path)


def test_build_judge_dataset_balanced_sampling():
    """With max_pos=1, max_neg=1 per step, each step yields at most 2 examples."""
    # 2 rows, 2 steps each, 3 comps per step (2 pos, 1 neg with alternating rating)
    rows = _make_synthetic_jsonl(n_rows=2, steps_per_row=2, comps_per_step=3)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = f.name
        _write_jsonl(rows, tmp_path)
    try:
        result = build_judge_dataset(tmp_path, max_pos_per_row=1, max_neg_per_row=1)
        # 2 rows * 2 steps * (at most 1 pos + at most 1 neg) = at most 8 examples
        assert len(result) <= 8, f"got {len(result)} examples, expected <= 8"
        # Check answer values are only YES or NO
        for ex in result:
            assert ex["answer"] in ("YES", "NO"), f"invalid answer: {ex['answer']!r}"
    finally:
        os.unlink(tmp_path)


def test_build_judge_dataset_prompt_uses_template():
    """Prompt should use JUDGE_PROMPT_TEMPLATE with problem and step text."""
    rows = _make_synthetic_jsonl(n_rows=1, steps_per_row=1, comps_per_step=1)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = f.name
        _write_jsonl(rows, tmp_path)
    try:
        result = build_judge_dataset(tmp_path, max_pos_per_row=1, max_neg_per_row=1)
        assert len(result) == 1
        ex = result[0]
        expected_prompt = JUDGE_PROMPT_TEMPLATE.format(
            problem="problem 0",
            step="step text r0 s0 c0",
        )
        assert ex["prompt"] == expected_prompt, (
            f"prompt mismatch.\nGot:      {ex['prompt']!r}\nExpected: {expected_prompt!r}"
        )
    finally:
        os.unlink(tmp_path)


def test_build_judge_dataset_limit_param():
    """limit=1 should process at most 1 row."""
    rows = _make_synthetic_jsonl(n_rows=5, steps_per_row=1, comps_per_step=2)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = f.name
        _write_jsonl(rows, tmp_path)
    try:
        result = build_judge_dataset(tmp_path, max_pos_per_row=1, max_neg_per_row=1, limit=1)
        # Only 1 row processed -> every example belongs to problem 0.
        pid0 = _problem_id("problem 0")
        assert result, "no examples produced under limit=1"
        for ex in result:
            assert ex["problem_id"] == pid0, f"unexpected problem from non-row-0: {ex!r}"
            assert ex["id"].startswith(pid0 + "_"), f"id not keyed on problem_id: {ex['id']!r}"
    finally:
        os.unlink(tmp_path)


def test_build_judge_dataset_forbid_problem_ids_no_leak():
    """forbid_problem_ids with no overlap -> no AssertionError, returns normally."""
    rows = _make_synthetic_jsonl(n_rows=1, steps_per_row=1, comps_per_step=2)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = f.name
        _write_jsonl(rows, tmp_path)
    try:
        # Forbid a problem that is not in this file -> no overlap, no raise.
        forbid = {_problem_id("some other problem not present")}
        result = build_judge_dataset(tmp_path, forbid_problem_ids=forbid)
        assert isinstance(result, list)
    finally:
        os.unlink(tmp_path)


def test_build_judge_dataset_forbid_problem_ids_leak_raises():
    """A test-split problem present in the train file -> D2 AssertionError fires.

    Mirrors the real usage: forbid set comes from judge_problem_ids(test_file).
    """
    rows = _make_synthetic_jsonl(n_rows=2, steps_per_row=1, comps_per_step=2)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = f.name
        _write_jsonl(rows, tmp_path)
    try:
        # judge_problem_ids over the same file == every problem leaks.
        forbid = judge_problem_ids(tmp_path)
        leaked_pid = _problem_id("problem 0")
        assert leaked_pid in forbid
        raised = False
        try:
            build_judge_dataset(tmp_path, forbid_problem_ids=forbid)
        except RuntimeError as exc:
            raised = True
            assert "D2 GUARD" in str(exc), f"message lacks 'D2 GUARD': {exc}"
            assert leaked_pid in str(exc), f"message lacks leaking problem_id: {exc}"
        assert raised, "Expected RuntimeError (D2 guard) was NOT raised"
    finally:
        os.unlink(tmp_path)


def test_build_judge_dataset_answer_matches_rating():
    """Verify YES corresponds to rating +1 and NO to rating -1."""
    rows = [
        {
            "question": {"problem": "p"},
            "label": {"steps": [{"completions": [
                {"rating": 1,  "text": "positive step"},
                {"rating": -1, "text": "negative step"},
            ]}]},
        }
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = f.name
        _write_jsonl(rows, tmp_path)
    try:
        result = build_judge_dataset(tmp_path, max_pos_per_row=1, max_neg_per_row=1)
        assert len(result) == 2
        by_step_text = {ex["prompt"].split("Proposed reasoning step:\n")[1].split("\n")[0]: ex["answer"]
                        for ex in result}
        assert by_step_text.get("positive step") == "YES", f"pos step not YES: {by_step_text}"
        assert by_step_text.get("negative step") == "NO", f"neg step not NO: {by_step_text}"
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_all() -> None:
    tests = [
        test_classify_vc_yes_vote_gold_pos1_is_correct,
        test_classify_vc_no_vote_gold_neg1_is_correct,
        test_classify_vc_yes_vote_gold_neg1_is_wrong,
        test_classify_vc_no_vote_gold_pos1_is_wrong,
        test_classify_vc_silent_no_box_is_silent,
        test_classify_vc_unparseable_box_is_silent,
        test_classify_vc_all_six_combinations,
        test_build_judge_dataset_returns_list_of_dicts,
        test_build_judge_dataset_balanced_sampling,
        test_build_judge_dataset_prompt_uses_template,
        test_build_judge_dataset_limit_param,
        test_build_judge_dataset_forbid_problem_ids_no_leak,
        test_build_judge_dataset_forbid_problem_ids_leak_raises,
        test_build_judge_dataset_answer_matches_rating,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {t.__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
