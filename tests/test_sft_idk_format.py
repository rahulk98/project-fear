"""Tests for P0-1: SFT IDK completion format.

Verifies that:
1. IDK_COMPLETION_TEMPLATE contains exactly \boxed{IDK} (single braces, not
   double braces {{ }}) so the model learns to emit the correct token sequence.
2. extract_boxed_answer applied to the template returns "IDK".
3. The normalized content ("idk") is in the IDK marker set recognised by
   is_idk_marker (replicated inline to avoid math_verify/torch dependency).

Run:
    python3 tests/test_sft_idk_format.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Import the completion template and the solvable completion builder from the
# SFT script (stdlib-only module; no torch/transformers imports at module level).
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from sft_warmstart_idk import (  # noqa: E402
    IDK_COMPLETION_TEMPLATE,
    PROMPT_TEMPLATE_IDK,
    _make_synthetic_idk_exemplars,
    _load_solvable_exemplars,
)

# Use the dependency-free boxed extractor from asym_rewards._boxed.
from asym_rewards._boxed import extract_boxed_answer  # noqa: E402


# ---------------------------------------------------------------------------
# Inline replication of is_idk_marker from reward.py so this test runs
# without math_verify or torch being installed (laptop environment).
# Logic: strip -> lower -> rstrip('.') -> must be in {"idk", "i.d.k", "?"}.
# ---------------------------------------------------------------------------
_IDK_MARKERS = {"idk", "i.d.k", "?"}


def _is_idk_marker(extracted: str) -> bool:
    if not extracted:
        return False
    s = extracted.strip().lower().rstrip(".").strip()
    return s in _IDK_MARKERS


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_idk_completion_no_double_braces():
    """IDK_COMPLETION_TEMPLATE must NOT contain {{ or }} (raw literal braces)."""
    assert "{{" not in IDK_COMPLETION_TEMPLATE, (
        "IDK_COMPLETION_TEMPLATE contains {{ (double-brace) which is a .format() "
        "escape and would produce a literal single brace only when .format() is "
        "called. This template is used RAW as the SFT completion and must contain "
        "single braces so the model learns \\boxed{IDK}, not \\boxed{{IDK}}."
    )
    assert "}}" not in IDK_COMPLETION_TEMPLATE, (
        "IDK_COMPLETION_TEMPLATE contains }} (double-brace closing escape). "
        "Use single-brace } in raw completion strings."
    )


def test_idk_completion_contains_boxed_idk():
    """IDK_COMPLETION_TEMPLATE must contain the literal string \\boxed{IDK}."""
    assert "\\boxed{IDK}" in IDK_COMPLETION_TEMPLATE, (
        f"IDK_COMPLETION_TEMPLATE does not contain \\boxed{{IDK}}. "
        f"Actual tail: {IDK_COMPLETION_TEMPLATE[-40:]!r}"
    )


def test_extract_boxed_answer_returns_idk():
    """extract_boxed_answer applied to IDK_COMPLETION_TEMPLATE returns 'IDK'."""
    extracted = extract_boxed_answer(IDK_COMPLETION_TEMPLATE)
    assert extracted == "IDK", (
        f"extract_boxed_answer(IDK_COMPLETION_TEMPLATE) returned {extracted!r}, "
        "expected 'IDK'. The completion template has malformed braces."
    )


def test_is_idk_marker_accepts_idk():
    """is_idk_marker('IDK') must be True (case-insensitive)."""
    assert _is_idk_marker("IDK"), "is_idk_marker must accept 'IDK'"
    assert _is_idk_marker("idk"), "is_idk_marker must accept 'idk'"
    assert _is_idk_marker("IDK."), "is_idk_marker must accept 'IDK.' (trailing period)"


def test_completion_pipeline_end_to_end():
    """Full pipeline: IDK_COMPLETION_TEMPLATE -> extract -> is_idk_marker -> True."""
    extracted = extract_boxed_answer(IDK_COMPLETION_TEMPLATE)
    assert _is_idk_marker(extracted), (
        f"end-to-end check failed: extracted={extracted!r}, "
        "is_idk_marker should return True for IDK completions."
    )


def test_prompt_template_double_braces_intentional():
    """PROMPT_TEMPLATE_IDK uses {{ }} which is correct: it goes through .format().

    After .format(problem='...') the output should contain single-brace \boxed{}.
    This verifies the asymmetry: PROMPT templates keep {{ }}, COMPLETION strings use { }.
    """
    formatted = PROMPT_TEMPLATE_IDK.format(problem="What is 2+2?")
    assert "\\boxed{}" in formatted, (
        "PROMPT_TEMPLATE_IDK.format() should produce \\boxed{} in the output."
    )
    assert "\\boxed{{" not in formatted, (
        "After .format(), PROMPT_TEMPLATE_IDK should not contain \\boxed{{ (double brace)."
    )


def test_all_synthetic_idk_exemplars_parse_correctly():
    """Every IDK exemplar from _make_synthetic_idk_exemplars must parse as IDK."""
    exemplars = _make_synthetic_idk_exemplars()
    assert len(exemplars) > 0, "No synthetic IDK exemplars returned"
    for i, ex in enumerate(exemplars):
        comp = ex["completion"]
        extracted = extract_boxed_answer(comp)
        assert _is_idk_marker(extracted), (
            f"Exemplar {i}: extract_boxed_answer={extracted!r}, "
            f"is_idk_marker returned False. Completion: {comp!r}"
        )


def test_solvable_exemplars_do_not_parse_as_idk():
    """Fallback solvable exemplars must NOT parse as IDK markers."""
    exemplars = _load_solvable_exemplars(4)
    assert len(exemplars) > 0, "No solvable exemplars returned"
    for i, ex in enumerate(exemplars):
        comp = ex["completion"]
        extracted = extract_boxed_answer(comp)
        assert not _is_idk_marker(extracted), (
            f"Solvable exemplar {i} incorrectly parses as IDK: "
            f"extracted={extracted!r}, completion={comp!r}"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_all() -> None:
    tests = [
        test_idk_completion_no_double_braces,
        test_idk_completion_contains_boxed_idk,
        test_extract_boxed_answer_returns_idk,
        test_is_idk_marker_accepts_idk,
        test_completion_pipeline_end_to_end,
        test_prompt_template_double_braces_intentional,
        test_all_synthetic_idk_exemplars_parse_correctly,
        test_solvable_exemplars_do_not_parse_as_idk,
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
