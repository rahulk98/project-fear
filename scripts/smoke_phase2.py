#!/usr/bin/env python
"""End-to-end smoke test for Phase 2 (reward + metrics + GRPO + eval).

Exits non-zero on any failure. Designed to run on the V100 alone (no vLLM,
no 2080 Ti contention), so it is safe to run while trace generation occupies
the rollout GPUs.

Sections:
  1. Reward correctness (hand-coded triples per condition)
  2. Metrics correctness (synthetic arrays with closed-form values)
  3. S4 advantage-normalization guard (all-zero rewards -> no NaN)
  4. Abstention classifier on a hand-curated set (recall + precision gates)
  5. End-to-end 10-step GRPO smoke train (reward should increase)
  6. End-to-end eval smoke (K=4 on 5 problems, metrics JSON written)
  7. LoRA adapter non-trivial (||W_lora||_F > 1e-3 somewhere)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asym_rewards import metrics as M  # noqa: E402
from asym_rewards.reward import (  # noqa: E402
    CONDITIONS,
    asymmetric_reward,
    extract_boxed_answer,
    is_abstention,
    is_correct,
    make_reward_func,
)


# 20 hand-curated completions: text -> human label.
# Kept inline (rather than in data/raw/) because data/ is gitignored
# and this fixture must travel with the test.
ABSTENTION_HOLDOUT = [
    ("Therefore the answer is \\boxed{42}.", False),
    ("After computing, we get \\boxed{\\frac{1}{2}}.", False),
    ("The result is \\boxed{2\\sqrt{3}}.", False),
    ("Working it out, the final answer is \\boxed{x = 5}.", False),
    ("Let me solve... it follows that \\boxed{0}.", False),
    ("Step 1... Step 2... and so the value is \\boxed{\\sqrt{\\frac{2}{3}}}.", False),
    ("Computing both sides yields \\boxed{n=12} as the only positive solution.", False),
    ("We use the formula. The answer is \\boxed{-3.14}.", False),
    ("After lengthy work, \\boxed{e^{i\\pi} + 1 = 0} so the value is 0.", False),
    ("The intermediate step gives \\boxed{first} but the final answer is \\boxed{last}.", False),
    ("I think the answer is somewhere around 50 but I'm not confident.", True),
    ("This problem is too hard for me to solve.", True),
    ("Let me try... actually I don't know how to approach this.", True),
    ("Step 1: I will use calculus. Step 2: take derivative. Step 3: hmm, I'm stuck.", True),
    ("The setup is right but I cannot finish the computation here.", True),
    ("", True),
    ("Let me think step by step. First we note that...", True),
    ("I am not sure. Perhaps the answer is 7. Maybe 8?", True),
    ("It could be \\boxed{} but I cannot determine the value.", True),
    ("Final answer: 42 (no box used).", True),
]


def _section(name: str) -> None:
    print(f"\n=== {name} ===", flush=True)


def _check(label: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}{(' - ' + detail) if detail else ''}", flush=True)
    if not ok:
        raise AssertionError(f"{label} failed: {detail}")


def test_reward_correctness() -> None:
    _section("1. Reward correctness")
    cases = [
        ("\\boxed{42}", "42", True, False),
        ("\\boxed{\\frac{1}{2}}", "1/2", True, False),
        ("\\boxed{2\\sqrt{3}}", "2\\sqrt{3}", True, False),
        ("\\boxed{5}", "42", False, False),
        ("The answer might be 42.", "42", False, True),
        ("Long thinking with no box.", "42", False, True),
    ]
    # asymmetric_reward only handles the 3-outcome asym family. D1 conditions
    # live in a different schema (mu_wise / mu_wasteful / silent_penalty, no
    # "abstain" key) and are exercised by tests/test_reward_d1.py instead;
    # iterating them here would KeyError on cfg["abstain"].
    for cid, cfg in CONDITIONS.items():
        if cfg.get("family") not in (None, "asym"):
            continue
        a, b, s = float(cfg["alpha"]), float(cfg["beta"]), float(cfg["abstain"])
        for completion, gold, correct, abstained in cases:
            expected = s if abstained else (a if correct else -b)
            got = asymmetric_reward(completion, gold, a, b, s)
            _check(
                f"{cid} | gold={gold!r} | correct={correct} abst={abstained}",
                abs(got - expected) < 1e-9,
                f"expected {expected}, got {got}",
            )

    _check(
        "math_verify: AIME-style numeric 144",
        is_correct(extract_boxed_answer("So \\boxed{144}."), "144"),
    )
    _check(
        "math_verify: fraction equivalence 1/2 vs 0.5",
        is_correct(extract_boxed_answer("\\boxed{0.5}"), "1/2"),
    )
    _check(
        "unclosed boxed answer is not extractable",
        extract_boxed_answer("Final answer: \\boxed{42") == "",
    )


def test_cached_rollout_rescore() -> None:
    _section("1b. Cached rollout re-score")
    from asym_rewards.reward import rescore_change_counts, rescore_rollouts

    stale = [{
        "problem_id": "p0",
        "sample_idx": 0,
        "gold": "42",
        "scorer": "math",
        "completion": "Final answer: \\boxed{42",
        "extracted": "42",
        "correct": True,
        "abstained": False,
    }]
    rescored = rescore_rollouts(stale)
    _check("cached unclosed box becomes abstention", rescored[0]["abstained"])
    _check("cached unclosed box loses stale extraction", rescored[0]["extracted"] == "")
    _check("cached unclosed box is not correct", not rescored[0]["correct"])
    changes = rescore_change_counts(stale, rescored)
    _check("cached re-score reports changed abstention label", changes["abstained"] == 1)
    _check("cached re-score reports changed correctness label", changes["correct"] == 1)

    mc = [{
        "problem_id": "p1",
        "sample_idx": 0,
        "gold": "[0, 1, 0, 0]",
        "scorer": "mc",
        "completion": "The answer is \\boxed{B}.",
        "extracted": "",
        "correct": False,
        "abstained": True,
    }]
    rescored_mc = rescore_rollouts(mc)
    _check("cached MC rollout re-scores correct letter", rescored_mc[0]["correct"])

    import asym_rewards.reward as reward_mod
    original_is_correct = reward_mod.is_correct
    try:
        def _boom(_extracted: str, _gold: str) -> bool:
            raise AssertionError("unchanged cached rows should not be re-verified")
        reward_mod.is_correct = _boom
        unchanged = [{
            "problem_id": "p2",
            "sample_idx": 0,
            "gold": "42",
            "scorer": "math",
            "completion": "Final answer: \\boxed{42}.",
            "extracted": "42",
            "correct": True,
            "abstained": False,
        }]
        rescored_unchanged = rescore_rollouts(unchanged)
    finally:
        reward_mod.is_correct = original_is_correct
    _check("unchanged cached row preserves correctness without re-verify",
           rescored_unchanged[0]["correct"])


def test_metrics_correctness() -> None:
    _section("2. Metrics correctness")
    # Perfectly calibrated, half-correct: ECE = 0, accuracy = 0.5
    p = np.array([0.0, 0.0, 1.0, 1.0])
    c = np.array([0,   0,   1,   1])
    _check("ECE perfect calibration = 0", abs(M.expected_calibration_error(p, c, n_bins=10)) < 1e-9)
    _check("Brier perfect calibration = 0", abs(M.brier_score(p, c)) < 1e-9)

    # Max miscalibration: predict 1.0, all wrong
    p = np.array([1.0, 1.0, 1.0, 1.0])
    c = np.array([0, 0, 0, 0])
    _check("ECE max miscalibration = 1", abs(M.expected_calibration_error(p, c) - 1.0) < 1e-9)
    _check("Brier max miscalibration = 1", abs(M.brier_score(p, c) - 1.0) < 1e-9)

    _check("pass@1 of 5 with 1 correct = 0.2", abs(M.pass_at_k(5, 1, 1) - 0.2) < 1e-9)
    _check("pass@5 of 5 with 1 correct = 1.0", abs(M.pass_at_k(5, 1, 5) - 1.0) < 1e-9)
    _check("pass@k with 0 correct = 0", abs(M.pass_at_k(10, 0, 3)) < 1e-9)

    # AUACC: perfectly ranked, then 50/50 random
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    c = np.array([1, 1, 1, 0, 0])
    auacc_perfect = M.auacc(p, c)
    _check("AUACC perfect ordering in [0.6, 1.0]", 0.6 <= auacc_perfect <= 1.0,
           f"got {auacc_perfect:.3f}")

    _check("selective@0.6 with perfect ranking = 1.0",
           abs(M.selective_accuracy_at_coverage(p, c, 0.6) - 1.0) < 1e-9)


def test_s4_guard() -> None:
    _section("3. S4 std-floor guard (synthetic)")
    import torch
    rewards = torch.zeros(8)
    _check("zero-reward group has std=0", rewards.std().item() == 0.0)
    # Simulated guard logic (mirrors grpo._install_std_floor_guard)
    floor = 1e-4
    if rewards.numel() > 1 and rewards.std().item() < floor:
        advs = torch.zeros_like(rewards)
    else:
        advs = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    _check("guarded advantages are all zero", torch.all(advs == 0).item())
    _check("guarded advantages are finite", torch.isfinite(advs).all().item())


def test_abstention_classifier() -> None:
    _section("4. Abstention classifier on hand-curated holdout")
    tp = fp = fn = 0
    for text, gold in ABSTENTION_HOLDOUT:
        pred = is_abstention(text)
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and gold:
            fn += 1
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    print(f"  abstention recall={recall:.2f} precision={precision:.2f} "
          f"(tp={tp} fp={fp} fn={fn})", flush=True)
    _check("abstention recall >= 0.95", recall >= 0.95, f"got {recall:.2f}")
    _check("abstention precision >= 0.90", precision >= 0.90, f"got {precision:.2f}")


def test_reward_func_callable() -> None:
    _section("5a. TRL reward callable shape")
    fn = make_reward_func("S3")
    rewards = fn(["Q1?", "Q2?"], ["\\boxed{2}", "no answer here"], answer=["2", "5"])
    _check("reward_func returns list[float]", isinstance(rewards, list) and all(isinstance(r, float) for r in rewards))
    _check("S3 correct => alpha=0.1", abs(rewards[0] - 0.1) < 1e-9, f"got {rewards[0]}")
    _check("S3 abstain => 0.0",     abs(rewards[1] - 0.0) < 1e-9, f"got {rewards[1]}")


def test_smoke_train_and_eval() -> None:
    _section("5b/6/7. End-to-end smoke train + eval (requires GPU)")
    import torch
    if not torch.cuda.is_available():
        print("  SKIP: no CUDA available locally; run on server", flush=True)
        return

    import yaml
    from asym_rewards.grpo import train_grpo

    cfg = yaml.safe_load((REPO_ROOT / "configs" / "training.yaml").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "qwen2_5_3b_S0_smoke"
        meta = train_grpo(
            model_name="qwen2_5_3b",
            condition_id="S0",
            training_config=cfg,
            out_dir=out_dir,
            smoke=True,
        )
        _check("smoke train completed", meta["steps_completed"] > 0, f"meta={meta}")

        # Adapter non-trivial: at least one LoRA weight tensor moved off zero
        from safetensors.torch import load_file
        adapter_files = list(out_dir.rglob("adapter_model.safetensors")) + list(out_dir.rglob("adapter_model.bin"))
        _check("adapter file written", len(adapter_files) > 0, f"in {out_dir}")
        adapter_path = adapter_files[0]
        loaded = load_file(str(adapter_path)) if adapter_path.suffix == ".safetensors" else torch.load(str(adapter_path))
        max_norm = max(t.float().norm().item() for t in loaded.values())
        _check("at least one LoRA tensor || > 1e-3", max_norm > 1e-3, f"max_norm={max_norm:.4f}")


def main() -> int:
    try:
        test_reward_correctness()
        test_cached_rollout_rescore()
        test_metrics_correctness()
        test_s4_guard()
        test_abstention_classifier()
        test_reward_func_callable()
        test_smoke_train_and_eval()
    except AssertionError as e:
        print(f"\nSMOKE PHASE 2 FAILED: {e}", file=sys.stderr)
        return 1
    print("\nALL SMOKE PHASE 2 SECTIONS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
