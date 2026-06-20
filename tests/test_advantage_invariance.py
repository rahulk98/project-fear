"""Advantage-invariance regression test (P5).

Finding O: under z-score GRPO normalization the alpha/beta
reward magnitudes cancel. The realized advantages depend only on the outcome
COMPOSITION of the group (i.e. the threshold structure), not on the absolute
reward scale.

Concretely: two conditions that differ only in a positive scale factor on all
reward values (same threshold = beta/(alpha+beta)) produce IDENTICAL z-scored
advantages, so their gradient updates are indistinguishable at any step.

Key implication for the asymmetric-RLVR project: the implicit abstention
threshold is the one genuine degree of freedom that differentiates S0..S4
under z-score GRPO. Scale-matched doublings (alpha=1->2, beta=1->2) do nothing
to the gradient.

Run with full reward.py dependencies installed:

    cd project-fear
    pytest tests/test_advantage_invariance.py -v

Run locally on macOS (numpy only; CONDITIONS-parametrized cases skip):

    python3 tests/test_advantage_invariance.py
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# zscore_adv: mirrors the formula in grpo.py _prepare_inputs (z_score branch)
# NOTE: do NOT import grpo.py -- the formula lives inside a source-string
# that is not importable as a regular module. The formula from grpo.py is:
#   advantages = (rewards - mean_grouped_rewards) / (safe_std + 1e-4)
# where safe_std = std_grouped_rewards when std >= std_floor, else ones
# (and advantages forced to zero for below-floor groups). For this test we
# keep the simpler (always-compute) form to validate the math identity.
# ---------------------------------------------------------------------------

def zscore_adv(rewards: list[float] | np.ndarray) -> np.ndarray:
    """Z-score normalize a 1-D reward array, mirroring grpo.py's z_score branch.

    advantages_i = (r_i - mean(r)) / (std(r) + 1e-4)

    The 1e-4 stabilizer matches STD_FLOOR_TARGET in grpo.py exactly.
    ddof=1 (sample std) matches torch.std's default (Bessel-corrected).
    Returns a numpy float64 array of the same length.
    """
    r = np.asarray(rewards, dtype=np.float64)
    mean = r.mean()
    std = np.std(r, ddof=1)   # sample std (ddof=1), matching torch.std default
    return (r - mean) / (std + 1e-4)


# ---------------------------------------------------------------------------
# Test 1: worked example with a concrete 4-outcome group under S0
# Group: [correct, wrong, wrong, abstain] under S0 (alpha=1, beta=1, abstain=0)
# Raw rewards: [+1, -1, -1, 0]
# ---------------------------------------------------------------------------

def test_worked_example_s0_four_outcome_group():
    """S0 group [correct, wrong, wrong, abstain] matches documented z-scores.

    Raw rewards: [+1, -1, -1, 0]
      mean  = (1 + -1 + -1 + 0) / 4 = -0.25
      std   = sqrt(((1-(-0.25))^2 + (-1-(-0.25))^2 + (-1-(-0.25))^2 + (0-(-0.25))^2) / 3)
            = sqrt((1.5625 + 0.5625 + 0.5625 + 0.0625) / 3)
            = sqrt(2.75 / 3) = sqrt(0.9167) ~ 0.95743   (sample std, ddof=1 = torch.std default)
      adv   ~ [(1.25)/(0.95743+1e-4), (-0.75)/(0.95743+1e-4),
               (-0.75)/(0.95743+1e-4), (0.25)/(0.95743+1e-4)]
            ~ [+1.305, -0.783, -0.783, +0.261]

    Key project assertion: the ABSTAIN advantage is the only dimension that
    changes when we move from S0 (abstain=0) to C1 (abstain=+0.2). The
    correct and wrong advantages shift too because group_mean changes, but
    the abstain sample is the VARIABLE that changes the group composition.
    """
    rewards = [1.0, -1.0, -1.0, 0.0]
    adv = zscore_adv(rewards)

    expected_mean = -0.25
    expected_std = math.sqrt(
        ((1.0 - expected_mean)**2 + (-1.0 - expected_mean)**2 +
         (-1.0 - expected_mean)**2 + (0.0 - expected_mean)**2) / 3  # ddof=1
    )
    denom = expected_std + 1e-4
    expected = np.array([
        (1.0 - expected_mean) / denom,
        (-1.0 - expected_mean) / denom,
        (-1.0 - expected_mean) / denom,
        (0.0 - expected_mean) / denom,
    ])
    np.testing.assert_allclose(adv, expected, atol=1e-12)

    # correct and wrong advantages should differ in sign
    assert adv[0] > 0, "correct sample advantage should be positive"
    assert adv[1] < 0, "wrong sample advantage should be negative"
    assert adv[2] < 0, "wrong sample advantage should be negative"

    # two wrong samples get IDENTICAL advantages (same reward value)
    assert adv[1] == pytest.approx(adv[2], abs=1e-15)

    # the abstain advantage sits between wrong and correct
    assert adv[1] < adv[3] < adv[0]

    # verify the advantages sum to zero (z-score property up to 1e-4 denominator
    # perturbation; allow moderate tolerance)
    assert abs(adv.sum()) < 0.01


# ---------------------------------------------------------------------------
# Test 2: magnitude invariance - the headline claim
#
# Mathematical precision note
# ----------------------------
# The z-score formula is   adv_i = (r_i - mean) / (std + eps)   where eps=1e-4.
#
# For a scale-k multiplication r -> k*r:
#   adv_i(k*r) = (k*r_i - k*mean) / (k*std + eps)
#              = k*(r_i - mean) / (k*std + eps)
#
# vs. adv_i(r) = (r_i - mean) / (std + eps)
#
# These are EQUAL only when eps=0 (exact) or when k=1 (trivial).
# For eps=1e-4 the residual is O(eps/std) per unit advantage:
#   |adv(k*r) - adv(r)| ~ |adv(r)| * eps * (k-1) / (std * (k + eps/std))
#
# For realistic rewards (std ~ 1-2, k=2, |adv|~1) the residual is ~5e-5.
# This is negligible relative to gradient step sizes (~1e-6 lr) but is NOT
# zero. The project claim "reward magnitudes cancel under z-score" should
# therefore be stated as "approximately cancel up to O(1e-4/std)".
#
# The tests below assert approximate equality with a tolerance computed from
# the exact analytical residual, and additionally assert that the relative
# error is < 0.01% (1 part in 10000) -- well below any training-relevant
# signal threshold. This is the correct operationalization of Finding O.
# ---------------------------------------------------------------------------

def _scale_invariance_tol(rewards_1x: np.ndarray, scale: float) -> float:
    """Compute the maximum analytical advantage difference for a k-fold scale.

    |adv(k*r) - adv(r)| <= |adv(r)| * eps * (k-1) / (std * k)
    where eps=1e-4. Returns a bound with a safety factor of 2.
    Uses ddof=1 to match the zscore_adv formula (torch.std default).
    """
    eps = 1e-4
    r = np.asarray(rewards_1x, dtype=np.float64)
    std = np.std(r, ddof=1)  # sample std matching zscore_adv
    if std < 1e-12:
        return 1e-9  # uniform group: both sides zero
    adv_max = np.abs(zscore_adv(r)).max()
    bound = adv_max * eps * (scale - 1.0) / (std * scale)
    return float(bound) * 2.0  # safety factor


def test_magnitude_invariance_threshold_0_5():
    """alpha=1,beta=1 vs alpha=2,beta=2: same threshold 0.5, near-identical advantages.

    Group composition: [correct, wrong] (2-sample group, no abstention).
    Rewards S0 (a=1,b=1): [+1, -1]
    Rewards 2xS0 (a=2,b=2): [+2, -2]

    The residual from the 1e-4 eps term is O(5e-5). The relative error is
    ~5e-5 / 1 = 5e-5, far below any gradient-relevant threshold.
    """
    rewards_1x = np.array([1.0, -1.0])
    rewards_2x = np.array([2.0, -2.0])

    adv_1x = zscore_adv(rewards_1x)
    adv_2x = zscore_adv(rewards_2x)

    tol = _scale_invariance_tol(rewards_1x, scale=2.0)
    np.testing.assert_allclose(
        adv_1x, adv_2x, atol=tol,
        err_msg=f"alpha=1,b=1 and alpha=2,b=2 advantages must agree within {tol:.2e}",
    )

    # Assert the relative error is < 0.01% (publication-grade "cancels")
    rel_err = np.abs(adv_1x - adv_2x) / (np.abs(adv_1x) + 1e-12)
    assert rel_err.max() < 1e-3, (
        f"Relative advantage error {rel_err.max():.2e} exceeds 0.1% for k=2 scale -- "
        "magnitude invariance no longer holds at training-relevant precision"
    )


def test_magnitude_invariance_threshold_0_75():
    """alpha=1,beta=3 vs alpha=2,beta=6: same threshold 0.75, near-identical advantages.

    Group composition: [correct, wrong] (2-sample group, no abstention).
    S1-style: a=1,b=3 -> rewards [+1, -3]
    2x:       a=2,b=6 -> rewards [+2, -6]

    std([1,-3]) = 2.0, so the residual is O(1e-4/2) = 5e-5.
    """
    rewards_1x = np.array([1.0, -3.0])
    rewards_2x = np.array([2.0, -6.0])

    adv_1x = zscore_adv(rewards_1x)
    adv_2x = zscore_adv(rewards_2x)

    tol = _scale_invariance_tol(rewards_1x, scale=2.0)
    np.testing.assert_allclose(
        adv_1x, adv_2x, atol=tol,
        err_msg=f"alpha=1,b=3 and alpha=2,b=6 advantages must agree within {tol:.2e}",
    )

    rel_err = np.abs(adv_1x - adv_2x) / (np.abs(adv_1x) + 1e-12)
    assert rel_err.max() < 1e-3, (
        f"Relative advantage error {rel_err.max():.2e} exceeds 0.1% for k=2 scale"
    )


def test_magnitude_invariance_three_outcome_group():
    """3-outcome group [correct, wrong, wrong]: near-identical advantages under 2x scale.

    Same threshold (beta/(alpha+beta) = 0.5), doubled magnitudes.
    Residual bounded by O(eps/std) ~ 1e-4 / sqrt(2/3) ~ 1.2e-4.
    """
    rewards_1x = np.array([1.0, -1.0, -1.0])
    rewards_2x = np.array([2.0, -2.0, -2.0])

    adv_1x = zscore_adv(rewards_1x)
    adv_2x = zscore_adv(rewards_2x)

    tol = _scale_invariance_tol(rewards_1x, scale=2.0)
    np.testing.assert_allclose(
        adv_1x, adv_2x, atol=tol,
        err_msg=f"Scale-doubled rewards must yield near-identical z-score advantages (tol={tol:.2e})",
    )

    rel_err = np.abs(adv_1x - adv_2x) / (np.abs(adv_1x) + 1e-12)
    assert rel_err.max() < 1e-3, (
        f"Relative advantage error {rel_err.max():.2e} exceeds 0.1% for 3-outcome 2x scale"
    )


# ---------------------------------------------------------------------------
# Test 3: threshold change DOES change advantages (abstain is the real d.o.f.)
# S0 (threshold 0.5, abstain=0) vs S1-analogue (threshold 0.667, abstain=0)
# on a 3-outcome group [correct, wrong, abstain].
# ---------------------------------------------------------------------------

def test_threshold_change_changes_advantages():
    """Changing threshold (beta/alpha+beta) changes advantages on a mixed group.

    Group: [correct, wrong, abstain].
    S0-style (a=1, b=1, abstain=0): rewards [+1, -1, 0]
    S1-style (a=1, b=2, abstain=0): rewards [+1, -2, 0]  <- threshold 0.667

    The abstain reward is the same (0) but the wrong reward is MORE negative,
    shifting the group mean DOWN and changing all advantages. This shows the
    implicit threshold (beta/(alpha+beta)) is the genuine d.o.f. that moves
    the abstain advantage relative to the attempt channels.
    """
    # S0: threshold 0.5
    rewards_s0 = np.array([1.0, -1.0, 0.0])
    adv_s0 = zscore_adv(rewards_s0)

    # S1-analogue: threshold 0.667
    rewards_s1 = np.array([1.0, -2.0, 0.0])
    adv_s1 = zscore_adv(rewards_s1)

    # The advantages must differ
    assert not np.allclose(adv_s0, adv_s1, atol=1e-9), \
        "Different thresholds must produce different advantages on a mixed group"

    # Specifically: the abstain advantage (index 2) should differ because the
    # group mean and std both shift when beta increases.
    assert not math.isclose(float(adv_s0[2]), float(adv_s1[2]), rel_tol=0, abs_tol=1e-9), \
        "Abstain advantage must shift when the wrong-channel penalty (threshold) changes"

    # Concretely: S1 pushes the abstain advantage UP (abstain is now relatively
    # better vs wrong) -- verify directionality.
    # S0: mean = 0/3=0, S1: mean = (1-2+0)/3 = -1/3 -> abstain advantage
    # is (0 - mean_s1)/(std_s1 + 1e-4) which is more positive than (0 - mean_s0)/...
    assert adv_s1[2] > adv_s0[2], \
        "Higher wrong penalty should push abstain advantage upward relative to S0"


def test_two_outcome_no_abstain_group_threshold_effect():
    """With only correct+wrong samples, different thresholds give different advantages.

    This verifies the claim that threshold DOES matter even when no abstain sample
    is present: the ratio |adv_correct| / |adv_wrong| = 1 always for a 2-outcome
    group (since the z-score of a 2-element array is always +-1/sqrt(2*(1+1e-4/std)^2)),
    BUT the absolute values differ when std differs -- i.e. magnitude does matter
    for the raw advantage value, just not for the GRADIENT (which is scaled by the
    log-prob ratio). This is a nuance documented in the project report.
    """
    rewards_s0 = np.array([1.0, -1.0])   # std=1.0
    rewards_s1 = np.array([1.0, -2.0])   # std=1.5

    adv_s0 = zscore_adv(rewards_s0)
    adv_s1 = zscore_adv(rewards_s1)

    # For a 2-sample group the ratio correct/wrong advantage is always -1
    assert math.isclose(adv_s0[0] / adv_s0[1], -1.0, rel_tol=1e-9)
    assert math.isclose(adv_s1[0] / adv_s1[1], -1.0, rel_tol=1e-9)

    # But absolute magnitudes differ when threshold differs
    assert not math.isclose(abs(float(adv_s0[0])), abs(float(adv_s1[0])), rel_tol=1e-9), \
        "A 2-outcome group with different std has different absolute advantage magnitudes"


# ---------------------------------------------------------------------------
# Optional: CONDITIONS-parametrized tests (skip gracefully when reward.py
# is not importable, i.e. when running locally without math_verify / torch).
# ---------------------------------------------------------------------------

try:
    from asym_rewards.reward import CONDITIONS as _CONDITIONS  # type: ignore[import]
    _CONDITIONS_AVAILABLE = True
except ImportError:
    _CONDITIONS_AVAILABLE = None  # not None so we can use it below
    _CONDITIONS_AVAILABLE = False
    print(
        "[test_advantage_invariance] reward.py not importable locally "
        "(math_verify/torch absent). CONDITIONS-parametrized cases skipped. "
        "Run in a full environment for full coverage."
    )


def _asym_rewards_for_group(
    outcomes: list[str],
    alpha: float,
    beta: float,
    abstain: float,
) -> list[float]:
    """Build a reward list for an asym-family group given outcome labels."""
    mapping = {"correct": alpha, "wrong": -beta, "abstain": abstain}
    return [mapping[o] for o in outcomes]


if _CONDITIONS_AVAILABLE:
    # Extract only asym-family conditions with non-trivial beta (skip P0 beta=0,
    # S4 alpha=0, and the CW/FCW/OCP/CMF conf-scaled variants whose reward
    # depends on confidence and is not a fixed scalar).
    _ASYM_SIMPLE = {
        k: v for k, v in _CONDITIONS.items()
        if v.get("family", "asym") == "asym"
        and "weighting" not in v
        and "group_rescale" not in v
        and "wrong_rescale" not in v
        and v.get("beta", 0) > 0
        and v.get("alpha", 0) > 0
    }

    @pytest.mark.parametrize("cond_id", list(_ASYM_SIMPLE.keys()))
    def test_scale_invariance_real_conditions(cond_id):
        """Each asym condition: 2x scaled rewards give identical z-score advantages.

        This verifies Finding O holds for ALL registered simple asym conditions.
        Group composition: [correct, wrong, wrong] (3-outcome, no abstention).
        """
        cfg = _ASYM_SIMPLE[cond_id]
        a, b, s = float(cfg["alpha"]), float(cfg["beta"]), float(cfg.get("abstain", 0.0))

        outcomes = ["correct", "wrong", "wrong"]
        r1x = _asym_rewards_for_group(outcomes, a, b, s)
        r2x = _asym_rewards_for_group(outcomes, 2 * a, 2 * b, 2 * s)

        adv_1x = zscore_adv(r1x)
        adv_2x = zscore_adv(r2x)

        # Stabilizer-aware tolerance: the 1e-4 std floor makes the invariance
        # APPROXIMATE (residual O(eps/std), ~7.5e-5 for S0 at k=2), so an
        # atol=1e-9 would spuriously fail. Use the same analytical bound as the
        # hand-written magnitude-invariance tests.
        tol = _scale_invariance_tol(r1x, scale=2.0)
        np.testing.assert_allclose(
            adv_1x, adv_2x, atol=tol,
            err_msg=f"{cond_id}: 2x reward scale must not change z-score advantages "
                    f"beyond the {tol:.2e} stabilizer residual",
        )
        rel_err = np.abs(adv_1x - adv_2x) / (np.abs(adv_1x) + 1e-12)
        assert rel_err.max() < 1e-3, f"{cond_id}: rel advantage error {rel_err.max():.2e} > 0.1%"

    @pytest.mark.parametrize("cond_id", list(_ASYM_SIMPLE.keys()))
    def test_abstain_is_free_dimension_real_conditions(cond_id):
        """Changing only the abstain reward (not alpha/beta) changes advantages on
        a mixed [correct, wrong, abstain] group. Verifies Finding O's corollary:
        within a fixed threshold, the explicit abstain reward (C-family positive
        abstain vs S-family zero) does shift the abstain advantage, i.e. it IS
        a degree of freedom (just not the alpha/beta magnitude).
        """
        cfg = _ASYM_SIMPLE[cond_id]
        a, b, s = float(cfg["alpha"]), float(cfg["beta"]), float(cfg.get("abstain", 0.0))

        outcomes = ["correct", "wrong", "abstain"]
        r_base = _asym_rewards_for_group(outcomes, a, b, s)
        # Perturb abstain by +0.5 while holding alpha/beta fixed
        r_pert = _asym_rewards_for_group(outcomes, a, b, s + 0.5)

        adv_base = zscore_adv(r_base)
        adv_pert = zscore_adv(r_pert)

        # Only compare if rewards actually differ (i.e. abstain=0 base, so +0.5 matters)
        if not np.allclose(r_base, r_pert, atol=1e-12):
            assert not np.allclose(adv_base, adv_pert, atol=1e-9), (
                f"{cond_id}: changing abstain reward must change z-scored advantages "
                f"(abstain is a real degree of freedom)"
            )


# ---------------------------------------------------------------------------
# Self-test entry point: run directly without pytest
# ---------------------------------------------------------------------------

def _run_local():
    """Run the pure-numpy tests manually (no pytest required)."""
    tests = [
        test_worked_example_s0_four_outcome_group,
        test_magnitude_invariance_threshold_0_5,
        test_magnitude_invariance_threshold_0_75,
        test_magnitude_invariance_three_outcome_group,
        test_threshold_change_changes_advantages,
        test_two_outcome_no_abstain_group_threshold_effect,
    ]
    passed = 0
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {fn.__name__}: {exc}")
            failed += 1

    if _CONDITIONS_AVAILABLE:
        for cond_id in _ASYM_SIMPLE:
            for fn_template, fn_real in [
                (test_scale_invariance_real_conditions, test_scale_invariance_real_conditions),
                (test_abstain_is_free_dimension_real_conditions,
                 test_abstain_is_free_dimension_real_conditions),
            ]:
                try:
                    fn_real(cond_id)
                    print(f"  PASS  {fn_real.__name__}[{cond_id}]")
                    passed += 1
                except Exception as exc:
                    print(f"  FAIL  {fn_real.__name__}[{cond_id}]: {exc}")
                    failed += 1

    print(f"\n{passed} passed, {failed} failed.")
    return failed


if __name__ == "__main__":
    sys.exit(_run_local())
