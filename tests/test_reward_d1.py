"""D1 reward dispatch + position-locked IDK detection + side-channel tests.

Run where `math_verify` is installed:

    cd project-fear
    pytest tests/test_reward_d1.py -v

Locally on macOS (math_verify not installed), the math_verify import will
fail in the module under test; the importorskip below keeps the file
loadable for tooling but pytest will mark every test SKIPPED.
"""
from __future__ import annotations

import pytest

math_verify = pytest.importorskip("math_verify")

from asym_rewards.reward import (  # noqa: E402
    CONDITIONS,
    OUTCOME_CORRECT,
    OUTCOME_SILENT,
    OUTCOME_WASTEFUL_IDK,
    OUTCOME_WISE_IDK,
    OUTCOME_WRONG,
    _LAST_BATCH_TELEMETRY,
    d1_reward,
    is_idk_marker,
    make_reward_func,
)


# ---------------------------------------------------------------------------
# Position-locked IDK detector
# ---------------------------------------------------------------------------

def test_idk_marker_accepts_canonical_forms():
    for s in ["IDK", "idk", "?", "Idk", "idk.", "IDK ", "?."]:
        assert is_idk_marker(s), s


def test_idk_marker_rejects_non_idk():
    for s in ["42", "1/2", "", "I don't know", "idk maybe", "??"]:
        assert not is_idk_marker(s), s


def test_d1_dispatch_cot_leaked_idk_is_not_idk():
    """A 'I don't know' inside the CoT but a correct \\boxed{42} -> correct."""
    r, outcome, info = d1_reward(
        "Hmm, I do not know honestly. But let me try... \\boxed{42}",
        "42",
        alpha=1.0, beta=1.0, mu_wise=0.5, mu_wasteful=0.5, silent_penalty=1.5,
    )
    assert outcome == OUTCOME_CORRECT
    assert info["idk_flag"] is False
    assert r == 1.0


# ---------------------------------------------------------------------------
# Five-outcome dispatch
# ---------------------------------------------------------------------------

def test_correct_outcome():
    r, outcome, _ = d1_reward(
        "computation ... \\boxed{42}", "42",
        alpha=1.0, beta=1.0, mu_wise=0.5, mu_wasteful=0.5, silent_penalty=1.5,
    )
    assert outcome == OUTCOME_CORRECT
    assert r == 1.0


def test_wrong_outcome():
    r, outcome, _ = d1_reward(
        "computation ... \\boxed{7}", "42",
        alpha=1.0, beta=1.0, mu_wise=0.5, mu_wasteful=0.5, silent_penalty=1.5,
    )
    assert outcome == OUTCOME_WRONG
    assert r == -1.0


def test_silent_outcome():
    r, outcome, _ = d1_reward(
        "no answer here, ran out of tokens", "42",
        alpha=1.0, beta=1.0, mu_wise=0.5, mu_wasteful=0.5, silent_penalty=1.5,
    )
    assert outcome == OUTCOME_SILENT
    assert r == -1.5


def test_idk_no_forced_decode_is_wise_default():
    """Offline / no-forced-decode mode: any IDK box gets wise reward."""
    r, outcome, info = d1_reward(
        "stuck on this. \\boxed{IDK}", "42",
        alpha=1.0, beta=1.0, mu_wise=0.5, mu_wasteful=0.5, silent_penalty=1.5,
    )
    assert outcome == OUTCOME_WISE_IDK
    assert r == 0.5
    assert info["idk_flag"] is True
    assert info["forced_decode_s"] == 0.0
    assert info["forced_answer"] is None


def test_idk_forced_decode_correct_is_wasteful():
    def fd(_prompt, _completion):
        return "42"
    r, outcome, info = d1_reward(
        "stuck on this. \\boxed{IDK}", "42",
        alpha=1.0, beta=1.0, mu_wise=0.5, mu_wasteful=0.5, silent_penalty=1.5,
        forced_decode_fn=fd, prompt="solve",
    )
    assert outcome == OUTCOME_WASTEFUL_IDK
    assert r == -0.5
    assert info["forced_correct"] is True
    assert info["forced_decode_s"] > 0


def test_idk_forced_decode_wrong_is_wise():
    def fd(_prompt, _completion):
        return "999"
    r, outcome, _ = d1_reward(
        "stuck on this. \\boxed{IDK}", "42",
        alpha=1.0, beta=1.0, mu_wise=0.5, mu_wasteful=0.5, silent_penalty=1.5,
        forced_decode_fn=fd, prompt="solve",
    )
    assert outcome == OUTCOME_WISE_IDK
    assert r == 0.5


# ---------------------------------------------------------------------------
# Reward hierarchy under the locked design
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cond_id", ["D1low", "D1mid", "D1asym", "FD1"])
def test_reward_hierarchy_silent_lt_wrong_lt_wasteful_lt_zero_lt_wise_lt_correct(cond_id):
    cfg = CONDITIONS[cond_id]
    a, b = cfg["alpha"], cfg["beta"]
    mu_w, mu_wa, silent = cfg["mu_wise"], cfg["mu_wasteful"], cfg["silent_penalty"]
    assert -silent < -b
    assert -b < -mu_wa
    assert -mu_wa < 0
    assert 0 < mu_w
    assert mu_w < a


# ---------------------------------------------------------------------------
# Batched make_reward_func + telemetry side channel
# ---------------------------------------------------------------------------

def test_d1_batch_populates_telemetry_in_order():
    fn = make_reward_func("D1mid")  # forced_decode_fn=None -> IDK -> wise
    completions = [
        "...\\boxed{42}",       # correct
        "...\\boxed{99}",       # wrong
        "...\\boxed{IDK}",      # wise_idk (no fd fn)
        "no box at all",        # silent
    ]
    rewards = fn(
        prompts=["p"] * 4,
        completions=completions,
        answer=["42"] * 4,
    )
    assert rewards == [1.0, -1.0, 0.5, -1.5]
    assert _LAST_BATCH_TELEMETRY["outcomes"] == [
        OUTCOME_CORRECT,
        OUTCOME_WRONG,
        OUTCOME_WISE_IDK,
        OUTCOME_SILENT,
    ]
    assert _LAST_BATCH_TELEMETRY["channel_rewards"] == rewards
    assert _LAST_BATCH_TELEMETRY["idk_flags"] == [False, False, True, False]
    assert _LAST_BATCH_TELEMETRY["n_samples"] == 4


def test_fd1_batch_applies_fear_asymmetry_on_attempt_channel():
    """FD1 = D1 family with alpha=1, beta=2; verify the four outcome rewards
    differ from D1low only on the wrong channel (-2 vs -1) and silent (-2.5 vs -1.5).
    """
    fn = make_reward_func("FD1")
    rewards = fn(
        prompts=["p"] * 4,
        completions=[
            "...\\boxed{42}",   # correct -> +alpha=+1
            "...\\boxed{99}",   # wrong -> -beta=-2 (FEAR asym)
            "...\\boxed{IDK}",  # wise_idk (no fd fn) -> +mu_wise=+0.3
            "no box at all",    # silent -> -silent_penalty=-2.5
        ],
        answer=["42"] * 4,
    )
    assert rewards == [1.0, -2.0, 0.3, -2.5]


def test_asym_backwards_compat_still_works():
    """S0 reward must still return the same values as before D1 landed."""
    fn = make_reward_func("S0")
    rs = fn(prompts=["p"] * 3, completions=[
        "...\\boxed{42}", "...\\boxed{99}", "no box",
    ], answer=["42"] * 3)
    assert rs == [1.0, -1.0, 0.0]  # alpha=+1, beta=1, abstain=0
    # asym family treats IDK box as a WRONG answer (no IDK channel)
    rs_idk = fn(prompts=["p"], completions=["...\\boxed{IDK}"], answer=["42"])
    assert rs_idk == [-1.0]


def test_cw_d1_condition_is_registered_with_attempts_only_weighting():
    cfg = CONDITIONS["CWD1"]
    assert cfg["family"] == "d1"
    assert cfg["weighting"] == "confidence"


def test_fcw_is_asym_family_with_fear_asymmetry_and_conf_weighting():
    """FCW = FEAR + CW: asym family (no IDK channel), beta=2 asymmetric,
    confidence weighting active."""
    cfg = CONDITIONS["FCW"]
    assert cfg["family"] == "asym"
    assert cfg["weighting"] == "confidence"
    assert cfg["alpha"] == 1.0
    assert cfg["beta"] == 2.0
    assert cfg["abstain"] == 0.0


# ---------------------------------------------------------------------------
# Always-IDK attractor sanity check (numerical, from project.md 3.5.3 table)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cond_id,forced_correct_rate,expected_ev",
    [
        # P(forced_correct=False) * mu_wise + P(forced_correct=True) * -mu_wasteful
        ("D1low",  0.5, 0.5 * 0.3  + 0.5 * (-0.3)),   # 0.0
        ("D1mid",  0.5, 0.5 * 0.5  + 0.5 * (-0.5)),   # 0.0
        ("D1asym", 0.5, 0.5 * 0.5  + 0.5 * (-0.4)),   # +0.05
        ("FD1",    0.5, 0.5 * 0.3  + 0.5 * (-0.3)),   # 0.0 (matched IDK channel to D1low)
    ],
)
def test_always_idk_expected_reward_matches_design(cond_id, forced_correct_rate, expected_ev):
    cfg = CONDITIONS[cond_id]
    ev = (1 - forced_correct_rate) * cfg["mu_wise"] + forced_correct_rate * (-cfg["mu_wasteful"])
    assert ev == pytest.approx(expected_ev, abs=1e-9)
