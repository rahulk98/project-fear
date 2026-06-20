"""Tests for the OCP/CMF/GRF fear-overconfidence variants.

Run where math_verify is installed:

    cd project-fear
    PYTHONPATH=src pytest tests/test_reward_fear_v2.py -v

These tests cover:
- CONDITIONS entries for OCP/CMF/GRF are present and well-formed.
- make_reward_func returns S0-equivalent raw rewards (the variant-specific
  modulation happens inside the TRL source-rewrite, not in the reward func).
- The rescale formulas (computed in pure torch here, mirroring the source-
  rewrite) match expected values on worked examples.

The variant-specific modulation in grpo.py runs only on a real GRPOTrainer
instance; these tests exercise the math via pure-torch equivalents.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
math_verify = pytest.importorskip("math_verify")

from asym_rewards.reward import (  # noqa: E402
    CONDITIONS,
    make_reward_func,
)


# ---------------------------------------------------------------------------
# Pure-torch equivalents of the source-rewrite math (kept in lockstep with
# CW_RESCALE_REPLACEMENT in grpo.py).
# ---------------------------------------------------------------------------

def _apply_ocp(rewards: torch.Tensor, conf: torch.Tensor, lambda_oc: float, tau_oc: float) -> torch.Tensor:
    wrong_mask = rewards < 0
    extra_pen = torch.clamp(conf - tau_oc, min=0.0) * lambda_oc
    return torch.where(wrong_mask, rewards - extra_pen, rewards)


def _apply_cmf(rewards: torch.Tensor, conf: torch.Tensor, gamma_cmf: float) -> torch.Tensor:
    wrong_mask = rewards < 0
    multiplier = 1.0 + gamma_cmf * conf
    return torch.where(wrong_mask, rewards * multiplier, rewards)


def _apply_grf(rewards: torch.Tensor, num_generations: int, delta_grf: float) -> torch.Tensor:
    G = num_generations
    assert rewards.numel() % G == 0
    B = rewards.numel() // G
    grouped = rewards.view(B, G)
    is_correct = (grouped > 0).to(grouped.dtype)
    correct_count = is_correct.sum(dim=1, keepdim=True)
    peer_rate = (correct_count - is_correct) / float(G - 1)
    wrong_mask = grouped < 0
    multiplier = 1.0 + delta_grf * peer_rate
    grouped = torch.where(wrong_mask, grouped * multiplier, grouped)
    return grouped.view(-1)


# ---------------------------------------------------------------------------
# CONDITIONS entries
# ---------------------------------------------------------------------------

def test_ocp_condition_present_and_wellformed():
    cfg = CONDITIONS["OCP"]
    assert cfg["family"] == "asym"
    assert cfg["alpha"] == 1.0
    assert cfg["beta"] == 1.0
    assert cfg["abstain"] == 0.0
    assert cfg["weighting"] == "confidence"
    assert cfg["wrong_rescale"] == "ocp"
    assert cfg["lambda_oc"] == 2.0
    assert cfg["tau_oc"] == 0.5


def test_cmf_condition_present_and_wellformed():
    cfg = CONDITIONS["CMF"]
    assert cfg["family"] == "asym"
    assert cfg["alpha"] == 1.0
    assert cfg["beta"] == 1.0
    assert cfg["abstain"] == 0.0
    assert cfg["weighting"] == "confidence"
    assert cfg["wrong_rescale"] == "cmf"
    assert cfg["gamma_cmf"] == 1.0


def test_grf_condition_present_and_wellformed():
    cfg = CONDITIONS["GRF"]
    assert cfg["family"] == "asym"
    assert cfg["alpha"] == 1.0
    assert cfg["beta"] == 1.0
    assert cfg["abstain"] == 0.0
    assert cfg["group_rescale"] == "grf"
    assert cfg["delta_grf"] == 1.0
    assert "weighting" not in cfg  # GRF does not need conf


# ---------------------------------------------------------------------------
# Reward function returns S0-equivalent raw values (pre-rescale)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cond_id", ["OCP", "CMF", "GRF"])
def test_reward_function_returns_s0_raw_values(cond_id):
    """The reward function itself returns +alpha / -beta / +abstain; the
    variant-specific modulation happens later in the trainer source-rewrite."""
    fn = make_reward_func(cond_id)
    completions = [
        "thinking ... \\boxed{42}",        # correct
        "thinking ... \\boxed{0}",         # wrong (gold != 0)
        "thinking ... no box here",        # silent
    ]
    golds = ["42", "42", "42"]
    rewards = fn(prompts=["p"] * 3, completions=completions, answer=golds)
    assert rewards == [1.0, -1.0, 0.0], (cond_id, rewards)


# ---------------------------------------------------------------------------
# OCP math
# ---------------------------------------------------------------------------

def test_ocp_uncertain_wrong_is_s0_baseline():
    """conf <= tau_oc -> no extra penalty; reward stays at -beta."""
    rewards = torch.tensor([1.0, -1.0, -1.0, 0.0])
    conf = torch.tensor([0.9, 0.3, 0.5, 0.1])
    out = _apply_ocp(rewards, conf, lambda_oc=2.0, tau_oc=0.5)
    assert out.tolist() == [1.0, -1.0, -1.0, 0.0]


def test_ocp_confident_wrong_gets_extra_penalty():
    rewards = torch.tensor([1.0, -1.0, -1.0, -1.0])
    conf = torch.tensor([0.9, 0.6, 0.9, 1.0])
    out = _apply_ocp(rewards, conf, lambda_oc=2.0, tau_oc=0.5)
    # correct (r=+1) untouched; wrong samples get -lambda*(conf-tau).
    assert out[0].item() == 1.0
    assert pytest.approx(out[1].item(), abs=1e-6) == -1.0 - 2.0 * 0.1  # -1.2
    assert pytest.approx(out[2].item(), abs=1e-6) == -1.0 - 2.0 * 0.4  # -1.8
    assert pytest.approx(out[3].item(), abs=1e-6) == -1.0 - 2.0 * 0.5  # -2.0


def test_ocp_identity_when_lambda_is_zero():
    rewards = torch.tensor([1.0, -1.0, -1.0, 0.0])
    conf = torch.tensor([0.9, 0.9, 0.9, 0.9])
    out = _apply_ocp(rewards, conf, lambda_oc=0.0, tau_oc=0.5)
    assert out.tolist() == [1.0, -1.0, -1.0, 0.0]


# ---------------------------------------------------------------------------
# CMF math
# ---------------------------------------------------------------------------

def test_cmf_correct_channel_untouched():
    rewards = torch.tensor([1.0, 1.0])
    conf = torch.tensor([0.9, 0.1])
    out = _apply_cmf(rewards, conf, gamma_cmf=1.0)
    assert out.tolist() == [1.0, 1.0]


def test_cmf_wrong_channel_scales_with_conf():
    rewards = torch.tensor([-1.0, -1.0, -1.0])
    conf = torch.tensor([0.3, 0.5, 0.9])
    out = _apply_cmf(rewards, conf, gamma_cmf=1.0)
    assert pytest.approx(out[0].item(), abs=1e-6) == -1.0 * 1.3
    assert pytest.approx(out[1].item(), abs=1e-6) == -1.0 * 1.5
    assert pytest.approx(out[2].item(), abs=1e-6) == -1.0 * 1.9


def test_cmf_identity_when_gamma_is_zero():
    rewards = torch.tensor([1.0, -1.0, -1.0, 0.0])
    conf = torch.tensor([0.9, 0.9, 0.5, 0.1])
    out = _apply_cmf(rewards, conf, gamma_cmf=0.0)
    assert out.tolist() == [1.0, -1.0, -1.0, 0.0]


# ---------------------------------------------------------------------------
# GRF math
# ---------------------------------------------------------------------------

def test_grf_all_wrong_group_is_s0_baseline():
    """In an all-wrong group peer_rate=0 -> R_wrong stays at -beta."""
    rewards = torch.tensor([-1.0, -1.0, -1.0, -1.0])  # one group of G=4, all wrong
    out = _apply_grf(rewards, num_generations=4, delta_grf=1.0)
    assert out.tolist() == [-1.0, -1.0, -1.0, -1.0]


def test_grf_one_wrong_amongst_correct_peers_gets_max_penalty():
    """G=4 group with 3 correct + 1 wrong: wrong sample's peer_rate=1 -> -2*beta."""
    rewards = torch.tensor([1.0, 1.0, 1.0, -1.0])
    out = _apply_grf(rewards, num_generations=4, delta_grf=1.0)
    assert out[0].item() == 1.0
    assert out[1].item() == 1.0
    assert out[2].item() == 1.0
    assert pytest.approx(out[3].item(), abs=1e-6) == -1.0 * (1.0 + 1.0 * 1.0)  # -2


def test_grf_mixed_group_arithmetic():
    """G=4 group with 2 correct + 2 wrong: each wrong sees peer_rate = 2/3."""
    rewards = torch.tensor([1.0, 1.0, -1.0, -1.0])
    out = _apply_grf(rewards, num_generations=4, delta_grf=1.0)
    expected_wrong = -1.0 * (1.0 + 1.0 * (2.0 / 3.0))  # ~ -1.667
    assert pytest.approx(out[2].item(), abs=1e-6) == expected_wrong
    assert pytest.approx(out[3].item(), abs=1e-6) == expected_wrong


def test_grf_identity_when_delta_is_zero():
    rewards = torch.tensor([1.0, 1.0, -1.0, -1.0])
    out = _apply_grf(rewards, num_generations=4, delta_grf=0.0)
    assert out.tolist() == [1.0, 1.0, -1.0, -1.0]


def test_grf_silent_samples_untouched():
    """Abstention (reward=0) is not in wrong_mask so it stays at 0."""
    rewards = torch.tensor([1.0, 0.0, -1.0, -1.0])  # one correct, one silent, two wrong
    out = _apply_grf(rewards, num_generations=4, delta_grf=1.0)
    assert out[1].item() == 0.0
    # The two wrongs see peer_correct_rate = 1/3 (one correct among the 3 others)
    expected_wrong = -1.0 * (1.0 + 1.0 * (1.0 / 3.0))
    assert pytest.approx(out[2].item(), abs=1e-6) == expected_wrong
    assert pytest.approx(out[3].item(), abs=1e-6) == expected_wrong
