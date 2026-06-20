"""Tests for the 2026-05-31 batch additions:

- ``loss_polarity`` flag (Option A / Zhu-strict).
- ``advantage_norm`` global_mean / global_mean_std modes.
- SC (self-consistency LOO) rescale block.

Run on a host with torch + math_verify available:

    PYTHONPATH=src pytest tests/test_polarity_global_sc.py -v

These mirror the source-rewrite math in
``src/asym_rewards/grpo.py::_install_prepare_inputs_guards`` via pure-torch
equivalents and assert the documented behaviour.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from asym_rewards.grpo import (  # noqa: E402
    ADVANTAGE_NORM_MODES,
    LOSS_POLARITY_MODES,
)
from asym_rewards.reward import CONDITIONS  # noqa: E402


# ---------------------------------------------------------------------------
# Pure-torch equivalents of the patched advantage / rescale math.
# Keep these in lockstep with grpo.py STD_FLOOR_REPLACEMENT and the SC block.
# ---------------------------------------------------------------------------

def _polarity_mask(rewards: torch.Tensor, polarity: str) -> torch.Tensor:
    if polarity == "positive_only":
        return rewards > 0
    if polarity == "negative_only":
        return rewards < 0
    return torch.ones_like(rewards, dtype=torch.bool)


def _group_masked_mean_std(rewards: torch.Tensor, polarity_mask: torch.Tensor, G: int):
    """Per-group masked mean and (unbiased) std over the polarity-allowed subset."""
    B = rewards.numel() // G
    grouped = rewards.view(B, G)
    mask = polarity_mask.view(B, G).to(rewards.dtype)
    mask_count = mask.sum(dim=1)
    safe_count = mask_count.clamp_min(1.0)
    masked_sum = (grouped * mask).sum(dim=1)
    mean_g = masked_sum / safe_count
    diff = (grouped - mean_g.unsqueeze(1)) * mask
    var_denom = (safe_count - 1.0).clamp_min(1.0)
    var_g = (diff * diff).sum(dim=1) / var_denom
    std_g = var_g.sqrt()
    empty = mask_count == 0
    mean_g = torch.where(empty, torch.zeros_like(mean_g), mean_g)
    std_g = torch.where(empty, torch.zeros_like(std_g), std_g)
    return mean_g, std_g, mask_count


def _advantages_z_score_polarity(
    rewards: torch.Tensor, G: int, polarity: str, std_floor: float = 1e-4
) -> torch.Tensor:
    """Z-score advantages with Option A polarity masking."""
    mask = _polarity_mask(rewards, polarity)
    if polarity == "both":
        mean_g = rewards.view(-1, G).mean(dim=1)
        std_g = rewards.view(-1, G).std(dim=1)
    else:
        mean_g, std_g, _ = _group_masked_mean_std(rewards, mask, G)
    mean_full = mean_g.repeat_interleave(G)
    std_full = std_g.repeat_interleave(G)
    low_std = std_full < std_floor
    safe_std = torch.where(low_std, torch.ones_like(std_full), std_full)
    adv = torch.where(low_std, torch.zeros_like(rewards), (rewards - mean_full) / (safe_std + 1e-4))
    if polarity != "both":
        adv = torch.where(mask, adv, torch.zeros_like(adv))
    return adv


def _advantages_global_mean(
    rewards: torch.Tensor, running_mean: float, running_std: float | None = None
) -> torch.Tensor:
    """Global-baseline advantages. running_mean used always; running_std only when not None."""
    g_mean = torch.full_like(rewards, running_mean)
    if running_std is None:
        return rewards - g_mean
    g_std = torch.full_like(rewards, max(running_std, 1e-4))
    return (rewards - g_mean) / (g_std + 1e-4)


def _sc_rescale_pure(rewards: torch.Tensor, extracted: list[str], G: int, delta_sc: float) -> torch.Tensor:
    """Pure-torch SC LOO rescale; mirrors the grpo.py SC block."""
    B = rewards.numel() // G
    loo = []
    for b in range(B):
        group = extracted[b * G:(b + 1) * G]
        for i, ext_i in enumerate(group):
            if not ext_i:
                loo.append(0.0)
            else:
                cnt = sum(1 for j, e in enumerate(group) if j != i and e == ext_i)
                loo.append(float(cnt) / max(G - 1, 1))
    loo_t = torch.tensor(loo, dtype=rewards.dtype)
    wrong = rewards < 0
    multiplier = 1.0 + delta_sc * loo_t
    return torch.where(wrong, rewards * multiplier, rewards)


# ---------------------------------------------------------------------------
# Mode + registry sanity
# ---------------------------------------------------------------------------

def test_advantage_norm_modes_extended():
    assert "global_mean" in ADVANTAGE_NORM_MODES
    assert "global_mean_std" in ADVANTAGE_NORM_MODES
    assert "z_score" in ADVANTAGE_NORM_MODES
    assert "raw_rloo" in ADVANTAGE_NORM_MODES


def test_loss_polarity_modes_registered():
    assert set(LOSS_POLARITY_MODES) == {"both", "positive_only", "negative_only"}


def test_sc_condition_registered():
    assert "SC" in CONDITIONS
    sc = CONDITIONS["SC"]
    assert sc["family"] == "asym"
    assert sc["group_rescale"] == "sc"
    assert "delta_sc" in sc


# ---------------------------------------------------------------------------
# (1) Polarity mask: zero gradient + masked-out absent from baseline statistics
# ---------------------------------------------------------------------------

def test_polarity_positive_only_excludes_wrong_from_baseline_and_grad():
    """One group of G=4 with rewards [+1, -1, +1, -1] (S0).

    PSR: mask keeps the two +1 samples. Mean over correct = +1, std = 0.
    std_floor guard zeros all advantages. Wrong samples' mask=False -> adv=0.
    """
    rewards = torch.tensor([1.0, -1.0, 1.0, -1.0])
    adv = _advantages_z_score_polarity(rewards, G=4, polarity="positive_only")
    # All advantages should be zero (degenerate within-correct variance).
    assert torch.allclose(adv, torch.zeros_like(adv))
    # And explicitly: wrong sample advantages are zero.
    assert adv[1].item() == 0.0
    assert adv[3].item() == 0.0


def test_polarity_negative_only_symmetric():
    """NSR symmetric: mask keeps wrong, std degenerate, advantages = 0."""
    rewards = torch.tensor([1.0, -1.0, 1.0, -1.0])
    adv = _advantages_z_score_polarity(rewards, G=4, polarity="negative_only")
    assert torch.allclose(adv, torch.zeros_like(adv))


def test_polarity_positive_only_nondegenerate_when_variance_in_correct():
    """When within-correct rewards vary (CW-style), PSR gives nonzero gradient."""
    # G=4 group with varying correct rewards (mimics CW's r = alpha * conf).
    rewards = torch.tensor([1.0, -0.5, 0.7, -0.2])
    adv = _advantages_z_score_polarity(rewards, G=4, polarity="positive_only")
    # Positive samples have nonzero advantage; negatives masked to 0.
    assert adv[1].item() == 0.0
    assert adv[3].item() == 0.0
    assert not torch.allclose(adv[0], torch.zeros(1))
    assert not torch.allclose(adv[2], torch.zeros(1))


def test_polarity_baseline_uses_only_allowed_samples():
    """Verify the masked mean equals mean over kept samples only."""
    rewards = torch.tensor([1.0, -0.5, 0.7, -0.2])
    mask = _polarity_mask(rewards, "positive_only")
    mean_g, _, count = _group_masked_mean_std(rewards, mask, G=4)
    # Manually: mean of {1.0, 0.7} = 0.85
    assert abs(mean_g.item() - 0.85) < 1e-6
    assert count.item() == 2.0


def test_polarity_both_unchanged():
    """polarity=both should produce stock z-score advantages."""
    rewards = torch.tensor([1.0, -0.5, 0.7, -0.2])
    adv_pol = _advantages_z_score_polarity(rewards, G=4, polarity="both")
    # Stock z-score over the full G:
    mean = rewards.mean()
    std = rewards.std()
    stock_adv = (rewards - mean) / (std + 1e-4)
    assert torch.allclose(adv_pol, stock_adv)


# ---------------------------------------------------------------------------
# (2) global_mean: FD1 != P0 (differ by predicted offset 3*P0 - 2)
# ---------------------------------------------------------------------------

def test_global_mean_breaks_affine_collapse_fd1_vs_p0():
    """2-outcome group {correct, wrong} on FD1 (1, -2) and P0 (1, 0).

    Under global_mean (running_mean fixed at 0.0 for the test), advantages
    differ by 3*P0 - 2 = FD1, i.e. (FD1 - P0) = (0, -2).
    """
    fd1_rewards = torch.tensor([1.0, -2.0])
    p0_rewards = torch.tensor([1.0, 0.0])
    # Pretend running global mean has converged to 0.0 (e.g., balanced data).
    g_mean = 0.0
    fd1_adv = _advantages_global_mean(fd1_rewards, running_mean=g_mean)
    p0_adv = _advantages_global_mean(p0_rewards, running_mean=g_mean)
    # FD1 = 3*P0 - 2 implies FD1_adv - P0_adv = (3-1)*P0 - 2 = (0, -2)
    diff = fd1_adv - p0_adv
    expected = torch.tensor([0.0, -2.0])
    assert torch.allclose(diff, expected)
    # And explicitly they are not identical (the original collapse claim breaks).
    assert not torch.allclose(fd1_adv, p0_adv)


# ---------------------------------------------------------------------------
# (3) global_mean_std: FD1 vs P0 differ by offset only (std normalizes scale)
# ---------------------------------------------------------------------------

def test_global_mean_std_offset_only_difference():
    """Under global_mean_std (with shared running_std=running_std), the
    multiplicative-scale component of FD1 = 3*P0 - 2 is absorbed into the
    denominator, leaving only the offset surviving.

    Concretely: with running_std_FD1 = 3 * running_std_P0 (since FD1 scales
    rewards by 3), advantages SHOULD be equal up to the offset / running_std.
    We approximate this by using each condition's own running std.
    """
    fd1_rewards = torch.tensor([1.0, -2.0])
    p0_rewards = torch.tensor([1.0, 0.0])
    # Running stats converge to the population mean/std at T=infty. Use sample
    # batch stats as the running-state proxy here.
    fd1_mean = fd1_rewards.mean().item()
    fd1_std = fd1_rewards.std().item()
    p0_mean = p0_rewards.mean().item()
    p0_std = p0_rewards.std().item()
    fd1_adv = _advantages_global_mean(fd1_rewards, fd1_mean, fd1_std)
    p0_adv = _advantages_global_mean(p0_rewards, p0_mean, p0_std)
    # The z-score under each condition's own EMA stats collapses to the same
    # ±1 advantage pattern (per-condition scale invariance restored by std).
    assert torch.allclose(fd1_adv, p0_adv, atol=1e-4)


def test_global_mean_std_shared_running_stats_preserves_affine_relation():
    """Under shared running stats (mean, std), the affine relation
    FD1 = 3*P0 - 2 survives into the advantage space.

    With shared_mean=0, shared_std=1:
      fd1_adv = (1, -2) / 1 = (1, -2)
      p0_adv  = (1,  0) / 1 = (1,  0)
      3*p0_adv - 2 = (1, -2) = fd1_adv.

    This is the "offset survives, scale absorbed into shared std" claim from
    the user spec: shared scale -> fd1 stays 3x-and-shift away from p0, but
    the AVERAGE gradient magnitude is normalized to comparable scale.
    """
    fd1_rewards = torch.tensor([1.0, -2.0])
    p0_rewards = torch.tensor([1.0, 0.0])
    shared_mean = 0.0
    shared_std = 1.0
    fd1_adv = _advantages_global_mean(fd1_rewards, shared_mean, shared_std)
    p0_adv = _advantages_global_mean(p0_rewards, shared_mean, shared_std)
    # Affine relation survives (modulo the eps in safe_std denom).
    expected = 3.0 * p0_adv - 2.0
    assert torch.allclose(fd1_adv, expected, atol=1e-3)


# ---------------------------------------------------------------------------
# (4) SC: LOO agreement multiplier applies to wrong samples only
# ---------------------------------------------------------------------------

def test_sc_loo_agreement_multiplier_wrong_only():
    """G=4 group. Extracted: ['A', 'A', 'B', 'A']. Rewards: [+1, -1, -1, -1].
    LOO agreement for sample 0 ('A'): 2 peers also 'A' out of 3 -> 2/3.
    For sample 1 ('A'): same, 2/3.
    For sample 2 ('B'): 0 peers -> 0.
    For sample 3 ('A'): 2/3.
    With delta_sc=1: wrong rewards multiplied by (1 + 2/3) for 'A'-agreement
    wrong samples; sample 2 multiplied by (1 + 0) = unchanged.
    """
    rewards = torch.tensor([1.0, -1.0, -1.0, -1.0])
    extracted = ["A", "A", "B", "A"]
    out = _sc_rescale_pure(rewards, extracted, G=4, delta_sc=1.0)
    # Correct sample untouched.
    assert out[0].item() == 1.0
    # Wrong samples with 2/3 LOO agreement scaled by 5/3.
    assert abs(out[1].item() - (-1.0 * (1 + 2 / 3))) < 1e-6
    assert abs(out[3].item() - (-1.0 * (1 + 2 / 3))) < 1e-6
    # Singleton wrong cluster: no agreement, no rescale.
    assert out[2].item() == -1.0


def test_sc_silent_sample_zero_agreement():
    """Empty extracted ('') gets 0 LOO agreement (not crashed)."""
    rewards = torch.tensor([1.0, -1.0])
    extracted = ["", ""]
    out = _sc_rescale_pure(rewards, extracted, G=2, delta_sc=1.0)
    # No agreement counted; wrong sample only multiplied by 1.
    assert out[0].item() == 1.0
    assert out[1].item() == -1.0


# ---------------------------------------------------------------------------
# (5) Zhu PSR / NSR: per-sample fixed advantage, no group normalization
# ---------------------------------------------------------------------------

def _advantages_zhu(rewards: torch.Tensor, mode: str) -> torch.Tensor:
    """Pure-torch equivalent of the zhu_psr / zhu_nsr branch."""
    if mode == "zhu_psr":
        mask = rewards > 0
        return mask.to(rewards.dtype)
    if mode == "zhu_nsr":
        mask = rewards < 0
        return -mask.to(rewards.dtype)
    raise ValueError(mode)


def test_zhu_psr_fixed_plus_one_on_correct():
    """G=4 group, rewards = [+1, -1, +1, -1]. PSR adv = [+1, 0, +1, 0]."""
    rewards = torch.tensor([1.0, -1.0, 1.0, -1.0])
    adv = _advantages_zhu(rewards, "zhu_psr")
    expected = torch.tensor([1.0, 0.0, 1.0, 0.0])
    assert torch.allclose(adv, expected)


def test_zhu_nsr_fixed_minus_one_on_wrong():
    """NSR adv = [0, -1, 0, -1]: pushes down the probability of wrong sequences."""
    rewards = torch.tensor([1.0, -1.0, 1.0, -1.0])
    adv = _advantages_zhu(rewards, "zhu_nsr")
    expected = torch.tensor([0.0, -1.0, 0.0, -1.0])
    assert torch.allclose(adv, expected)


def test_zhu_psr_independent_of_reward_magnitude():
    """The fixed-advantage property: scaling rewards 3x leaves PSR adv unchanged.

    This is what makes Zhu PSR break the FD1 = 3*P0 - 2 affine-collapse: the
    advantage no longer scales with the reward, so FD1 and P0 produce identical
    PSR advantages because both have correct samples at the same positions.
    """
    p0_rewards = torch.tensor([1.0, 0.0])  # P0: correct=+1, wrong=0 (sign 0, not <0)
    fd1_rewards = torch.tensor([1.0, -2.0])  # FD1: correct=+1, wrong=-2
    p0_adv = _advantages_zhu(p0_rewards, "zhu_psr")
    fd1_adv = _advantages_zhu(fd1_rewards, "zhu_psr")
    # Both should have advantage = 1 on correct, 0 on wrong (because PSR
    # masks on r > 0 / correct).
    assert p0_adv.tolist() == [1.0, 0.0]
    assert fd1_adv.tolist() == [1.0, 0.0]


def test_zhu_modes_in_registry():
    assert "zhu_psr" in ADVANTAGE_NORM_MODES
    assert "zhu_nsr" in ADVANTAGE_NORM_MODES


def test_sc_correct_samples_never_rescaled():
    """Even with high LOO agreement, correct samples (r > 0) untouched."""
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
    extracted = ["A", "A", "A", "A"]
    out = _sc_rescale_pure(rewards, extracted, G=4, delta_sc=1.0)
    assert torch.allclose(out, rewards)
