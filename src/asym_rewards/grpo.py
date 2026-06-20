"""GRPO LoRA training for the FEAR asymmetric-reward sweep on Qwen-2.5-1.5B.

Wraps `trl.GRPOTrainer` (TRL >= 0.14.0; first release with GRPO) with:
- per-condition asymmetric reward callback (`asym_rewards.reward.make_reward_func`)
- LoRA via peft
- vLLM-accelerated rollouts co-resident with the policy on cuda:0
- a single TRL 0.15 `_prepare_inputs` source patch that guards all-empty
  completions and S4 alpha=0 groups with std < std_floor
- per-call seed override for multi-seed sweeps on key conditions (ICLR error bars)

Training data: MATH-train levels 4-5 + GSM8K-train top-token-length quartile.
This focused-hard-problem set is where the asymmetric signal matters most.

dtype is configured at the YAML level (bf16 on Ampere+ GPUs, fp16 fallback on
Volta/Turing). smoke=True overrides production HP for memory: G=2, batch=2,
prompt=512, completion=256, accum=1. per_device_train_batch_size must be
divisible by num_generations (TRL 0.15.2 constraint), so smoke uses 2/2.
"""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import textwrap
from typing import Any

import pandas as pd
from datasets import Dataset
from peft import LoraConfig

from .data import load_benchmark
from .models import load_model
from .reward import CONDITIONS, make_reward_func


PROMPT_TEMPLATE = (
    "Solve the following problem step by step. "
    "Put your final answer in \\boxed{{}}.\n\n"
    "Problem: {problem}"
)

# IDK-seeded variant (Option A, instruction only): explicitly tells the model
# that \boxed{IDK} is a valid answer when uncertain. The few-shot variant
# (commit c6468e1) showed wrong_frac climbing from 0.75 to 1.0 by step 150
# while still producing zero IDK emissions, suggesting the example was
# polluting training dynamics rather than seeding the channel. Reverting to
# the cleaner instruction-only prompt (commit db8bb59 lineage).
PROMPT_TEMPLATE_IDK = (
    "Solve the following problem step by step. "
    "Put your final answer in \\boxed{{}}. "
    "If you cannot solve the problem, write \\boxed{{IDK}} as your answer.\n\n"
    "Problem: {problem}"
)

HF_BASE_MODEL_IDS = {
    "qwen2_5_0_5b": "Qwen/Qwen2.5-0.5B",
    "qwen2_5_1_5b": "Qwen/Qwen2.5-1.5B",
    "qwen2_5_math_1_5b": "Qwen/Qwen2.5-Math-1.5B",
    "qwen2_5_3b": "Qwen/Qwen2.5-3B",
    "llama3_2_1b": "meta-llama/Llama-3.2-1B",
}

HF_UPLOAD_ALLOW_PATTERNS = [
    "README.md",
    "meta.json",
    "adapter_config.json",
    "adapter_model.safetensors",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "training_args.bin",
]

# Additional patterns required when uploading a full-fine-tuned model
# (peft_config=None path). Adapter-only runs continue to use HF_UPLOAD_ALLOW_PATTERNS.
HF_UPLOAD_ALLOW_PATTERNS_FULLFT = HF_UPLOAD_ALLOW_PATTERNS + [
    # Sharded weights: HF Transformers default shard threshold is 5GB. A 1.5B
    # bf16 fullFT (~3GB) fits unsharded, but a 3B+ fullFT writes
    # model-00001-of-N.safetensors and similar. The glob lookup in the
    # upload loop expands this pattern to every shard.
    "model-*.safetensors",
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
]


def _hf_base_model_id(model_name: str) -> str | None:
    hf_model_id = HF_BASE_MODEL_IDS.get(model_name)
    return hf_model_id


def _fix_hf_base_model_refs(out_dir: Path, model_name: str) -> None:
    hf_model_id = _hf_base_model_id(model_name)
    if not hf_model_id:
        return

    readme = out_dir / "README.md"
    if readme.exists():
        text = readme.read_text()
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            if line.startswith("base_model:"):
                lines[idx] = f"base_model: {hf_model_id}"
                readme.write_text("\n".join(lines) + "\n")
                break

    adapter_config = out_dir / "adapter_config.json"
    if adapter_config.exists():
        data = json.loads(adapter_config.read_text())
        data["base_model_name_or_path"] = hf_model_id
        adapter_config.write_text(json.dumps(data, indent=2) + "\n")


def _build_training_dataset(smoke: bool = False, idk_prompt: bool = False) -> Dataset:
    """MATH-L4/5 (~1800) + GSM8K top-token-length quartile (~1800).

    When idk_prompt=True the rollouts are formatted with PROMPT_TEMPLATE_IDK
    so the model is told explicitly that \\boxed{IDK} is an acceptable
    abstention. Used to probe whether the d1 IDK channel is reachable when
    seeded.
    """
    math_df = load_benchmark("math_train")
    if "level" in math_df.columns:
        math_df = math_df[math_df["level"].isin([4, 5])].reset_index(drop=True)

    gsm_df = load_benchmark("gsm8k_train")
    prompt_len = gsm_df["prompt"].str.len()
    gsm_df = gsm_df[prompt_len >= prompt_len.quantile(0.75)].reset_index(drop=True)

    cols = ["id", "prompt", "answer"]
    combined = pd.concat([math_df[cols], gsm_df[cols]], ignore_index=True)
    if smoke:
        combined = combined.head(50).reset_index(drop=True)

    return Dataset.from_dict({
        "prompt": [
            (PROMPT_TEMPLATE_IDK if idk_prompt else PROMPT_TEMPLATE).format(problem=p)
            for p in combined["prompt"]
        ],
        "answer": combined["answer"].astype(str).tolist(),
        "id": combined["id"].astype(str).tolist(),
    })


ADVANTAGE_NORM_MODES = (
    "z_score",
    "raw_rloo",
    "global_mean",
    "global_mean_std",
    # Zhu et al. 2025 PSR/NSR: per-sample fixed-constant advantage,
    # NO group normalization. zhu_psr pushes UP correct sequences
    # (advantage = +1 on correct, 0 elsewhere); zhu_nsr pushes DOWN wrong
    # sequences (advantage = -1 on wrong, 0 elsewhere). These modes IGNORE
    # loss_polarity (the polarity is baked into the advantage). They sidestep
    # the GRPO group-relative machinery that zeros within-polarity advantages
    # on fixed-per-outcome rewards (Option A polarity null observed
    # 2026-05-31 NSR S0 run; see log.md).
    "zhu_psr",
    "zhu_nsr",
    # zhu_signed: passes the raw signed reward (+alpha/-beta/+abstain) as the
    # advantage, with NO baseline subtraction and NO std normalization. Unlike
    # zhu_psr/zhu_nsr which overwrite the advantage with a fixed unit polarity,
    # zhu_signed preserves the FULL magnitude of each per-sample reward. This
    # means the gradient update is proportional to the reward magnitude (e.g.
    # a beta=2 wrong push is twice as strong as a beta=1 push). Useful for
    # studying whether unnormalized asymmetric rewards produce behavioral
    # effects faster than z-score normalized ones, at the cost of no within-
    # group variance control. Collapse guard: per-batch abstention-rate and an
    # outcome-entropy proxy are logged every step; training raises/early-aborts
    # only when abstention stays > 0.97 across a rolling 10-step window (true
    # all-silence collapse), without zeroing individual high-abstention batches
    # (see STD_FLOOR_REPLACEMENT zhu_signed branch).
    "zhu_signed",
)
LOSS_POLARITY_MODES = ("both", "positive_only", "negative_only")


def _zhu_signed_collapse_guard(trainer, outcomes: list) -> None:
    """Collapse guard for the zhu_signed advantage normalisation mode.

    Appends per-batch abstention-rate and outcome-entropy-proxy metrics to
    trainer._metrics, maintains a rolling 10-step abstention history on
    trainer._zhu_signed_abstain_hist, and raises RuntimeError when abstention
    stays above 0.97 across ALL of the last 10 steps (true all-silence collapse).

    Design intent: a SINGLE high-abstention batch (e.g. S4, ~97% silent + a
    few wrong) still carries valid wrong-channel gradient and must not be
    zeroed. We abort only on SUSTAINED collapse to save compute in the true
    all-silence basin.

    Args:
        trainer: the GRPOTrainer instance (needs _metrics dict-like and the
            _zhu_signed_abstain_hist list attribute).
        outcomes: per-sample outcome label list from _LAST_BATCH_TELEMETRY.
    """
    import math as _math

    n_zs = len(outcomes)
    if n_zs > 0:
        n_silent_zs = sum(1 for o in outcomes if o == 'silent')
        abstain_rate_zs = n_silent_zs / n_zs
        unique_outcomes_zs = len(set(outcomes))
        entropy_proxy_zs = _math.log(unique_outcomes_zs) if unique_outcomes_zs > 1 else 0.0
    else:
        abstain_rate_zs = 0.0
        entropy_proxy_zs = 0.0

    trainer._metrics.setdefault('zhu_signed_abstain_rate', []).append(abstain_rate_zs)
    trainer._metrics.setdefault('zhu_signed_outcome_entropy_proxy', []).append(entropy_proxy_zs)

    zs_hist = getattr(trainer, '_zhu_signed_abstain_hist', None)
    if zs_hist is None:
        zs_hist = []
        trainer._zhu_signed_abstain_hist = zs_hist
    zs_hist.append(abstain_rate_zs)
    if len(zs_hist) > 10:
        zs_hist.pop(0)
    if len(zs_hist) >= 10 and all(r > 0.97 for r in zs_hist):
        trainer._metrics.setdefault('zhu_signed_collapse_abort', []).append(1.0)
        raise RuntimeError(
            "zhu_signed collapse: abstention > 0.97 sustained over the last 10 "
            "steps; aborting to avoid wasted compute in the all-silence basin."
        )


def _install_prepare_inputs_guards(
    base_cls,
    std_floor: float,
    advantage_norm: str = "z_score",
    use_cwar: bool = False,
    cwar_attempts_only: bool = False,
    wrong_rescale: str = "cw",
    lambda_oc: float = 0.0,
    tau_oc: float = 0.5,
    gamma_cmf: float = 0.0,
    grf_delta: float | None = None,
    sc_delta: float | None = None,
    num_generations: int = 1,
    loss_polarity: str = "both",
    global_ema_momentum: float = 0.99,
):
    """Return a GRPOTrainer subclass with guarded TRL 0.15 `_prepare_inputs`.

    Patches applied to TRL 0.15.2 `_prepare_inputs` via source-rewrite:
    - empty-completion guard (avoids argmax crash on zero-width is_eos)
    - std-floor + advantage-norm patch (S4 alpha=0 degenerate case zeros
      advantages when intra-group std < std_floor; advantage_norm switches
      between z_score and raw_rloo formulas).
    - Rescale block (installed when use_cwar=True OR grf_delta is not None):
        * extra policy forward pass to capture per-token policy log-probs
          (only runs when _asym_use_cwar=True at trainer-instance time).
        * post-reward rescale, dispatched by wrong_rescale kind:
            "cw"  : r_i <- sign(r_i) * |r_i| * conf_i (CW / FCW / CWD1)
            "ocp" : r_i <- r_i - lambda_oc * max(0, conf_i - tau_oc),
                    applied only to wrong samples (Overconfidence Penalty)
            "cmf" : r_i <- r_i * (1 + gamma_cmf * conf_i),
                    applied only to wrong samples (Confidence-Modulated Fear)
        * GRF block (runs when grf_delta is not None, independent of conf):
            For each rollout group of `num_generations`, computes per-sample
            peer_correct_rate = (#correct - is_self_correct) / (G - 1),
            then r_i <- r_i * (1 + grf_delta * peer_rate_i) on wrong samples.

      All variants feed into whichever advantage_norm is selected.

    advantage_norm:
        - "z_score" (default): (rewards - mean) / (std + 1e-4), TRL's stock behavior.
        - "raw_rloo": rewards - mean (mean-baseline subtraction without std-divide).
          Closer to Zhu et al. 2025 W-REINFORCE / standard RLOO; isolates whether
          GRPO's within-group z-normalization is the gap driver vs raw weighted
          advantages. Std-floor guard still zeros advantages on uniform-reward
          groups for stability.

    New (2026-05-28) extra params:
        wrong_rescale : one of {"cw", "ocp", "cmf"}; default "cw" for back-compat.
            Only consulted when use_cwar=True.
        lambda_oc, tau_oc : OCP hyperparams (ignored unless wrong_rescale="ocp").
        gamma_cmf : CMF hyperparam (ignored unless wrong_rescale="cmf").
        grf_delta : float or None. When not None, triggers GRF rescale.
        num_generations : group size G, used by the GRF block.
    """
    if advantage_norm not in ADVANTAGE_NORM_MODES:
        raise ValueError(
            f"advantage_norm must be one of {ADVANTAGE_NORM_MODES}, got {advantage_norm!r}"
        )
    if loss_polarity not in LOSS_POLARITY_MODES:
        raise ValueError(
            f"loss_polarity must be one of {LOSS_POLARITY_MODES}, got {loss_polarity!r}"
        )

    original = getattr(base_cls, "_prepare_inputs", None)
    if original is None:
        raise RuntimeError(f"{base_cls.__name__} has no _prepare_inputs method to guard")

    EMPTY_COMP_TARGET = "    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]"
    EMPTY_COMP_REPLACEMENT = """\
    if is_eos.size(1) > 0:
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]"""

    STD_FLOOR_TARGET = "    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)"
    # Unified replacement: dispatches at runtime on self._asym_advantage_norm
    # and self._asym_loss_polarity. Handles four advantage_norm modes
    # (z_score, raw_rloo, global_mean, global_mean_std) crossed with three
    # loss_polarity modes (both, positive_only, negative_only).
    #
    # Polarity mask (Option A / Zhu-strict): masked-out samples are EXCLUDED
    # from both the baseline statistics (per-group masked mean/std) AND the
    # gradient (advantage forced to 0). Mask source: outcomes telemetry
    # ("correct"/"wrong") if its length matches rewards, else fall back to
    # sign(rewards). When polarity != "both" and all surviving samples in a
    # group share the same reward, std=0 and the std_floor guard zeros the
    # advantage - an honest null when the within-polarity reward signal has
    # zero variance.
    #
    # Global modes: maintain a single running EMA (mean, std) on the trainer
    # instance, updated each step from the realized rewards (post-rescale).
    # Polarity mask, when non-default, scopes the EMA update too (only
    # polarity-allowed samples feed the running stats).
    STD_FLOOR_REPLACEMENT = """\
    from asym_rewards.reward import _LAST_BATCH_TELEMETRY, OUTCOME_CORRECT, OUTCOME_WRONG
    std_floor = torch.as_tensor(
        self._asym_std_floor,
        dtype=rewards.dtype,
        device=rewards.device,
    )

    polarity = getattr(self, '_asym_loss_polarity', 'both')
    adv_norm = getattr(self, '_asym_advantage_norm', 'z_score')
    outcomes_pol = _LAST_BATCH_TELEMETRY.get('outcomes', [])
    if polarity == 'positive_only':
        if len(outcomes_pol) == rewards.numel():
            polarity_mask = torch.tensor(
                [o == OUTCOME_CORRECT for o in outcomes_pol],
                device=rewards.device, dtype=torch.bool,
            )
        else:
            polarity_mask = rewards > 0
    elif polarity == 'negative_only':
        if len(outcomes_pol) == rewards.numel():
            polarity_mask = torch.tensor(
                [o == OUTCOME_WRONG for o in outcomes_pol],
                device=rewards.device, dtype=torch.bool,
            )
        else:
            polarity_mask = rewards < 0
    else:
        polarity_mask = torch.ones_like(rewards, dtype=torch.bool)

    G_pol = int(getattr(self, '_asym_num_generations', rewards.numel()))
    if polarity != 'both' and G_pol > 1 and rewards.numel() % G_pol == 0:
        B_pol = rewards.numel() // G_pol
        grouped_rewards = rewards.view(B_pol, G_pol)
        grouped_mask = polarity_mask.view(B_pol, G_pol).to(rewards.dtype)
        mask_count = grouped_mask.sum(dim=1)
        # Telemetry: effective per-group sample count, empty-group count
        self._metrics.setdefault('loss_polarity_effective_count', []).append(
            mask_count.float().mean().item()
        )
        self._metrics.setdefault('loss_polarity_empty_groups', []).append(
            float((mask_count == 0).sum().item())
        )
        safe_count = mask_count.clamp_min(1.0)
        masked_sum = (grouped_rewards * grouped_mask).sum(dim=1)
        masked_mean_g = masked_sum / safe_count
        diff = (grouped_rewards - masked_mean_g.unsqueeze(1)) * grouped_mask
        var_denom = (safe_count - 1.0).clamp_min(1.0)
        masked_var_g = (diff * diff).sum(dim=1) / var_denom
        masked_std_g = masked_var_g.sqrt()
        # Empty groups: pin to zero so std_floor guard zeros the advantage
        empty_g = mask_count == 0
        masked_mean_g = torch.where(empty_g, torch.zeros_like(masked_mean_g), masked_mean_g)
        masked_std_g = torch.where(empty_g, torch.zeros_like(masked_std_g), masked_std_g)
        mean_grouped_rewards = masked_mean_g.repeat_interleave(G_pol)
        std_grouped_rewards = masked_std_g.repeat_interleave(G_pol)

    if adv_norm in ('global_mean', 'global_mean_std'):
        # Bootstrap / update the running EMA from polarity-allowed rewards only.
        # Skip the EMA update when the polarity mask is empty so excluded
        # samples never contaminate the baseline. If the FIRST batch has an
        # empty mask, we MUST also defer the bootstrap (codex P2 2026-05-31
        # round 2) - writing a placeholder 0.0 would bias all subsequent EMA
        # updates toward 0 via the 0.99/0.01 mixing. On a deferred-bootstrap
        # step the advantage is forced to zero (no gradient signal) so the
        # step is effectively a no-op.
        skip_ema_update = bool(polarity != 'both' and not polarity_mask.any())
        momentum = float(getattr(self, '_asym_global_ema_momentum', 0.99))
        prev_mean = getattr(self, '_asym_global_mean', None)
        if not skip_ema_update:
            if polarity_mask.any():
                denom_pol = polarity_mask.float().sum().clamp_min(1)
                batch_mean_val = (rewards * polarity_mask.float()).sum() / denom_pol
                if denom_pol > 1:
                    batch_var_val = (
                        ((rewards - batch_mean_val) * polarity_mask.float()).pow(2).sum()
                        / (denom_pol - 1).clamp_min(1)
                    )
                    batch_std_val = batch_var_val.sqrt()
                else:
                    batch_std_val = torch.zeros_like(batch_mean_val)
            else:
                batch_mean_val = rewards.mean()
                batch_std_val = (
                    rewards.std() if rewards.numel() > 1 else torch.zeros_like(batch_mean_val)
                )
            if prev_mean is None:
                self._asym_global_mean = float(batch_mean_val.item())
            else:
                self._asym_global_mean = (
                    momentum * prev_mean + (1.0 - momentum) * float(batch_mean_val.item())
                )
            if adv_norm == 'global_mean_std':
                prev_std = getattr(self, '_asym_global_std', None)
                if prev_std is None:
                    self._asym_global_std = float(batch_std_val.item())
                else:
                    self._asym_global_std = (
                        momentum * prev_std + (1.0 - momentum) * float(batch_std_val.item())
                    )
        else:
            self._metrics.setdefault('global_ema_skipped_steps', []).append(1.0)

        cur_mean = getattr(self, '_asym_global_mean', None)
        if cur_mean is None:
            # Deferred bootstrap: empty-mask first batch. No gradient signal.
            advantages = torch.zeros_like(rewards)
            self._metrics.setdefault('global_ema_deferred_bootstrap', []).append(1.0)
        else:
            global_mean_t = torch.full_like(rewards, cur_mean)
            if adv_norm == 'global_mean':
                advantages = rewards - global_mean_t
            else:
                cur_std = getattr(self, '_asym_global_std', None)
                std_use = max(float(cur_std), float(self._asym_std_floor))
                global_std_t = torch.full_like(rewards, std_use)
                advantages = (rewards - global_mean_t) / (global_std_t + 1e-4)
            self._metrics.setdefault('global_mean_running', []).append(self._asym_global_mean)
            if adv_norm == 'global_mean_std':
                self._metrics.setdefault('global_std_running', []).append(self._asym_global_std)
    elif adv_norm in ('zhu_psr', 'zhu_nsr'):
        # Zhu-style fixed-constant advantage on the matching polarity, zero
        # elsewhere. No group normalization, no baseline subtraction.
        # Implemented from outcomes telemetry; falls back to sign of rewards
        # when telemetry is unavailable (unit-test safety net).
        if adv_norm == 'zhu_psr':
            if len(outcomes_pol) == rewards.numel():
                zhu_mask = torch.tensor(
                    [o == OUTCOME_CORRECT for o in outcomes_pol],
                    device=rewards.device, dtype=torch.bool,
                )
            else:
                zhu_mask = rewards > 0
            advantages = zhu_mask.to(rewards.dtype)
        else:
            if len(outcomes_pol) == rewards.numel():
                zhu_mask = torch.tensor(
                    [o == OUTCOME_WRONG for o in outcomes_pol],
                    device=rewards.device, dtype=torch.bool,
                )
            else:
                zhu_mask = rewards < 0
            advantages = -zhu_mask.to(rewards.dtype)
        # Telemetry: per-batch nonzero advantage count.
        self._metrics.setdefault('zhu_nonzero_advantage_count', []).append(
            float(zhu_mask.sum().item())
        )
    elif adv_norm == 'zhu_signed':
        # zhu_signed: pass raw signed rewards as advantages; no baseline
        # subtraction, no std normalization. Preserves full magnitude so
        # +alpha/-beta/+abstain scale differences drive gradient proportionally.
        # Unlike zhu_psr/zhu_nsr (unit polarity), the magnitude is live.
        advantages = rewards.clone()
        # Telemetry: mean signed advantage and count of non-zero samples.
        self._metrics.setdefault('zhu_signed_advantage_mean', []).append(
            float(rewards.mean().item())
        )
        self._metrics.setdefault('zhu_signed_nonzero_count', []).append(
            float((rewards != 0).sum().item())
        )
        # Collapse guard delegated to the module-level helper (unit-testable).
        # _LAST_BATCH_TELEMETRY is already in exec scope (used elsewhere in this
        # block). _zhu_signed_collapse_guard is injected via exec_globals below.
        _zhu_signed_collapse_guard(self, _LAST_BATCH_TELEMETRY.get('outcomes', []))
    elif adv_norm == 'z_score':
        low_std = std_grouped_rewards < std_floor
        safe_std = torch.where(low_std, torch.ones_like(std_grouped_rewards), std_grouped_rewards)
        advantages = torch.where(
            low_std,
            torch.zeros_like(rewards),
            (rewards - mean_grouped_rewards) / (safe_std + 1e-4),
        )
    else:
        low_std = std_grouped_rewards < std_floor
        advantages = torch.where(
            low_std,
            torch.zeros_like(rewards),
            rewards - mean_grouped_rewards,
        )

    if polarity != 'both' and adv_norm not in ('zhu_psr', 'zhu_nsr', 'zhu_signed'):
        advantages = torch.where(polarity_mask, advantages, torch.zeros_like(advantages))"""

    # CWAR: capture per-token policy log-probs after the ref-logps block.
    # We re-call _get_per_token_logps on self.model (adapter ENABLED, i.e. policy)
    # since TRL 0.15.2 only computes them in compute_loss, too late for reward
    # rescaling. Costs one extra forward pass per training step (~0.5 s at 0.5B).
    CW_LOGPS_TARGET = "    # Decode the generated completions"
    CW_LOGPS_REPLACEMENT = """\
    if getattr(self, '_asym_use_cwar', False):
        with torch.inference_mode():
            policy_per_token_logps_cw = self._get_per_token_logps(
                self.model, prompt_completion_ids, attention_mask, logits_to_keep
            )
    else:
        policy_per_token_logps_cw = None

    # Decode the generated completions"""

    # CWAR / OCP / CMF / GRF rescale: rewrite rewards in-place before group
    # statistics are computed. Dispatches on `self._asym_wrong_rescale`:
    #   - "cw"  : symmetric multiplicative scaling by per-sample conf (CW/FCW)
    #   - "ocp" : piecewise additive penalty on wrong only (Overconfidence Penalty)
    #   - "cmf" : multiplicative scaling on wrong only (Confidence-Modulated Fear)
    # The GRF block runs independently after this and uses group composition
    # (not conf) to scale wrong samples by peer_correct_rate.
    # conf_i = exp(mean over completion tokens of policy log-prob), clamped [0,1].
    # rewards is already gathered (B*G across processes); conf must be gathered too.
    CW_RESCALE_TARGET = "    # Compute grouped-wise rewards"
    CW_RESCALE_REPLACEMENT = """\
    if policy_per_token_logps_cw is not None:
        mask_cw = completion_mask.float()
        token_counts_cw = mask_cw.sum(dim=1).clamp_min(1.0)
        mean_logp_cw = (policy_per_token_logps_cw * mask_cw).sum(dim=1) / token_counts_cw
        conf_cw_local = torch.exp(mean_logp_cw).clamp(0.0, 1.0)
        conf_cw = gather(conf_cw_local).to(rewards.dtype)
        if getattr(self, '_asym_cwar_attempts_only', False):
            # CW-D1 path: scale only outcome in {correct, wrong}; leave IDK
            # channels (wise/wasteful) and silent unscaled. Telemetry order
            # matches the local rewards tensor on single-GPU; we fall back to
            # the legacy nonzero mask when the lengths disagree (multi-GPU
            # gather or unexpected rollout pruning).
            from asym_rewards.reward import (
                _LAST_BATCH_TELEMETRY,
                OUTCOME_CORRECT,
                OUTCOME_WRONG,
            )
            outcomes_cw = _LAST_BATCH_TELEMETRY.get('outcomes', [])
            if len(outcomes_cw) == rewards.numel():
                attempt_mask_cw = torch.tensor(
                    [o == OUTCOME_CORRECT or o == OUTCOME_WRONG for o in outcomes_cw],
                    device=rewards.device, dtype=torch.bool,
                )
            else:
                attempt_mask_cw = rewards != 0
        else:
            attempt_mask_cw = rewards != 0
        wrong_mask_cw = attempt_mask_cw & (rewards < 0)
        kind_cw = getattr(self, '_asym_wrong_rescale', 'cw')
        if kind_cw == 'cw':
            sign_cw = torch.sign(rewards)
            rewards = torch.where(
                attempt_mask_cw,
                sign_cw * torch.abs(rewards) * conf_cw,
                rewards,
            )
        elif kind_cw == 'ocp':
            lambda_oc = float(getattr(self, '_asym_lambda_oc', 0.0))
            tau_oc = float(getattr(self, '_asym_tau_oc', 0.5))
            extra_pen = torch.clamp(conf_cw - tau_oc, min=0.0) * lambda_oc
            rewards = torch.where(wrong_mask_cw, rewards - extra_pen, rewards)
            if wrong_mask_cw.any():
                n_conf_wrong = (wrong_mask_cw & (conf_cw > tau_oc)).sum().item()
                self._metrics.setdefault('ocp_n_confident_wrong', []).append(float(n_conf_wrong))
                self._metrics.setdefault('ocp_extra_pen_mean', []).append(extra_pen[wrong_mask_cw].mean().item())
        elif kind_cw == 'cmf':
            gamma_cmf = float(getattr(self, '_asym_gamma_cmf', 0.0))
            multiplier = 1.0 + gamma_cmf * conf_cw
            rewards = torch.where(wrong_mask_cw, rewards * multiplier, rewards)
            if wrong_mask_cw.any():
                self._metrics.setdefault('cmf_conf_wrong_mean', []).append(conf_cw[wrong_mask_cw].mean().item())
        else:
            raise ValueError(f"Unknown wrong_rescale kind: {kind_cw!r}")
        self._metrics.setdefault("conf_cw", []).append(conf_cw.mean().item())

    grf_delta_local = getattr(self, '_asym_grf_delta', None)
    if grf_delta_local is not None:
        G = int(getattr(self, '_asym_num_generations', 1))
        if G > 1 and rewards.numel() % G == 0:
            B = rewards.numel() // G
            grouped = rewards.view(B, G)
            is_correct_g = (grouped > 0).to(grouped.dtype)
            correct_count = is_correct_g.sum(dim=1, keepdim=True)
            peer_rate = (correct_count - is_correct_g) / float(G - 1)
            wrong_mask_g = grouped < 0
            multiplier_g = 1.0 + float(grf_delta_local) * peer_rate
            grouped = torch.where(wrong_mask_g, grouped * multiplier_g, grouped)
            rewards = grouped.view(-1)
            if wrong_mask_g.any():
                self._metrics.setdefault('grf_peer_rate_on_wrong', []).append(
                    peer_rate[wrong_mask_g].mean().item()
                )

    sc_delta_local = getattr(self, '_asym_sc_delta', None)
    if sc_delta_local is not None:
        from asym_rewards.reward import _LAST_BATCH_TELEMETRY as _SC_TEL
        G_sc = int(getattr(self, '_asym_num_generations', 1))
        extracted_sc = _SC_TEL.get('extracted', [])
        if G_sc > 1 and rewards.numel() % G_sc == 0 and len(extracted_sc) == rewards.numel():
            B_sc = rewards.numel() // G_sc
            loo_agree_list = []
            for b in range(B_sc):
                group_ext = extracted_sc[b * G_sc:(b + 1) * G_sc]
                for i, ext_i in enumerate(group_ext):
                    if not ext_i:
                        loo_agree_list.append(0.0)
                    else:
                        cnt = 0
                        for j, ext_j in enumerate(group_ext):
                            if j != i and ext_j == ext_i:
                                cnt += 1
                        loo_agree_list.append(float(cnt) / max(G_sc - 1, 1))
            loo_agree_t = torch.tensor(loo_agree_list, device=rewards.device, dtype=rewards.dtype)
            wrong_mask_sc = rewards < 0
            multiplier_sc = 1.0 + float(sc_delta_local) * loo_agree_t
            rewards = torch.where(wrong_mask_sc, rewards * multiplier_sc, rewards)
            if wrong_mask_sc.any():
                self._metrics.setdefault('sc_loo_agree_on_wrong', []).append(
                    loo_agree_t[wrong_mask_sc].mean().item()
                )

    # Compute grouped-wise rewards"""

    source = textwrap.dedent(inspect.getsource(original))
    if EMPTY_COMP_TARGET in source:
        source = source.replace(EMPTY_COMP_TARGET, EMPTY_COMP_REPLACEMENT)
    else:
        print("WARNING: empty-completion guard pattern not found in _prepare_inputs; skipping", flush=True)

    if STD_FLOOR_TARGET in source:
        source = source.replace(STD_FLOOR_TARGET, STD_FLOOR_REPLACEMENT)
    else:
        print("WARNING: std-floor guard pattern not found in _prepare_inputs; skipping", flush=True)

    needs_rescale_block = use_cwar or (grf_delta is not None) or (sc_delta is not None)
    if needs_rescale_block:
        # The LOGPS patch defines `policy_per_token_logps_cw` (possibly None
        # when _asym_use_cwar=False, e.g. GRF-only). The rescale block needs
        # the name to exist, so install both whenever either path fires.
        if CW_LOGPS_TARGET in source:
            source = source.replace(CW_LOGPS_TARGET, CW_LOGPS_REPLACEMENT, 1)
        else:
            raise RuntimeError("rescale patch failed: CW_LOGPS_TARGET not found in _prepare_inputs source")
        if CW_RESCALE_TARGET in source:
            source = source.replace(CW_RESCALE_TARGET, CW_RESCALE_REPLACEMENT, 1)
        else:
            raise RuntimeError("rescale patch failed: CW_RESCALE_TARGET not found in _prepare_inputs source")

    source = source.replace("super()._prepare_inputs", "super(_AsymBaseGRPOTrainer, self)._prepare_inputs")
    namespace: dict[str, Any] = {}
    exec_globals = dict(original.__globals__)
    exec_globals["_AsymBaseGRPOTrainer"] = base_cls
    exec_globals["_zhu_signed_collapse_guard"] = _zhu_signed_collapse_guard
    exec(compile(source, inspect.getsourcefile(original) or "<asym-guarded-grpo-prep>", "exec"), exec_globals, namespace)
    base_cls._prepare_inputs = namespace["_prepare_inputs"]

    class StdFloorGRPOTrainer(base_cls):
        pass

    norm_tag = {
        "z_score": "StdFloor",
        "raw_rloo": "RawRLOO",
        "global_mean": "GlobalMean",
        "global_mean_std": "GlobalMeanStd",
        "zhu_psr": "ZhuPSR",
        "zhu_nsr": "ZhuNSR",
        "zhu_signed": "ZhuSigned",
    }[advantage_norm]
    StdFloorGRPOTrainer.__name__ = f"{norm_tag}{base_cls.__name__}"
    StdFloorGRPOTrainer.__qualname__ = StdFloorGRPOTrainer.__name__
    StdFloorGRPOTrainer._asym_std_floor = float(std_floor)
    StdFloorGRPOTrainer._asym_advantage_norm = advantage_norm
    StdFloorGRPOTrainer._asym_use_cwar = bool(use_cwar)
    StdFloorGRPOTrainer._asym_cwar_attempts_only = bool(cwar_attempts_only)
    StdFloorGRPOTrainer._asym_wrong_rescale = str(wrong_rescale)
    StdFloorGRPOTrainer._asym_lambda_oc = float(lambda_oc)
    StdFloorGRPOTrainer._asym_tau_oc = float(tau_oc)
    StdFloorGRPOTrainer._asym_gamma_cmf = float(gamma_cmf)
    StdFloorGRPOTrainer._asym_grf_delta = (
        float(grf_delta) if grf_delta is not None else None
    )
    StdFloorGRPOTrainer._asym_sc_delta = (
        float(sc_delta) if sc_delta is not None else None
    )
    StdFloorGRPOTrainer._asym_num_generations = int(num_generations)
    StdFloorGRPOTrainer._asym_loss_polarity = str(loss_polarity)
    StdFloorGRPOTrainer._asym_global_ema_momentum = float(global_ema_momentum)
    StdFloorGRPOTrainer._asym_global_mean = None
    StdFloorGRPOTrainer._asym_global_std = None
    return StdFloorGRPOTrainer


def _make_forced_decode_fn(model, tokenizer, max_new_tokens: int = 32):
    """Build the D1 forced-decode counterfactual helper.

    Given (prompt, completion) for an IDK sample, runs greedy generation with
    `prompt + completion + "\\boxed{"` as the prefix and `max_new_tokens`
    cap, then extracts everything up to the first closing brace as the
    forced answer. Used by reward_d1 to dispatch wise_idk vs wasteful_idk.

    Uses HF transformers `generate()` on the live training model (no separate
    vLLM engine). Wall-time on Qwen-1.5B is ~80-120 ms per call; with G=8
    and an ~20% IDK rate that is roughly 1.5 calls per step, well under the
    <10% step-time overhead promise.
    """
    import torch

    SENTINEL = "\nReconsider: \\boxed{"
    sentinel_ids = tokenizer(
        SENTINEL, return_tensors="pt", add_special_tokens=False
    )["input_ids"]
    n_sentinel = sentinel_ids.shape[1]
    BUDGET = 2048 - n_sentinel

    def forced_decode_fn(prompt: str, completion: str) -> str:
        prefix_raw = (prompt or "") + (completion or "")
        # Tokenize the raw prefix without truncation, then keep the TAIL via
        # manual slicing so the recent reasoning and the model's own
        # \boxed{IDK} survive. Right-truncation would drop the completion
        # tail (where IDK lives) and leave the counterfactual to label a
        # stale, IDK-free prefix as wise_idk vs wasteful_idk.
        all_ids = tokenizer(
            prefix_raw, return_tensors="pt", add_special_tokens=False,
        )["input_ids"]
        if all_ids.shape[1] > BUDGET:
            all_ids = all_ids[:, -BUDGET:]
        input_ids = torch.cat([all_ids, sentinel_ids], dim=1).to(model.device)
        attn = torch.ones_like(input_ids)
        # Forced decode must be deterministic: disable LoRA dropout by
        # toggling to eval mode around generate(); torch.inference_mode alone
        # does not suppress dropout.
        was_training = model.training
        if was_training:
            model.eval()
        try:
            with torch.inference_mode():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
                )
        finally:
            if was_training:
                model.train()
        gen_ids = out[0, input_ids.shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        if "}" in text:
            text = text.split("}", 1)[0]
        return text.strip()

    return forced_decode_fn


def _make_telemetry_callback(family: str, weighting: str | None):
    """Build a TrainerCallback that logs per-step reward telemetry to wandb.

    Reads `asym_rewards.reward._LAST_BATCH_TELEMETRY` after each step and
    publishes outcome-class frequencies, per-channel reward means,
    forced-decode timing (D1 family), and confidence summary (CWAR weighting).
    Cheap; runs each step. The expensive group-level diagnostics
    (abstain-advantage gauge, group reward variance, valid_group_frac)
    are tracked separately in `_install_prepare_inputs_guards` and surface
    through `self._metrics` per TRL's standard logging path.
    """
    from transformers.trainer_callback import TrainerCallback

    class DetailedTelemetryCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            try:
                import wandb
            except ImportError:
                return
            if wandb.run is None:
                return
            from .reward import _LAST_BATCH_TELEMETRY, OUTCOME_CLASSES
            outcomes = _LAST_BATCH_TELEMETRY.get("outcomes", [])
            n = len(outcomes)
            if n == 0:
                return
            log: dict[str, float] = {}
            # Outcome-class frequencies. Always log all 5 classes so plots
            # have a stable axis (D1 family); for asym family only 3 classes
            # ever fire (correct / wrong / silent) but the others log as 0.
            for cls in OUTCOME_CLASSES:
                log[f"outcome/{cls}_frac"] = sum(1 for o in outcomes if o == cls) / n
            # Per-channel reward decomposition (mean per class, NaN-safe).
            channel_rewards = _LAST_BATCH_TELEMETRY.get("channel_rewards", [])
            if channel_rewards:
                for cls in OUTCOME_CLASSES:
                    bucket = [r for r, o in zip(channel_rewards, outcomes) if o == cls]
                    if bucket:
                        log[f"reward/{cls}_mean"] = sum(bucket) / len(bucket)
            # D1-family forced-decode timing + IDK rate.
            if family == "d1":
                fd_s = _LAST_BATCH_TELEMETRY.get("forced_decode_s", [])
                idk_flags = _LAST_BATCH_TELEMETRY.get("idk_flags", [])
                n_idk = int(sum(1 for f in idk_flags if f))
                log["d1/idk_rate"] = n_idk / n
                log["d1/forced_decode_s"] = float(sum(fd_s))
                # Under FD1xsil silent samples ALSO trigger forced decode;
                # count them in the divisor so the per-sample timing is
                # accurate. Falls back to n_idk for non-extended conditions.
                n_forced = sum(
                    1 for o in outcomes
                    if o in ("wise_idk", "wasteful_idk",
                             "silent_wise", "silent_wasteful")
                )
                if n_forced > 0:
                    log["d1/forced_decode_per_sample_s"] = float(sum(fd_s) / n_forced)
                # Always-IDK collapse detector: fraction of (idk + silent)
                # samples per batch; a sustained >=0.875 (= 7/8 of G=8) over
                # 50 steps trips the wandb alert downstream.
                n_idk_or_silent = sum(
                    1 for o in outcomes
                    if o in ("wise_idk", "wasteful_idk", "silent",
                             "silent_wise", "silent_wasteful")
                )
                log["collapse/idk_or_silent_frac"] = n_idk_or_silent / n
            wandb.log(log, step=state.global_step)

    return DetailedTelemetryCallback()


def train_grpo(
    model_name: str,
    condition_id: str,
    training_config: dict[str, Any],
    out_dir: Path,
    smoke: bool = False,
    seed: int | None = None,
    advantage_norm: str = "z_score",
    full_finetune: bool = False,
    loss_polarity: str = "both",
    lr_override: float | None = None,
    max_grad_norm: float | None = None,
    idk_prompt: bool = False,
    resume_from_checkpoint: str | None = None,
    warm_start_adapter: str | None = None,
) -> dict[str, Any]:
    """Train a single (model, condition, seed) combo and save the LoRA adapter.

    If `seed` is None, falls back to training_config['common']['seed'] (default 42).
    The seed is propagated to GRPOConfig.seed, the run_name suffix, and the HF
    repo template so multi-seed runs land in separate output dirs and wandb runs.

    advantage_norm selects the per-group advantage scheme: "z_score" matches
    TRL's stock GRPO; "raw_rloo" subtracts the group mean without dividing by
    std (Zhu W-REINFORCE / RLOO-style). Non-default modes append a suffix to
    `run_tag` so artifacts do not collide with z-score runs at the same
    (model, condition, seed).

    `warm_start_adapter` vs `resume_from_checkpoint` are distinct, mutually
    exclusive operations: `resume_from_checkpoint` continues an interrupted GRPO
    run from a full TRL checkpoint (restores optimizer/scheduler/step count and
    requires trainer_state.json); `warm_start_adapter` starts a FRESH GRPO run
    with the LoRA initialized from a pretrained adapter directory (e.g. the P6
    SFT IDK warm-start), training the SAME adapter from step 0 with a fresh
    optimizer. A bare adapter dir has no trainer_state.json, so it must use
    `warm_start_adapter`, never `resume_from_checkpoint`.

    Returns a meta dict with run stats (steps, final reward, wandb url if any).
    """
    if condition_id not in CONDITIONS:
        raise KeyError(f"Unknown condition '{condition_id}'")
    if advantage_norm not in ADVANTAGE_NORM_MODES:
        raise ValueError(
            f"advantage_norm must be one of {ADVANTAGE_NORM_MODES}, got {advantage_norm!r}"
        )
    if loss_polarity not in LOSS_POLARITY_MODES:
        raise ValueError(
            f"loss_polarity must be one of {LOSS_POLARITY_MODES}, got {loss_polarity!r}"
        )
    if advantage_norm in ("zhu_psr", "zhu_nsr", "zhu_signed") and loss_polarity != "both":
        print(
            f"WARN: loss_polarity={loss_polarity!r} is ignored by advantage_norm={advantage_norm!r}; "
            f"canonicalizing to 'both' so run_tag, HF repo, and meta.json reflect the actual training.",
            flush=True,
        )
        loss_polarity = "both"

    if warm_start_adapter is not None:
        if resume_from_checkpoint is not None:
            raise ValueError(
                "warm_start_adapter and resume_from_checkpoint are mutually exclusive: the "
                "former initializes a fresh GRPO run from a pretrained adapter, the latter "
                "continues a TRL checkpoint. Pass only one."
            )
        if full_finetune:
            raise ValueError("warm_start_adapter is LoRA-only and cannot combine with full_finetune.")
        _ac_path = Path(warm_start_adapter) / "adapter_config.json"
        if not _ac_path.exists():
            raise FileNotFoundError(
                f"warm_start_adapter={warm_start_adapter!r} has no adapter_config.json -- it is not a "
                "LoRA adapter directory. (A TRL checkpoint goes via resume_from_checkpoint instead.)"
            )
        # Geometry guard: the warm-start adapter must share the sweep's LoRA shape
        # (r / alpha / target_modules) or the resulting adapter is not comparable to
        # the rest of the conditions. Existence alone is not enough -- a stray adapter
        # from another sweep would load fine and silently train at the wrong rank.
        _ac = json.loads(_ac_path.read_text())
        _cfg_lora = training_config["lora"]
        _mismatch = []
        if int(_ac.get("r", -1)) != int(_cfg_lora["r"]):
            _mismatch.append(f"r={_ac.get('r')} != {_cfg_lora['r']}")
        if int(_ac.get("lora_alpha", -1)) != int(_cfg_lora["alpha"]):
            _mismatch.append(f"lora_alpha={_ac.get('lora_alpha')} != {_cfg_lora['alpha']}")
        if set(_ac.get("target_modules") or []) != set(_cfg_lora["target_modules"]):
            _mismatch.append(
                f"target_modules={sorted(_ac.get('target_modules') or [])} != "
                f"{sorted(_cfg_lora['target_modules'])}"
            )
        if _mismatch:
            raise ValueError(
                f"warm_start_adapter LoRA geometry does not match configs/training.yaml "
                f"({'; '.join(_mismatch)}). The warm-start adapter must use the same LoRA "
                "shape as the rest of the sweep for comparability."
            )
        # Base-model guard: same LoRA geometry does NOT imply the same base. An
        # adapter trained on a different same-shape base (e.g. qwen2_5_math_1_5b vs
        # qwen2_5_1_5b) would load its deltas onto the wrong base and the later
        # adapter_config rewrite would mislabel it as model_name. Reject unless the
        # adapter's recorded base matches model_name in any of its forms (the local
        # MODEL_DIRS leaf is model_name; the HF id and its leaf are also accepted).
        _accepted_base = {model_name}
        _hf_base = HF_BASE_MODEL_IDS.get(model_name)
        if _hf_base:
            _accepted_base.add(_hf_base)
            _accepted_base.add(_hf_base.rsplit("/", 1)[-1])
        _adapter_base = str(_ac.get("base_model_name_or_path") or "").rstrip("/")
        if not _adapter_base:
            raise ValueError(
                f"warm_start_adapter {warm_start_adapter!r} has no base_model_name_or_path in "
                f"adapter_config.json -- its base model cannot be verified against {model_name!r}. "
                "Refusing to load an adapter of unknown provenance."
            )
        _adapter_base_leaf = _adapter_base.rsplit("/", 1)[-1]
        if _adapter_base not in _accepted_base and _adapter_base_leaf not in _accepted_base:
            raise ValueError(
                f"warm_start_adapter base model {_adapter_base!r} does not match the GRPO "
                f"model {model_name!r} (accepted: {sorted(_accepted_base)}). Refusing to load "
                "an adapter trained on a different base onto this model."
            )

    from trl import GRPOConfig, GRPOTrainer

    cfg_lora = training_config["lora"]
    cfg_grpo = training_config["grpo"]
    cfg_common = training_config["common"]
    cond = CONDITIONS[condition_id]
    dtype = cfg_common.get("dtype", "bf16")
    effective_seed = int(seed) if seed is not None else int(cfg_common.get("seed", 42))
    norm_suffix = "" if advantage_norm == "z_score" else f"_{advantage_norm}"
    fullft_suffix = "_fullft" if full_finetune else ""
    polarity_suffix = "" if loss_polarity == "both" else f"_{loss_polarity}"
    idk_suffix = "_idk" if idk_prompt else ""
    run_tag = f"{model_name}_{condition_id}_s{effective_seed}{norm_suffix}{polarity_suffix}{idk_suffix}{fullft_suffix}"

    if smoke:
        cfg_grpo = {
            **cfg_grpo,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "num_generations": 2,
            "max_prompt_length": 512,
            "max_completion_length": 256,
        }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(
        model_name,
        dtype=dtype,
        device=cfg_common.get("primary_device", "cuda:0"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Defensive fix for fp16 + gradient_checkpointing + PEFT + accelerate bug
    # observed on Qwen2 0.5B at step 213: somewhere in the
    # autocast/gradient-checkpointing tape, input_ids gets restored as a
    # FloatTensor before reaching embed_tokens, which crashes the embedding
    # lookup. Coerce input_ids back to Long at the Qwen2Model.forward entry.
    # Applied to BOTH Qwen2Model and LlamaModel (cross-arch fear-v2 probe).
    import torch as _torch
    from transformers.models.qwen2 import modeling_qwen2 as _qwen2_mod
    if not getattr(_qwen2_mod.Qwen2Model.forward, "_asym_long_coerce", False):
        _orig_qwen2_model_forward = _qwen2_mod.Qwen2Model.forward
        def _qwen2_model_forward_long_coerce(self, input_ids=None, *args, **kwargs):
            if input_ids is not None and _torch.is_floating_point(input_ids):
                input_ids = input_ids.long()
            return _orig_qwen2_model_forward(self, input_ids=input_ids, *args, **kwargs)
        _qwen2_model_forward_long_coerce._asym_long_coerce = True
        _qwen2_mod.Qwen2Model.forward = _qwen2_model_forward_long_coerce
    try:
        from transformers.models.llama import modeling_llama as _llama_mod
        if not getattr(_llama_mod.LlamaModel.forward, "_asym_long_coerce", False):
            _orig_llama_model_forward = _llama_mod.LlamaModel.forward
            def _llama_model_forward_long_coerce(self, input_ids=None, *args, **kwargs):
                if input_ids is not None and _torch.is_floating_point(input_ids):
                    input_ids = input_ids.long()
                return _orig_llama_model_forward(self, input_ids=input_ids, *args, **kwargs)
            _llama_model_forward_long_coerce._asym_long_coerce = True
            _llama_mod.LlamaModel.forward = _llama_model_forward_long_coerce
    except ImportError:
        pass

    if full_finetune:
        lora_config = None
        # Full-FT path: drop lr, vllm_mem, force grad checkpointing to fit in
        # 38 GB co-resident with vLLM on the A100 40 GB. Overrides config-level
        # values only if not explicitly set by the caller.
        cfg_grpo = {
            **cfg_grpo,
            "learning_rate": float(cfg_grpo.get("learning_rate_full_ft", 5e-7)),
            "vllm_gpu_memory_utilization": float(
                cfg_grpo.get("vllm_gpu_memory_utilization_full_ft", 0.25)
            ),
            "gradient_checkpointing": True,
        }
    else:
        lora_config = LoraConfig(
            r=int(cfg_lora["r"]),
            lora_alpha=int(cfg_lora["alpha"]),
            lora_dropout=float(cfg_lora["dropout"]),
            target_modules=list(cfg_lora["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        )

    if warm_start_adapter is not None:
        # Warm-start: load the pretrained LoRA (e.g. the P6 SFT IDK adapter) as a
        # TRAINABLE PeftModel here, then pass peft_config=None so the trainer does
        # NOT re-wrap with a fresh randomly-initialized adapter. GRPO then continues
        # training the SAME adapter from step 0, keeping it comparable to every other
        # adapter in the sweep (D1: same base, same LoRA, no merge-and-new-adapter).
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, warm_start_adapter, is_trainable=True)
        lora_config = None
        print(f"[warm-start] loaded LoRA adapter from {warm_start_adapter} (is_trainable=True); "
              f"GRPO continues this adapter from step 0", flush=True)

    family = cond.get("family", "asym")
    # VC silently cancels its confidence signal under z_score / raw_rloo because
    # within-group z-score normalization removes the per-sample reward magnitude
    # that SC-LOO rescaling encodes (Finding-O failure mode). This would make a
    # VC run look like "VC doesn't work" when it actually never transmitted signal.
    # Require a magnitude-preserving norm so the effect reaches the gradient.
    if family == "vc" and advantage_norm in ("z_score", "raw_rloo"):
        raise ValueError(
            f"advantage_norm={advantage_norm!r} cancels VC's confidence signal under "
            "within-group normalization (Finding-O). Use --advantage-norm zhu_signed "
            "or global_mean so reward magnitude reaches the gradient."
        )
    if family == "vc":
        # Verifier-Calibration trains on the PRM800K phase-2 TRAIN split (judge
        # YES/NO over solution steps), NOT the MATH/GSM8K math-answer dataset.
        # Feeding the math dataset here would crash _fn_vc at step 0 (gold is a
        # math expression, not YES/NO). Eval/judging uses the TEST split.
        from asym_rewards.judge import build_judge_dataset, judge_problem_ids
        from asym_rewards.paths import PROCESSED_DIR
        train_path = PROCESSED_DIR / "phase2_train.jsonl"
        test_path = PROCESSED_DIR / "phase2_test.jsonl"
        # Hard abort: without the test split we cannot guarantee train/test leak
        # isolation (forbid=None would silently skip the D2 leak check).
        if not test_path.exists():
            raise FileNotFoundError(
                f"VC training requires {test_path} to enforce the D2 leak guard "
                "(no TEST problem may appear in training). "
                "Download PRM800K and run `make data` first."
            )
        forbid = judge_problem_ids(test_path)
        judge_limit = 256 if smoke else None
        judge_rows = build_judge_dataset(
            train_path, max_pos_per_row=1, max_neg_per_row=1,
            limit=judge_limit, forbid_problem_ids=forbid,
        )
        dataset = Dataset.from_list(judge_rows)
        print(f"[vc] built judge dataset: {len(judge_rows)} examples from {train_path} "
              f"(forbid {0 if forbid is None else len(forbid)} test problems)", flush=True)
    else:
        dataset = _build_training_dataset(smoke=smoke, idk_prompt=idk_prompt)
    forced_decode_fn = None
    if family == "d1":
        forced_decode_fn = _make_forced_decode_fn(model, tokenizer)
    reward_func = make_reward_func(condition_id, forced_decode_fn=forced_decode_fn)

    max_steps = 10 if smoke else int(cfg_grpo["max_steps"])
    use_vllm = False if smoke else bool(cfg_grpo.get("use_vllm", False))

    if cfg_grpo.get("use_vllm"):
        import torch
        import trl.trainer.grpo_trainer as _grpo_mod
        _orig_LLM = _grpo_mod.LLM
        _vllm_mem = float(cfg_grpo.get("vllm_gpu_memory_utilization", 0.25))
        _vllm_max_len = int(cfg_grpo["max_prompt_length"]) + int(cfg_grpo["max_completion_length"])
        # bf16 on Ampere+ (cc>=8.0: A100/H100); fp16 fallback on Volta/Turing
        # (V100 cc 7.0, 2080 Ti cc 7.5) which lack native bf16 kernels. Forcing
        # fp16 on Ampere triggers an illegal-memory-access in vLLM 0.6.4 when
        # the training model is bf16 (precision mismatch between weights and
        # sampler kernels), so we MUST switch on capability here.
        _cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
        _vllm_dtype = "bfloat16" if _cap[0] >= 8 else "float16"
        def _patched_LLM(*args, **kwargs):
            kwargs["dtype"] = _vllm_dtype
            kwargs["gpu_memory_utilization"] = _vllm_mem
            kwargs["max_model_len"] = _vllm_max_len  # cap KV-cache reservation
            kwargs["enforce_eager"] = True  # skip cudagraph capture (saves ~1 GB)
            # Co-resident vLLM + PEFT training on Ampere triggers an illegal
            # memory access in the sampler around step 3 with vLLM 0.6.4's
            # default cache settings. Disable prefix caching (cross-request
            # KV-block reuse) and CPU swap (forbids CPU<->GPU KV transfers)
            # to keep the rollout allocator deterministic between weight syncs.
            kwargs["enable_prefix_caching"] = False
            kwargs["swap_space"] = 0
            return _orig_LLM(*args, **kwargs)
        _grpo_mod.LLM = _patched_LLM
        # Force min_tokens=1 on every SamplingParams so no generation is empty.
        # Empty generations crash TRL's _prepare_inputs at argmax(dim=1) on a zero-size tensor.
        _orig_SP = _grpo_mod.SamplingParams
        def _patched_SP(*args, **kwargs):
            kwargs.setdefault("min_tokens", 1)
            return _orig_SP(*args, **kwargs)
        _grpo_mod.SamplingParams = _patched_SP

    effective_lr = float(lr_override) if lr_override is not None else float(cfg_grpo["learning_rate"])
    grpo_kwargs = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(cfg_grpo["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg_grpo["gradient_accumulation_steps"]),
        num_generations=int(cfg_grpo["num_generations"]),
        max_prompt_length=int(cfg_grpo["max_prompt_length"]),
        max_completion_length=int(cfg_grpo["max_completion_length"]),
        learning_rate=effective_lr,
        max_steps=max_steps,
        save_steps=int(cfg_grpo["save_steps"]),
        logging_steps=int(cfg_grpo["logging_steps"]),
        temperature=float(cfg_grpo["temperature"]),
        beta=float(cfg_grpo["beta"]),
        use_vllm=use_vllm,
        vllm_device=cfg_common.get("rollout_device", "cuda:1") if use_vllm else None,
        report_to=("wandb" if not smoke else "none"),
        run_name=run_tag,
        seed=effective_seed,
        bf16=(dtype == "bf16"),
        fp16=(dtype == "fp16"),
        gradient_checkpointing=bool(cfg_grpo.get("gradient_checkpointing", False)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if max_grad_norm is not None:
        grpo_kwargs["max_grad_norm"] = float(max_grad_norm)
    grpo_config = GRPOConfig(**grpo_kwargs)

    if grpo_config.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    use_cwar = cond.get("weighting") == "confidence"
    cwar_attempts_only = use_cwar and family == "d1"  # CW-D1: scale only attempt channels
    wrong_rescale_kind = cond.get("wrong_rescale", "cw")
    grf_delta_cfg = cond.get("delta_grf") if cond.get("group_rescale") == "grf" else None
    sc_delta_cfg = cond.get("delta_sc") if cond.get("group_rescale") == "sc" else None
    GuardedGRPOTrainer = _install_prepare_inputs_guards(
        GRPOTrainer,
        float(cfg_grpo.get("std_floor", 1e-4)),
        advantage_norm=advantage_norm,
        use_cwar=use_cwar,
        cwar_attempts_only=cwar_attempts_only,
        wrong_rescale=wrong_rescale_kind,
        lambda_oc=float(cond.get("lambda_oc", 0.0)),
        tau_oc=float(cond.get("tau_oc", 0.5)),
        gamma_cmf=float(cond.get("gamma_cmf", 0.0)),
        grf_delta=grf_delta_cfg,
        sc_delta=sc_delta_cfg,
        num_generations=int(cfg_grpo.get("num_generations", 1)),
        loss_polarity=loss_polarity,
    )

    trainer = GuardedGRPOTrainer(
        model=model,
        reward_funcs=[reward_func],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.add_callback(_make_telemetry_callback(family, cond.get("weighting")))

    # Resume support: if resume_from_checkpoint is an HF repo id (contains "/"
    # and not an absolute path), snapshot-download to a local cache first so
    # TRL/HF Trainer can load adapter + optimizer + scheduler + rng state.
    resume_arg: str | None = None
    if resume_from_checkpoint:
        if Path(resume_from_checkpoint).is_dir():
            resume_arg = resume_from_checkpoint
        else:
            from huggingface_hub import snapshot_download
            cache_dir = Path("/tmp") / "resume_ckpt" / resume_from_checkpoint.replace("/", "__")
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"[resume] downloading {resume_from_checkpoint} -> {cache_dir}", flush=True)
            snapshot_download(repo_id=resume_from_checkpoint, local_dir=str(cache_dir),
                              token=os.environ.get("HF_TOKEN"))
            resume_arg = str(cache_dir)
        print(f"[resume] passing resume_from_checkpoint={resume_arg} to trainer.train()", flush=True)

    train_result = trainer.train(resume_from_checkpoint=resume_arg) if resume_arg else trainer.train()
    trainer.save_model(str(out_dir))
    if not full_finetune:
        _fix_hf_base_model_refs(out_dir, model_name)

    train_metrics = getattr(train_result, "metrics", None) or {}
    meta = {
        "model": model_name,
        "condition": condition_id,
        "seed": effective_seed,
        "alpha": cond.get("alpha"),
        "beta": cond.get("beta"),
        "abstain": cond.get("abstain"),
        "family": family,
        "weighting": cond.get("weighting"),
        "advantage_norm": advantage_norm,
        "loss_polarity": loss_polarity,
        "lr_override": lr_override,
        "max_grad_norm": max_grad_norm,
        "learning_rate": effective_lr,
        "idk_prompt": idk_prompt,
        "warm_start_adapter": warm_start_adapter,
        "full_finetune": full_finetune,
        "grpo_beta": float(cfg_grpo["beta"]),
        "steps_completed": int(getattr(train_result, "global_step", max_steps)),
        "smoke": smoke,
        "final_train_loss": train_metrics.get("train_loss"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    hf_repo_template = cfg_common.get("hf_repo_template")
    if hf_repo_template and not smoke:
        from huggingface_hub import HfApi
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("WARNING: hf_repo_template set but HF_TOKEN env var missing; skipping push", flush=True)
        else:
            repo_id = hf_repo_template.format(
                model=model_name, condition=condition_id, seed=effective_seed
            ) + norm_suffix + polarity_suffix + idk_suffix + fullft_suffix
            api = HfApi(token=token)
            api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
            upload_patterns = (
                HF_UPLOAD_ALLOW_PATTERNS_FULLFT if full_finetune else HF_UPLOAD_ALLOW_PATTERNS
            )
            # Treat each entry as a glob so sharded fullFT weights
            # ("model-00001-of-N.safetensors", ...) are uploaded. Literal
            # names match themselves via glob, so adapter-only and unsharded
            # fullFT paths behave the same as before.
            for pattern in upload_patterns:
                for local_path in sorted(out_dir.glob(pattern)):
                    if local_path.is_file():
                        api.upload_file(
                            path_or_fileobj=str(local_path),
                            path_in_repo=local_path.name,
                            repo_id=repo_id,
                            repo_type="model",
                        )
            meta["hf_repo"] = repo_id
            hf_base_model_id = _hf_base_model_id(model_name)
            if hf_base_model_id:
                meta["base_model_hf"] = hf_base_model_id
            (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            api.upload_file(
                path_or_fileobj=str(out_dir / "meta.json"),
                path_in_repo="meta.json",
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"Pushed adapter to https://huggingface.co/{repo_id}", flush=True)

    return meta
