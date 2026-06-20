#!/usr/bin/env python
"""CLI wrapper for GRPO LoRA training on a single (model, condition, seed).

    python scripts/train_grpo.py --model qwen2_5_1_5b --condition S0
    python scripts/train_grpo.py --model qwen2_5_1_5b --condition S0 --seed 123
    python scripts/train_grpo.py --model qwen2_5_1_5b --condition S0 --smoke

The smoke path runs 10 steps on 50 prompts without vLLM; use it as the
end-to-end sanity check before launching the full 1500-step run.

Multi-seed sweeps (ICLR error bars): run S0/S3/C2 with seeds 42, 123, 456;
the single-seed default (42) is fine for the other four conditions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asym_rewards.grpo import ADVANTAGE_NORM_MODES, LOSS_POLARITY_MODES, train_grpo  # noqa: E402
from asym_rewards.reward import CONDITIONS  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="model name in src/asym_rewards/paths.py::MODEL_DIRS")
    p.add_argument("--condition", required=True, choices=list(CONDITIONS), help="reward condition id")
    p.add_argument("--seed", type=int, default=None,
                   help="override the config seed; output dir suffixed with _s{seed}")
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "training.yaml"))
    p.add_argument("--out-root", default=str(REPO_ROOT / "outputs" / "grpo"))
    p.add_argument("--smoke", action="store_true", help="10-step smoke run (no vllm, 50 prompts)")
    p.add_argument("--beta", type=float, default=None,
                   help="override GRPO KL coefficient (use --beta 0.0 to match Zhu PSR/NSR script)")
    p.add_argument("--advantage-norm", choices=list(ADVANTAGE_NORM_MODES), default="z_score",
                   help="z_score (TRL default), raw_rloo (mean-baseline only, Zhu W-REINFORCE/RLOO), "
                        "global_mean (running EMA mean as baseline; breaks per-group affine invariance), "
                        "or global_mean_std (running EMA mean + std).")
    p.add_argument("--loss-polarity", choices=list(LOSS_POLARITY_MODES), default="both",
                   help="Option A (Zhu-strict) polarity mask. 'positive_only' = Zhu PSR "
                        "(wrong samples excluded from baseline AND gradient). 'negative_only' = NSR. "
                        "Per-group mean/std recomputed over polarity-allowed subset; masked-out "
                        "advantages forced to 0. Empty-group telemetry exposes degenerate groups.")
    p.add_argument("--lr-override", type=float, default=None,
                   help="Override config learning_rate. Use for global_baseline stability runs.")
    p.add_argument("--max-grad-norm", type=float, default=None,
                   help="Pass-through to GRPOConfig.max_grad_norm. Use for global_baseline stability runs.")
    p.add_argument("--idk-prompt", action="store_true",
                   help="Train with the IDK-seeded prompt template: explicitly tells the model that "
                        "\\boxed{IDK} is acceptable when uncertain. Test whether the d1 IDK channel is "
                        "reachable on Qwen-1.5B given base-model probe = 1/80k spontaneous IDK on math. "
                        "Appends _idk to run_tag and HF repo.")
    p.add_argument("--full-finetune", action="store_true",
                   help="full-parameter fine-tune (peft_config=None). Drops lr to "
                        "configs.training.grpo.learning_rate_full_ft (default 5e-7) and "
                        "vllm_gpu_memory_utilization to ..._full_ft (default 0.25), "
                        "forces gradient checkpointing. Appends _fullft to run_tag and HF repo.")
    p.add_argument("--resume-from-checkpoint", default=None, type=str,
                   help="Resume training from a checkpoint. Accepts (a) a local TRL checkpoint dir "
                        "such as outputs/grpo/<tag>/checkpoint-300, OR (b) an HF repo id such as "
                        "ndsanjana/fear-qwen2_5_1_5b-SC-s42_raw_rloo-ckpt300. In case (b) the repo is "
                        "snapshot-downloaded to a local cache dir before passing to TRL.train(). "
                        "For a bare LoRA adapter (no trainer_state.json), use --warm-start-adapter "
                        "instead; this flag is for full TRL checkpoints only.")
    p.add_argument("--warm-start-adapter", default=None, type=str,
                   help="Initialize the LoRA from a pretrained adapter directory (e.g. the P6 SFT "
                        "IDK warm-start at outputs/sft/idk_warmstart_ep1), then train a FRESH GRPO "
                        "run from step 0 on that same adapter. Use this (NOT --resume-from-checkpoint) "
                        "for a bare adapter dir: a warm-start is not a checkpoint resume, so there is "
                        "no optimizer/scheduler/step state to restore.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.beta is not None:
        config["grpo"]["beta"] = float(args.beta)
    effective_seed = args.seed if args.seed is not None else int(config["common"].get("seed", 42))
    sub = "smoke" if args.smoke else "grpo"
    norm_suffix = "" if args.advantage_norm == "z_score" else f"_{args.advantage_norm}"
    effective_polarity = args.loss_polarity
    if args.advantage_norm in ("zhu_psr", "zhu_nsr", "zhu_signed") and effective_polarity != "both":
        # Mirror train_grpo's canonicalization HERE so the tag / out_dir / HF
        # name match the actual training (these modes encode/preserve polarity
        # in the advantage, so loss_polarity is forced to 'both' downstream).
        effective_polarity = "both"
    polarity_suffix = "" if effective_polarity == "both" else f"_{effective_polarity}"
    idk_suffix = "_idk" if args.idk_prompt else ""
    fullft_suffix = "_fullft" if args.full_finetune else ""
    tag = f"{args.model}_{args.condition}_s{effective_seed}{norm_suffix}{polarity_suffix}{idk_suffix}{fullft_suffix}"
    out_dir = Path(args.out_root).parent / sub / tag

    meta = train_grpo(
        model_name=args.model,
        condition_id=args.condition,
        training_config=config,
        out_dir=out_dir,
        smoke=args.smoke,
        seed=args.seed,
        advantage_norm=args.advantage_norm,
        full_finetune=args.full_finetune,
        loss_polarity=args.loss_polarity,
        lr_override=args.lr_override,
        max_grad_norm=args.max_grad_norm,
        idk_prompt=args.idk_prompt,
        resume_from_checkpoint=args.resume_from_checkpoint,
        warm_start_adapter=args.warm_start_adapter,
    )
    print(f"OK trained {tag} -> {out_dir}")
    print(f"   seed={meta['seed']} steps={meta['steps_completed']} "
          f"family={meta.get('family')} weighting={meta.get('weighting')} "
          f"advantage_norm={meta['advantage_norm']} polarity={meta.get('loss_polarity')} "
          f"lr={meta.get('learning_rate')} max_grad_norm={meta.get('max_grad_norm')} "
          f"full_finetune={meta.get('full_finetune')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
