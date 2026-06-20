#!/usr/bin/env python
"""Extract last-token hidden states from a (possibly adapter-merged) causal LM.

Reads an existing *_rollouts.jsonl produced by eval_calibration.py, joins with
reconstructed prompts, runs batched forward passes on a GPU host, and writes a
compressed NPZ + companion .meta.json to --out-dir.

NOTE: Run this script only on an IDLE eval GPU. A 2080 Ti with more than ~1 GB
already used by another tenant will OOM during model load (gpu_memory_utilization
is implicitly the full device memory here, unlike vLLM's explicit 0.85 cap).
Check `nvidia-smi` before launching and pick a card with <1 GB used / <10% util.

    python scripts/extract_hidden_states.py \\
        --model qwen2_5_1_5b \\
        --condition S2 \\
        --adapter outputs/grpo/qwen2_5_1_5b_S2 \\
        --rollout outputs/eval/qwen2_5_1_5b_S2_bfloat16_math_test_rollouts.jsonl \\
        --benchmark math_test \\
        --max-per-label 256 \\
        --out-dir outputs/probing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asym_rewards.paths import MODEL_DIRS  # noqa: E402


# Prompt templates mirrored from eval_calibration.py so the join reconstructs
# the exact strings that were used during rollout generation.
_PROMPT_TEMPLATE = (
    "Solve the following problem step by step. "
    "Put your final answer in \\boxed{{}}.\n\n"
    "Problem: {problem}"
)
_MMLU_TEMPLATE = (
    "Answer the following multiple-choice question. "
    "Put the letter (A, B, C, or D) of the correct answer in \\boxed{{}}.\n\n"
    "Question: {problem}\n"
    "{choices_block}"
)
_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _prompts_by_pid(bench: str) -> dict[str, tuple[str, str, str]]:
    """Reconstruct prompt text per problem_id for a given benchmark.

    Mirrors eval_calibration._prompts_by_pid so the join is consistent.
    Returns pid -> (prompt_text, gold_answer, scorer).
    """
    import pandas as pd
    from asym_rewards.data import load_benchmark

    df = load_benchmark(bench)
    is_mc = bench == "mmlu_stem" or ("choices" in df.columns and df["choices"].notna().any())
    out: dict[str, tuple[str, str, str]] = {}
    for _, row in df.iterrows():
        pid = str(row["id"])
        gold = str(row["answer"])
        if is_mc:
            choices_val = row.get("choices")
            choices = list(choices_val) if choices_val is not None else []
            choices_block = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices))
            prompt = _MMLU_TEMPLATE.format(problem=str(row["prompt"]), choices_block=choices_block)
            scorer = "mc"
        else:
            prompt = _PROMPT_TEMPLATE.format(problem=str(row["prompt"]))
            scorer = "math"
        out[pid] = (prompt, gold, scorer)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract last-token hidden states from a causal LM for rollout items."
    )
    p.add_argument("--model", required=True, choices=list(MODEL_DIRS),
                   help="Model key into MODEL_DIRS (e.g. qwen2_5_1_5b)")
    p.add_argument("--condition", required=True,
                   help="Condition label used in output filename (e.g. S2, base)")
    p.add_argument("--adapter", default=None,
                   help="Path to a PEFT LoRA adapter directory; omit for base model")
    p.add_argument("--rollout", required=True,
                   help="Path to *_rollouts.jsonl produced by eval_calibration.py")
    p.add_argument("--benchmark", required=True,
                   help="Benchmark name used to reconstruct prompts (e.g. math_test)")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-per-label", type=int, default=None,
                   help="Maximum samples per label class (balanced subsample)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Sequences per forward pass (reduce if VRAM is tight)")
    p.add_argument("--max-len", type=int, default=2048,
                   help="Tokeniser truncation length in tokens")
    p.add_argument("--all-layers", action="store_true", default=True,
                   help="Gather hidden states from every layer (default: True)")
    p.add_argument("--last-layer-only", action="store_true",
                   help="Override --all-layers: gather only the final layer")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "probing"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from asym_rewards import probing as P

    all_layers = not args.last_layer_only
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  loading benchmark prompts for '{args.benchmark}' ...")
    prompts_by_pid = _prompts_by_pid(args.benchmark)
    print(f"  {len(prompts_by_pid)} problems in prompt table")

    rollout_path = Path(args.rollout)
    items = P.iter_rollout_records(
        rollout_path=rollout_path,
        prompts_by_pid=prompts_by_pid,
        max_per_label=args.max_per_label,
    )
    label_counts = {}
    for it in items:
        label_counts[it["label"]] = label_counts.get(it["label"], 0) + 1
    print(f"  {len(items)} items after join+subsample: {label_counts}")

    if not items:
        print("  ERROR: no items matched between rollout file and prompt table. Exiting.")
        return 1

    print(f"  loading model '{args.model}' (dtype={args.dtype}, device={args.device}) ...")
    model, tokenizer = P.load_probing_model(
        model_name=args.model,
        adapter_path=args.adapter,
        dtype=args.dtype,
        device=args.device,
    )
    print(f"  model loaded. Extracting hidden states (all_layers={all_layers}) ...")

    H, y, meta = P.extract_last_token_states(
        model=model,
        tokenizer=tokenizer,
        items=items,
        batch_size=args.batch_size,
        all_layers=all_layers,
        max_len=args.max_len,
    )
    print(f"  H shape: {H.shape}, y shape: {y.shape}")

    # Build layer name list: "embed" for index 0, "layer_N" for the rest.
    if all_layers:
        n_layers = H.shape[1]
        layer_names = ["embed"] + [f"layer_{i}" for i in range(n_layers - 1)]
    else:
        layer_names = ["last"]

    stem = f"{args.model}_{args.condition}_{args.dtype}_{args.benchmark}_hidden"
    out_path = out_dir / f"{stem}.npz"
    config = {
        "model": args.model,
        "condition": args.condition,
        "adapter": args.adapter,
        "dtype": args.dtype,
        "device": args.device,
        "benchmark": args.benchmark,
        "max_per_label": args.max_per_label,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "all_layers": all_layers,
    }
    P.save_probing_npz(
        out_path=out_path,
        H=H,
        y=y,
        meta=meta,
        layer_names=layer_names,
        config=config,
        source_rollout=str(rollout_path.resolve()),
    )
    print(f"  saved -> {out_path}")
    print(f"  companion -> {out_path.with_suffix('').with_suffix('.meta.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
