#!/usr/bin/env python
"""Calibration / selective-prediction eval for a trained adapter.

    python scripts/eval_calibration.py --model qwen2_5_3b --condition S0 \\
        --adapter outputs/grpo/qwen2_5_3b_S0 \\
        --benchmarks math_test,gsm8k_test,aime_2024,aime_2025,mmlu_stem \\
        --k 16 --diff-baseline

For each benchmark, generate K rollouts per problem via vLLM on cuda:1,
score with math_verify, write rollouts JSONL + metrics JSON.

With --diff-baseline, also run the base model on a fixed 100-problem
MATH-test slice with identical seeds for a behavioral delta report
(Tier 2 of the "did it learn?" gate).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asym_rewards import metrics as M  # noqa: E402
from asym_rewards.data import load_benchmark  # noqa: E402
from asym_rewards.paths import MODEL_DIRS  # noqa: E402
from asym_rewards.reward import (  # noqa: E402
    CONDITIONS,
    extract_boxed_answer,
    is_correct,
    is_idk_marker,
    rescore_change_counts,
    rescore_rollouts,
    score_mc,
)


PROMPT_TEMPLATE = (
    "Solve the following problem step by step. "
    "Put your final answer in \\boxed{{}}.\n\n"
    "Problem: {problem}"
)

# IDK-seeded variant (Option A, instruction only): matches src/asym_rewards/grpo.py::PROMPT_TEMPLATE_IDK.
# Selected via --idk-prompt for evals of adapters trained with that template.
PROMPT_TEMPLATE_IDK = (
    "Solve the following problem step by step. "
    "Put your final answer in \\boxed{{}}. "
    "If you cannot solve the problem, write \\boxed{{IDK}} as your answer.\n\n"
    "Problem: {problem}"
)

# MMLU-STEM rows are multiple-choice with `choices: list[str]` and `answer`
# storing the numeric index (0..3). Show the options and ask for the letter.
MMLU_TEMPLATE = (
    "Answer the following multiple-choice question. "
    "Put the letter (A, B, C, or D) of the correct answer in \\boxed{{}}.\n\n"
    "Question: {problem}\n"
    "{choices_block}"
)

MMLU_TEMPLATE_IDK = (
    "Answer the following multiple-choice question. "
    "Put the letter (A, B, C, or D) of the correct answer in \\boxed{{}}. "
    "If you do not know the answer, write \\boxed{{IDK}} instead.\n\n"
    "Question: {problem}\n"
    "{choices_block}"
)

# Resolved templates: chosen at main() based on --idk-prompt.
_ACTIVE_PROMPT_TEMPLATE = PROMPT_TEMPLATE
_ACTIVE_MMLU_TEMPLATE = MMLU_TEMPLATE

_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def seed_for(problem_id: str, sample_idx: int) -> int:
    digest = hashlib.sha256(f"{problem_id}:{sample_idx}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


def _parse_seed(adapter: str | None, default_seed: int = 42) -> int | None:
    """Extract the training seed encoded in an adapter dir name (``..._s<N>``).

    Returns the int seed, or None when no adapter / no ``_s<N>`` token is present
    (e.g. base-model eval). Tolerant of trailing variant suffixes such as
    ``_s42_raw_rloo`` (matches the first ``_s<digits>`` group)."""
    if not adapter:
        return None
    m = re.search(r"_s(\d+)", os.path.basename(os.path.normpath(adapter)))
    return int(m.group(1)) if m else None


def derive_seed_tag(adapter: str | None, output_tag: str, default_seed: int = 42) -> str:
    """Disambiguator inserted into the output stem between <condition> and <dtype>.

    Precedence: an explicit ``--output-tag`` always wins (returned as ``_<tag>``).
    Otherwise, if the adapter encodes a NON-default training seed, auto-insert
    ``_s<seed>`` so different-seed evals of the same condition land in distinct
    filename slots instead of silently overwriting / re-scoring each other. The
    default seed (42) keeps legacy untagged filenames for backward compatibility
    so existing results files are never orphaned."""
    if output_tag:
        return f"_{output_tag}"
    seed = _parse_seed(adapter, default_seed)
    if seed is not None and seed != default_seed:
        return f"_s{seed}"
    return ""


def provenance_mismatch(existing: dict, current: dict,
                        keys: tuple[str, ...] = ("adapter", "k", "dtype", "idk_prompt", "idk_prompt_template_hash")) -> list[tuple]:
    """Keys whose recorded value differs from the current run's, for the guard.

    Returns a list of ``(key, existing_value, current_value)`` tuples. A None in
    the existing record is treated as 'unknown' and never flagged (legacy/partial
    sidecars do not trigger a false clash)."""
    out: list[tuple] = []
    for key in keys:
        ev = existing.get(key)
        cv = current.get(key)
        if ev is not None and ev != cv:
            out.append((key, ev, cv))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--condition", default="base",
                   help="condition label for output filenames; use 'base' for untuned model")
    p.add_argument("--adapter", default=None,
                   help="path to trained LoRA adapter dir (peft format); omit for base model eval")
    p.add_argument("--benchmarks", default="math_test,gsm8k_test,aime_2024,aime_2025,mmlu_stem")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="0 = full benchmark (smoke: small int)")
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "training.yaml"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "eval"))
    p.add_argument("--diff-baseline", action="store_true", help="also run base model for delta report")
    p.add_argument("--deep-passatk", action="store_true",
                   help="additionally run K=256 on a fixed 200-problem MATH-test slice")
    p.add_argument("--probe-abstentions", action="store_true",
                   help="for each abstained sample, force '\\boxed{' continuation and capture "
                        "the answer + top-5 first-token logprobs (fear-vs-uncertainty probe)")
    p.add_argument("--greedy", action="store_true",
                   help="paired-comparison mode: force temperature=0 and K=1 so cross-condition "
                        "transitions become a clean natural experiment (same problem, same starting "
                        "context, only weights differ). Writes to a separate `_greedy` filename so it "
                        "does not collide with the stochastic K=16 run.")
    p.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16"],
                   help="vLLM dtype; 'auto' picks bf16 on Ampere+, fp16 on Volta/Turing")
    p.add_argument("--idk-prompt", action="store_true",
                   help="Use IDK-seeded prompt templates (matches src/asym_rewards/grpo.py "
                        "PROMPT_TEMPLATE_IDK / MMLU_TEMPLATE_IDK). Required when evaluating an "
                        "adapter trained with --idk-prompt so the inference prompt matches training.")
    p.add_argument("--output-tag", default="",
                   help="optional disambiguator inserted into the output filename stem between "
                        "<condition> and <dtype>. Use when running the same --condition with a "
                        "different --adapter (e.g. S2 main vs S2 raw_rloo variant) so the eval "
                        "writes to a distinct filename slot instead of re-scoring cached rollouts "
                        "from the prior run. Must match [A-Za-z0-9_-]+. Empty = auto: a NON-default "
                        "training seed in --adapter (_s<N>, N!=42) auto-inserts _s<N>; default seed "
                        "stays legacy untagged. An explicit --output-tag overrides the auto seed tag.")
    args = p.parse_args()
    if args.output_tag and not re.fullmatch(r"[A-Za-z0-9_-]+", args.output_tag):
        p.error(f"--output-tag must match [A-Za-z0-9_-]+; got: {args.output_tag!r}")
    if args.condition in CONDITIONS and args.adapter is None:
        p.error(f"--adapter is required when --condition is a trained condition ({args.condition}); "
                "omit --condition or pass --condition base for untuned base model eval")
    if args.greedy and args.diff_baseline:
        p.error("--greedy is incompatible with --diff-baseline: the diff report compares K=4 samples "
                "but greedy mode produces K=1, so samples 1..3 would be reported as 'missing' and "
                "every problem would falsely count as a changed answer. Run diff-baseline against "
                "the K=16 stochastic eval, not greedy.")
    return args


def _resolve_dtype(dtype: str, rollout_device: str) -> str:
    """Resolve 'auto' to bfloat16 on Ampere+ (cc>=8) and float16 on Volta/Turing.

    Queries the rollout GPU, not device 0, so a host with mixed GPUs (e.g. V100 at
    index 0 + 2080 Ti at index 1 with rollout_device=cuda:1) is resolved correctly.
    """
    if dtype != "auto":
        return dtype
    import torch
    if not torch.cuda.is_available():
        return "bfloat16"
    device_idx = int(rollout_device.split(":")[-1]) if ":" in rollout_device else 0
    major, _ = torch.cuda.get_device_capability(device_idx)
    return "bfloat16" if major >= 8 else "float16"


def _vllm_engine(model_path: str, adapter_path: str | None, rollout_device: str,
                 max_completion_length: int, dtype: str = "bfloat16",
                 probe_headroom: int = 0):
    """Spin up a vLLM engine; if `adapter_path`, merge LoRA into the base weights first.

    `probe_headroom` widens max_model_len when the caller plans to resume
    generation from existing completions (the abstention probe appends
    \\boxed{ to (prompt + completion) and asks for ~64 more tokens, which
    can overshoot the default headroom on the worst-case "rambling
    abstention" where the original completion ran to max_completion_length).
    """
    from vllm import LLM

    device_idx = int(rollout_device.split(":")[-1]) if ":" in rollout_device else 0
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(device_idx))

    if adapter_path:
        # Detect full-fine-tune vs LoRA adapter. Full-FT saves the model
        # directly (config.json + model.safetensors); LoRA saves adapter_config.json
        # and is merged into the base for vLLM consumption. Prefer the meta.json
        # signal since we control it; fall back to adapter_config.json presence.
        import json as _json
        meta_path = Path(adapter_path) / "meta.json"
        full_finetune = False
        if meta_path.exists():
            try:
                full_finetune = bool(_json.loads(meta_path.read_text()).get("full_finetune", False))
            except Exception:
                full_finetune = False
        if not full_finetune and not (Path(adapter_path) / "adapter_config.json").exists():
            full_finetune = (Path(adapter_path) / "config.json").exists()

        if full_finetune:
            # Full-FT path: directory IS the model; load straight into vLLM,
            # no merge step required. Tokenizer files live alongside the
            # model weights in the same dir (HF upload included them).
            weights_path = adapter_path
        else:
            # LoRA path: merge into a standalone-weights cache so vLLM can load
            # without runtime LoRA support. Cache by dtype so a fp16 run does
            # not reuse a bf16-saved merge (and vice versa).
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
            merged_dir = Path(adapter_path) / f"_merged_for_vllm_{dtype}"
            if not merged_dir.exists():
                base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
                merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
                merged.save_pretrained(str(merged_dir))
                AutoTokenizer.from_pretrained(model_path).save_pretrained(str(merged_dir))
            weights_path = str(merged_dir)
    else:
        weights_path = model_path

    return LLM(model=weights_path, dtype=dtype, gpu_memory_utilization=0.85,
               max_model_len=max_completion_length + 1024 + probe_headroom)


def _generate_k(llm, prompts: list[str], seeds: list[int], temperature: float, max_tokens: int) -> list[str]:
    """Batched vLLM generation, one completion per (prompt, seed) pair.

    vLLM accepts a parallel list of SamplingParams - one per prompt - so the
    per-sample seed is preserved while still letting the engine batch the
    forward pass across all prompts in one shot.
    """
    from vllm import SamplingParams

    params = [SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=s, n=1) for s in seeds]
    outputs = llm.generate(prompts, params, use_tqdm=True)
    return [out.outputs[0].text for out in outputs]


def _score_rollouts(
    prompts_meta: list[dict],
    completions: list[str],
    family: str | None = None,
) -> list[dict]:
    """Attach extracted/correct/abstained to each completion.

    For the d1 family, `\\boxed{IDK}` (and the other IDK markers in
    reward._IDK_MARKERS) counts as abstention, matching the training-side
    semantics in d1_reward. Without this, D1/CWD1 rollouts with an explicit
    IDK box are scored as wrong-non-abstained and the abstention rate,
    selective coverage, and diff metrics are biased.
    """
    is_d1 = (family == "d1")
    rollouts = []
    for meta, completion in zip(prompts_meta, completions):
        extracted = extract_boxed_answer(completion)
        if meta.get("scorer") == "mc":
            correct = score_mc(extracted, meta["gold"])
        else:
            correct = is_correct(extracted, meta["gold"])
        abstained = (extracted == "") or (is_d1 and is_idk_marker(extracted))
        if abstained:
            correct = False
        rollouts.append({
            **meta,
            "completion": completion,
            "extracted": extracted,
            "correct": correct,
            "abstained": abstained,
        })
    return rollouts


def _rollouts_to_records(rollouts: list[dict]) -> list[dict]:
    by_pid: dict[str, list[dict]] = {}
    for r in rollouts:
        by_pid.setdefault(r["problem_id"], []).append(
            {"correct": r["correct"], "abstained": r["abstained"]}
        )
    return [{"problem_id": pid, "samples": ss} for pid, ss in by_pid.items()]


def _compute_metrics(rollouts: list[dict], k: int) -> dict:
    records = _rollouts_to_records(rollouts)
    p_hat = np.array([
        sum(s["correct"] for s in r["samples"]) / max(len(r["samples"]), 1)
        for r in records
    ])
    correct_any = np.array([int(any(s["correct"] for s in r["samples"])) for r in records])
    return {
        "n_problems": len(records),
        "n_samples_total": sum(len(r["samples"]) for r in records),
        "accuracy_pass1": M.accuracy(records),
        "abstention_rate": M.abstention_rate(records),
        "ece": M.expected_calibration_error(p_hat, correct_any),
        "brier": M.brier_score(p_hat, correct_any),
        "auacc": M.auacc(p_hat, correct_any),
        "selective_acc_at_50": M.selective_accuracy_at_coverage(p_hat, correct_any, 0.5),
        "selective_acc_at_80": M.selective_accuracy_at_coverage(p_hat, correct_any, 0.8),
        "pass_at_1": M.pass_at_k_corpus(records, 1),
        "pass_at_4": M.pass_at_k_corpus(records, min(4, k)),
        "pass_at_k": M.pass_at_k_corpus(records, k),
    }


def _build_prompts(df: pd.DataFrame, k: int, bench: str = "") -> tuple[list[str], list[int], list[dict]]:
    is_mc = bench == "mmlu_stem" or ("choices" in df.columns and df["choices"].notna().any())
    prompts, seeds, meta = [], [], []
    for _, row in df.iterrows():
        pid = str(row["id"])
        gold = str(row["answer"])
        if is_mc:
            choices_val = row.get("choices")
            choices = list(choices_val) if choices_val is not None else []
            choices_block = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices))
            prompt = _ACTIVE_MMLU_TEMPLATE.format(problem=str(row["prompt"]), choices_block=choices_block)
            scorer = "mc"
        else:
            prompt = _ACTIVE_PROMPT_TEMPLATE.format(problem=str(row["prompt"]))
            scorer = "math"
        for s in range(k):
            prompts.append(prompt)
            seeds.append(seed_for(pid, s))
            meta.append({"problem_id": pid, "sample_idx": s, "gold": gold, "scorer": scorer})
    return prompts, seeds, meta


def _eval_benchmark(llm, bench: str, k: int, limit: int, temperature: float, max_tokens: int,
                    out_path: Path, family: str | None = None) -> list[dict]:
    df = load_benchmark(bench)
    if limit > 0:
        df = df.head(limit).reset_index(drop=True)
    prompts, seeds, meta = _build_prompts(df, k, bench=bench)
    print(f"  {bench}: {len(df)} problems x K={k} = {len(prompts)} rollouts")
    completions = _generate_k(llm, prompts, seeds, temperature, max_tokens)
    rollouts = _score_rollouts(meta, completions, family=family)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rollouts:
            f.write(json.dumps(r) + "\n")
    return rollouts


def _prompts_by_pid(bench: str) -> dict[str, tuple[str, str, str]]:
    """Rebuild the eval prompt for every benchmark row, keyed by problem_id.

    Returns pid -> (prompt_text, gold_answer, scorer in {'math','mc'}). Used by
    the abstention probe, which needs the exact original prompt to resume from.
    """
    df = load_benchmark(bench)
    is_mc = bench == "mmlu_stem" or ("choices" in df.columns and df["choices"].notna().any())
    out: dict[str, tuple[str, str, str]] = {}
    for _, row in df.iterrows():
        pid = str(row["id"]); gold = str(row["answer"])
        if is_mc:
            choices_val = row.get("choices")
            choices = list(choices_val) if choices_val is not None else []
            choices_block = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices))
            prompt = _ACTIVE_MMLU_TEMPLATE.format(problem=str(row["prompt"]), choices_block=choices_block)
            scorer = "mc"
        else:
            prompt = _ACTIVE_PROMPT_TEMPLATE.format(problem=str(row["prompt"]))
            scorer = "math"
        out[pid] = (prompt, gold, scorer)
    return out


def _probe_abstentions(llm, bench: str, rollouts: list[dict], out_path: Path) -> dict:
    """Force '\\boxed{' continuation on every abstained sample; capture answer + top-5 logprobs.

    Returns a small summary dict. JSONL written to `out_path`, one row per
    probe with: forced_completion, forced_extracted, forced_correct, top5_first_token.
    """
    from vllm import SamplingParams

    abstained = [r for r in rollouts if r.get("abstained")]
    if not abstained:
        # Touch the final path atomically (write+rename) so existence is a
        # reliable "this benchmark is done" marker even with zero probes.
        tmp_empty = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_empty.write_text("")
        tmp_empty.replace(out_path)
        return {"n_abstentions": 0, "n_probed": 0, "forced_correct": 0,
                "forced_correct_rate_among_abstentions": 0.0}

    prompts_by_pid = _prompts_by_pid(bench)
    probe_prompts: list[str] = []
    probe_meta: list[dict] = []
    for r in abstained:
        if r["problem_id"] not in prompts_by_pid:
            continue
        base_prompt, gold, scorer = prompts_by_pid[r["problem_id"]]
        completion = (r.get("completion") or "").rstrip()
        # Append "\\boxed{" so the next token IS the start of the forced answer;
        # stop on "}" to cap a short, well-formed extraction.
        probe_prompt = f"{base_prompt}{completion}\n\\boxed{{"
        probe_prompts.append(probe_prompt)
        probe_meta.append({
            "problem_id": r["problem_id"], "sample_idx": r["sample_idx"],
            "gold": gold, "scorer": scorer,
        })

    # Do NOT stop on '}' - many valid math answers have nested braces
    # (e.g. \frac{1}{2}, \sqrt{x+1}). Generate enough text to comfortably hold
    # the longest typical boxed answer, then post-process to slice at the
    # matching closing brace (depth-1 since we already consumed the opening
    # '{' in the prefix). Stop on a paragraph break or a fresh \boxed{ open
    # (defensive: model occasionally restarts the box).
    params = SamplingParams(temperature=0.0, max_tokens=64, logprobs=5,
                            stop=["\n\n", "\\boxed{"])
    outputs = llm.generate(probe_prompts, params, use_tqdm=True)

    def _slice_to_matching_brace(text: str) -> str:
        """Slice `text` up to the '}' that closes the prefix's '\\boxed{'.

        The prefix consumed one '{' so we start at depth 1. The first time
        depth returns to 0 is the answer's closing brace. If no matching
        brace is found within `text`, return the whole string (rare; the
        model produced longer than max_tokens without closing - we will
        still try to extract via is_correct, which is brace-tolerant).
        """
        depth = 1
        for i, c in enumerate(text):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[:i]
        return text

    # Write to a `.tmp` sibling first, then atomically rename only after the
    # loop finishes. Prevents a SIGINT/OOM mid-write from leaving a truncated
    # JSONL that the rerun existence check would treat as complete.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    n_correct = 0
    with tmp_path.open("w") as f:
        for meta, out in zip(probe_meta, outputs):
            comp = out.outputs[0]
            forced_text = comp.text
            # vLLM returns logprobs as List[Optional[Dict[int, Logprob]]] - one per
            # generated token. We capture the top-5 at position 0 because that is
            # the model's first committed token of the answer.
            top5: list[dict] = []
            if comp.logprobs and comp.logprobs[0]:
                first = comp.logprobs[0]
                ranked = sorted(first.items(), key=lambda kv: getattr(kv[1], "rank", 999))
                for tid, lp in ranked[:5]:
                    top5.append({
                        "token_id": int(tid),
                        "token": getattr(lp, "decoded_token", ""),
                        "logprob": float(lp.logprob),
                        "rank": int(getattr(lp, "rank", -1)),
                    })
            sliced = _slice_to_matching_brace(forced_text)
            forced_extracted = sliced.strip()
            if meta["scorer"] == "mc":
                forced_correct = score_mc(forced_extracted, meta["gold"])
            else:
                forced_correct = is_correct(forced_extracted, meta["gold"])
            if forced_correct:
                n_correct += 1
            f.write(json.dumps({
                "problem_id": meta["problem_id"],
                "sample_idx": meta["sample_idx"],
                "gold": meta["gold"],
                "scorer": meta["scorer"],
                "forced_completion": forced_text,
                "forced_extracted": forced_extracted,
                "forced_correct": bool(forced_correct),
                "top5_first_token": top5,
            }) + "\n")
    # Atomic rename: only declare the probe file complete once the loop
    # finishes without exception. A partial `.tmp` file left behind by an
    # earlier crash is overwritten on the next run rather than mistaken
    # for completed output.
    tmp_path.replace(out_path)

    return {
        "n_abstentions": len(abstained),
        "n_probed": len(probe_meta),
        "forced_correct": n_correct,
        "forced_correct_rate_among_abstentions": (n_correct / max(len(probe_meta), 1)),
    }


def _diff_baseline(model_path: str, rollout_device: str, max_tokens: int, k_diff: int,
                   slice_size: int, trained_rollouts: list[dict], out_path: Path,
                   dtype: str = "bfloat16", family: str | None = None) -> None:
    """Run base model on the same MATH-test problems; write delta report."""
    df = load_benchmark("math_test").head(slice_size).reset_index(drop=True)
    slice_pids = set(df["id"].astype(str))
    base_llm = _vllm_engine(model_path, adapter_path=None,
                            rollout_device=rollout_device, max_completion_length=max_tokens,
                            dtype=dtype)
    prompts, seeds, meta = _build_prompts(df, k_diff, bench="math_test")
    base_rollouts = _score_rollouts(
        meta, _generate_k(base_llm, prompts, seeds, 0.7, max_tokens), family=family,
    )

    trained_in_slice = [r for r in trained_rollouts
                        if r["problem_id"] in slice_pids and r["sample_idx"] < k_diff]
    trained_pids = {r["problem_id"] for r in trained_in_slice}

    base_metrics = _compute_metrics(base_rollouts, k_diff)
    trained_metrics = _compute_metrics(trained_in_slice, k_diff)

    base_by_key = {(r["problem_id"], r["sample_idx"]): r["extracted"] for r in base_rollouts}
    trained_by_key = {(r["problem_id"], r["sample_idx"]): r["extracted"] for r in trained_in_slice}
    diff_pids = {
        pid for pid in trained_pids
        if any(base_by_key.get((pid, s)) != trained_by_key.get((pid, s)) for s in range(k_diff))
    }

    base_first_by_pid: dict[str, dict] = {}
    for r in base_rollouts:
        base_first_by_pid.setdefault(r["problem_id"], r)
    trained_first_by_pid: dict[str, dict] = {}
    for r in trained_in_slice:
        trained_first_by_pid.setdefault(r["problem_id"], r)

    sample_diffs = [
        {
            "problem_id": pid,
            "base":    [base_first_by_pid[pid]] if pid in base_first_by_pid else [],
            "adapter": [trained_first_by_pid[pid]] if pid in trained_first_by_pid else [],
        }
        for pid in list(diff_pids)[:5]
    ]

    diff_report = {
        "slice_size": slice_size,
        "k_diff": k_diff,
        "accuracy_delta": trained_metrics["accuracy_pass1"] - base_metrics["accuracy_pass1"],
        "abstention_rate_delta": trained_metrics["abstention_rate"] - base_metrics["abstention_rate"],
        "ece_delta": trained_metrics["ece"] - base_metrics["ece"],
        "brier_delta": trained_metrics["brier"] - base_metrics["brier"],
        "n_problems_with_different_answer": len(diff_pids),
        "n_problems_total": len(trained_pids),
        "sample_diffs": sample_diffs,
        "base_metrics": base_metrics,
        "trained_metrics": trained_metrics,
    }
    out_path.write_text(json.dumps(diff_report, indent=2, default=str))


def main() -> int:
    args = parse_args()
    if args.idk_prompt:
        global _ACTIVE_PROMPT_TEMPLATE, _ACTIVE_MMLU_TEMPLATE
        _ACTIVE_PROMPT_TEMPLATE = PROMPT_TEMPLATE_IDK
        _ACTIVE_MMLU_TEMPLATE = MMLU_TEMPLATE_IDK
        print("  --idk-prompt: using IDK-seeded prompt templates for all benchmarks")
    config = yaml.safe_load(Path(args.config).read_text())
    eval_cfg = config["eval"]
    common = config["common"]
    rollout_device = common.get("rollout_device", "cuda:1")
    max_tokens = int(eval_cfg["max_completion_length"])
    temperature = float(eval_cfg["temperature"])
    k_samples = args.k

    if args.greedy:
        temperature = 0.0
        k_samples = 1
        print(f"  --greedy: forcing temperature=0.0, K=1 (paired cross-condition mode)")

    model_path = str(MODEL_DIRS[args.model])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Family of the trained condition (d1 / asym / None for base). Threaded
    # into _score_rollouts so D1 IDK boxes count as abstentions in eval,
    # matching the training-side d1_reward semantics.
    cond_family = (
        CONDITIONS[args.condition].get("family") if args.condition in CONDITIONS else None
    )

    resolved_dtype = _resolve_dtype(args.dtype, rollout_device)
    print(f"  vLLM dtype: {resolved_dtype} (--dtype={args.dtype}, rollout_device={rollout_device})")
    # Reserve 128 extra tokens of model-len headroom when probing: 64 for the
    # forced \boxed{} continuation + some slack for prompt+completion overshoot
    # on worst-case rambling abstentions.
    probe_headroom = 128 if args.probe_abstentions else 0
    llm = _vllm_engine(model_path, adapter_path=args.adapter,
                       rollout_device=rollout_device, max_completion_length=max_tokens,
                       dtype=resolved_dtype, probe_headroom=probe_headroom)

    # Output filenames include resolved dtype so a bf16 result file is never
    # silently reused for a fp16 rerun (different numerics => different rollouts).
    # `--greedy` adds a suffix so paired-mode runs do not collide with the K=16
    # stochastic eval (different sampling regime => not interchangeable).
    # `--output-tag` inserts a disambiguator between condition and dtype so the
    # same --condition with a different --adapter (e.g. raw_rloo variant) writes
    # to a distinct slot instead of hitting the "rollouts file exists" re-score path.
    suffix = "_greedy" if args.greedy else ""
    tag = derive_seed_tag(args.adapter, args.output_tag)
    stem = f"{args.model}_{args.condition}{tag}_{resolved_dtype}{suffix}"
    # Provenance stamped into every rollouts sidecar + metrics file so a result
    # can always be traced to the adapter/seed/k/dtype that produced it (the
    # filename alone encodes neither seed nor k). The re-score fast-path below
    # refuses to silently re-grade a cached file whose provenance clashes.
    import hashlib
    idk_template_hash = (
        hashlib.sha256(PROMPT_TEMPLATE_IDK.encode("utf-8")).hexdigest()[:12]
        if args.idk_prompt else None
    )
    provenance = {
        "adapter":   args.adapter,
        "seed":      _parse_seed(args.adapter),
        "k":         k_samples,  # effective K (greedy forces 1); NOT args.k, so greedy reruns do not falsely clash
        "dtype":     resolved_dtype,
        "condition": args.condition,
        "greedy":    bool(args.greedy),
        # Prompt template choice: distinct prompt -> distinct rollouts; cache
        # reuse across modes would silently mislabel the result. Codex P2
        # fix 2026-06-01. The hash captures the actual template content so
        # changing PROMPT_TEMPLATE_IDK invalidates the prior cache automatically.
        "idk_prompt": bool(args.idk_prompt),
        "idk_prompt_template_hash": idk_template_hash,
    }
    benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    all_rollouts_by_bench: dict[str, list[dict]] = {}
    for bench in benches:
        rollouts_path = out_dir / f"{stem}_{bench}_rollouts.jsonl"
        meta_path = out_dir / f"{stem}_{bench}_rollouts.meta.json"
        if rollouts_path.exists():
            # Guard: the stem already encodes condition+tag+dtype, so a cached
            # file whose recorded adapter/k/dtype differs from this run is a real
            # provenance clash the filename did not capture (e.g. a second seed
            # or a different K landing on the same slot). Refuse to re-score it
            # silently - that would emit numbers attributed to the wrong run.
            if meta_path.exists():
                clashes = provenance_mismatch(json.loads(meta_path.read_text()), provenance)
                if clashes:
                    detail = "; ".join(f"{k}: cached={ev!r} vs current={cv!r}" for k, ev, cv in clashes)
                    raise SystemExit(
                        f"[{bench}] cached rollouts at {rollouts_path} were produced with a different "
                        f"provenance ({detail}). Refusing to silently re-score and mislabel the result. "
                        f"Pass a distinct --output-tag (or remove the stale file) to write a new slot.")
            else:
                print(f"  {bench}: rollouts exist with NO provenance sidecar (legacy file); "
                      f"re-scoring but provenance is UNVERIFIED")
            print(f"  {bench}: rollouts file exists, loading + re-scoring")
            with rollouts_path.open() as f:
                rollouts = [json.loads(line) for line in f]
            rescored = rescore_rollouts(rollouts, family=cond_family)
            changes = rescore_change_counts(rollouts, rescored)
            if any(changes.values()):
                print(f"  {bench}: re-score changed cached labels {changes}; rewriting rollouts")
                with rollouts_path.open("w") as f:
                    for r in rescored:
                        f.write(json.dumps(r) + "\n")
            rollouts = rescored
        else:
            rollouts = _eval_benchmark(llm, bench, k_samples, args.limit, temperature, max_tokens, rollouts_path, family=cond_family)
            meta_path.write_text(json.dumps(provenance, indent=2))
        all_rollouts_by_bench[bench] = rollouts
        metrics = _compute_metrics(rollouts, k_samples)
        if args.greedy:
            # At K=1, p_hat equals correct_any exactly, so ECE/Brier collapse to
            # 0 and AUACC/selective-accuracy are tautological. Drop the misleading
            # calibration fields; keep only what is well-defined at K=1.
            metrics = {
                "_mode": "greedy_K1",
                "_note": ("Calibration metrics (ECE, Brier, AUACC, selective_acc) are "
                          "ill-defined at K=1 and have been omitted. Use the K=16 "
                          "stochastic eval for those."),
                "n_problems":      metrics["n_problems"],
                "n_samples_total": metrics["n_samples_total"],
                "accuracy_pass1":  metrics["accuracy_pass1"],
                "abstention_rate": metrics["abstention_rate"],
            }
        metrics["_provenance"] = provenance
        (out_dir / f"{stem}_{bench}_metrics.json").write_text(
            json.dumps(metrics, indent=2)
        )
        if args.greedy:
            print(f"  {bench}: acc={metrics['accuracy_pass1']:.3f} abst={metrics['abstention_rate']:.3f} (greedy, K=1)")
        else:
            print(f"  {bench}: acc={metrics['accuracy_pass1']:.3f} abst={metrics['abstention_rate']:.3f} "
                  f"ece={metrics['ece']:.3f} auacc={metrics['auacc']:.3f}")

        if args.probe_abstentions:
            probes_path = out_dir / f"{stem}_{bench}_probes.jsonl"
            summary_path = out_dir / f"{stem}_{bench}_probes_summary.json"
            # Skip only when BOTH artifacts exist. If only the JSONL is present
            # (rerun after a crash between rename and summary write), recompute
            # the summary from the JSONL without re-running the GPU probe.
            if probes_path.exists() and summary_path.exists():
                print(f"  {bench}: probes + summary exist, skipping probe")
            elif probes_path.exists():
                print(f"  {bench}: probes exist but summary missing - recomputing summary only")
                with probes_path.open() as f:
                    probes = [json.loads(line) for line in f if line.strip()]
                n_correct = sum(1 for p in probes if p.get("forced_correct"))
                probe_summary = {
                    "n_abstentions": sum(1 for r in rollouts if r.get("abstained")),
                    "n_probed": len(probes),
                    "forced_correct": n_correct,
                    "forced_correct_rate_among_abstentions": n_correct / max(len(probes), 1),
                }
                summary_path.write_text(json.dumps(probe_summary, indent=2))
            else:
                probe_summary = _probe_abstentions(llm, bench, rollouts, probes_path)
                summary_path.write_text(json.dumps(probe_summary, indent=2))
                print(f"  {bench}: probed {probe_summary['n_probed']} abstentions, "
                      f"forced-correct={probe_summary['forced_correct']} "
                      f"({probe_summary['forced_correct_rate_among_abstentions']:.3f})")

    if args.deep_passatk and not args.greedy and "math_test" in all_rollouts_by_bench:
        k_deep = int(eval_cfg["k_samples_deep"])
        print(f"  deep pass@k on MATH-test slice (K={k_deep}, {eval_cfg['deep_subset_size']} problems)")
        df = load_benchmark("math_test").head(int(eval_cfg["deep_subset_size"])).reset_index(drop=True)
        prompts, seeds, meta = _build_prompts(df, k_deep, bench="math_test")
        deep_rollouts = _score_rollouts(meta, _generate_k(llm, prompts, seeds, temperature, max_tokens), family=cond_family)
        records = _rollouts_to_records(deep_rollouts)
        deep = {"k": k_deep, "n_problems": len(records)}
        for k_val in [1, 4, 16, 64, k_deep]:
            deep[f"pass_at_{k_val}"] = M.pass_at_k_corpus(records, k_val)
        (out_dir / f"{stem}_math_test_passatk.json").write_text(
            json.dumps(deep, indent=2)
        )

    if args.diff_baseline and "math_test" in all_rollouts_by_bench:
        print("  --diff-baseline: tearing down trained engine before loading base")
        # vLLM holds ~gpu_memory_utilization * VRAM. On 11 GB Turing this fills the
        # card, so the base engine OOMs on init unless we explicitly free the trained one.
        import gc
        import torch
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("  --diff-baseline: running base model for delta report")
        _diff_baseline(
            model_path=model_path,
            rollout_device=rollout_device,
            max_tokens=max_tokens,
            k_diff=4,
            slice_size=100,
            trained_rollouts=all_rollouts_by_bench["math_test"],
            out_path=out_dir / f"{stem}_diff.json",
            dtype=resolved_dtype,
            family=cond_family,
        )

    print(f"OK eval {args.model}_{args.condition} -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
