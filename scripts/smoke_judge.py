#!/usr/bin/env python
"""Smoke test: use each trained adapter as a step-correctness judge on PRM800K.

Loads N rows from `data/processed/phase2_test.jsonl`, prompts each adapter
with (problem, proposed reasoning step) and asks for `\\boxed{YES}` or
`\\boxed{NO}`. Maps:

    \\boxed{YES}   -> predicted rating  +1
    \\boxed{NO}    -> predicted rating  -1
    no boxed      -> predicted rating   0 (abstain)

PRM800K human ratings are in {-1, 0, +1}. We sample only steps with a
definite +/- rating (skip rating=0 neutrals) to keep the binary task clean
on the gold side; the judge still has a 3-way output space, so abstention
shows up in its own bucket.

Writes per-condition agreement / abstention / YES-NO skew and AUROC metrics
to a JSON.

Usage (smoke, on one 2080 Ti shared with ollama):
    python scripts/smoke_judge.py \\
        --conditions S0,S1,S2,S3,S4,C1,C2 \\
        --n-rows 20 --gpu-memory-utilization 0.5 \\
        --rollout-device cuda:5

P1 AUROC flags:
    --k 8 --balanced --max-pos-per-row 2 --max-neg-per-row 2

P3 reference-augmented judge:
    --reference
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asym_rewards.judge import (  # noqa: E402
    JUDGE_PROMPT_TEMPLATE,
    judge_label,
    aggregate_votes,
    matched_subset,
    score_condition,
    matched_coverage_report,
)
from asym_rewards.paths import MODEL_DIRS, PROCESSED_DIR  # noqa: E402

# Back-compat alias: the label parser + prompt template now live in
# asym_rewards.judge (shared with the vc reward family). Kept so existing call
# sites in this module are unchanged.
_judge_label = judge_label

# Math-solving prompt used in P3 self-consistency reference generation.
# Mirrors eval_calibration.PROMPT_TEMPLATE verbatim to keep distributions aligned.
SOLVE_PROMPT = (
    "Solve the following problem step by step. "
    "Put your final answer in \\boxed{{}}.\n\n"
    "Problem: {problem}"
)

# Threshold below which a condition is considered degenerate for matched-coverage.
_MATCHED_TINY_N = 50



# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _iter_judge_examples(jsonl_path: Path, n_rows: int,
                         balanced: bool = False,
                         max_pos_per_row: int = 1,
                         max_neg_per_row: int = 1):
    """Yield (problem, step_text, gold_rating) triples.

    Default mode (balanced=False):
        One example per row - the FIRST step with a definite +/- rating. Rows
        where every step is rating=0 or has no completions are skipped.
        n_rows controls how many ROWS are scanned (not yielded).

    Balanced mode (balanced=True):
        Iterate ALL steps of each row. From each row yield up to
        max_pos_per_row positives (gold==+1) and max_neg_per_row negatives
        (gold==-1). n_rows still controls rows scanned.
    """
    yielded = 0
    rows_scanned = 0
    with jsonl_path.open() as f:
        for line in f:
            if rows_scanned >= n_rows:
                return
            row = json.loads(line)
            rows_scanned += 1
            question = row.get("question") or {}
            label = row.get("label") or {}
            problem = question.get("problem")
            steps = label.get("steps") or []
            if not problem or not steps:
                continue

            if not balanced:
                # Original first-step loop: unchanged behaviour.
                for step in steps:
                    completions = step.get("completions") or []
                    chosen = step.get("chosen_completion")
                    if chosen is not None and 0 <= chosen < len(completions):
                        comp = completions[chosen]
                    else:
                        comp = completions[0] if completions else None
                    if comp is None:
                        continue
                    rating = comp.get("rating")
                    text = comp.get("text") or ""
                    if rating in (-1, 1) and text.strip():
                        yield problem, text, int(rating)
                        yielded += 1
                        break
            else:
                # Balanced mode: collect all candidate steps, then yield up to
                # max_pos and max_neg from this row.
                pos_candidates: list[tuple[str, int]] = []
                neg_candidates: list[tuple[str, int]] = []
                for step in steps:
                    completions = step.get("completions") or []
                    chosen = step.get("chosen_completion")
                    if chosen is not None and 0 <= chosen < len(completions):
                        comp = completions[chosen]
                    else:
                        comp = completions[0] if completions else None
                    if comp is None:
                        continue
                    rating = comp.get("rating")
                    text = comp.get("text") or ""
                    if rating == 1 and text.strip():
                        pos_candidates.append((text, 1))
                    elif rating == -1 and text.strip():
                        neg_candidates.append((text, -1))
                for text, rat in pos_candidates[:max_pos_per_row]:
                    yield problem, text, rat
                    yielded += 1
                for text, rat in neg_candidates[:max_neg_per_row]:
                    yield problem, text, rat
                    yielded += 1


# ---------------------------------------------------------------------------
# Engine + generation
# ---------------------------------------------------------------------------

def _build_engine(model_path: str, adapter_path: str | None, rollout_device: str,
                  dtype: str, gpu_memory_utilization: float, max_model_len: int):
    """LoRA-merge + spin up a vLLM engine on a single CUDA device.

    Mirrors `eval_calibration._vllm_engine` but stays standalone so the smoke
    has no import-time dependency on the eval script. Adapter is merged into
    a cached `_merged_for_vllm_{dtype}` dir to match the eval cache convention.
    """
    from vllm import LLM

    device_idx = int(rollout_device.split(":")[-1]) if ":" in rollout_device else 0
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(device_idx))

    if adapter_path:
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

    return LLM(model=weights_path, dtype=dtype,
               gpu_memory_utilization=gpu_memory_utilization,
               max_model_len=max_model_len)


def _self_consistency_reference(llm, problem: str, k_sc: int = 8,
                                 temperature: float = 0.7,
                                 max_tokens: int = 512) -> str:
    """Generate K self-consistency samples for `problem`, extract the mode
    boxed answer, and return a compact reference string to inject into the
    judge prompt.

    This is the P3 helper. It makes a SEPARATE generate call (math-solving
    prompt, temperature>0) before the judge call and returns a one-line
    summary string of the form:

        "A self-consistency check of {agreement:.0%} of 8 attempts gave
        answer: {mode_answer}."

    If no boxed answer is extractable from any sample, returns an empty
    string (so callers can skip injection on failure).

    NOTE: this adds a second vLLM generate call per problem per condition.
    Latency scales with k_sc * max_tokens.
    """
    from vllm import SamplingParams
    # Import deferred to avoid circular-import hazard at module load.
    from asym_rewards.reward import extract_boxed_answer

    solve_prompt = SOLVE_PROMPT.format(problem=problem)
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=k_sc)
    outputs = llm.generate([solve_prompt], params, use_tqdm=False)
    texts = [o.text for o in outputs[0].outputs]

    # Tally boxed answers.
    answers: list[str] = []
    for t in texts:
        ans = extract_boxed_answer(t)
        if ans and ans.strip():
            answers.append(ans.strip())

    if not answers:
        return ""

    # Mode answer (first-seen wins ties, stable over list order).
    from collections import Counter
    counts = Counter(answers)
    mode_answer, mode_count = counts.most_common(1)[0]
    agreement = mode_count / len(texts)

    return (
        f"A self-consistency check of {agreement:.0%} of {k_sc} attempts "
        f"gave answer: {mode_answer}."
    )


# ---------------------------------------------------------------------------
# Condition runner
# ---------------------------------------------------------------------------

def _run_condition(prompts: list[str], gold_ratings: list[int],
                   adapter_path: str | None, model_path: str,
                   rollout_device: str, dtype: str,
                   gpu_memory_utilization: float, max_tokens: int,
                   max_model_len: int, temperature: float = 0.0,
                   k: int = 1,
                   reference: bool = False,
                   problems: list[str] | None = None) -> dict:
    """Run a single condition and return aggregated stats + per-row records.

    New parameters vs. the original:
        k: number of judge samples per prompt (SamplingParams.n=k). Votes are
           aggregated via majority; confidence is votes_yes / k.
        reference: when True, prepend a self-consistency reference block
           (P3 mode) to each judge prompt before generation. Requires
           `problems` to be provided (parallel to `prompts`).
        problems: original problem texts, needed for the P3 solve pass.
    """
    import numpy as np
    from vllm import SamplingParams

    llm = _build_engine(model_path, adapter_path, rollout_device, dtype,
                        gpu_memory_utilization, max_model_len)

    # P3: optionally inject self-consistency references into prompts.
    effective_prompts = list(prompts)
    if reference:
        if problems is None or len(problems) != len(prompts):
            raise ValueError("--reference requires problems list aligned with prompts")
        for i, problem in enumerate(problems):
            ref_text = _self_consistency_reference(llm, problem,
                                                   k_sc=8,
                                                   temperature=0.7,
                                                   max_tokens=512)
            if ref_text:
                # Inject before "Answer: " at the end of the judge prompt.
                # The template ends with "Answer: "; prepend the reference.
                original = effective_prompts[i]
                insertion = f"\n\nReference: {ref_text}\n"
                if original.endswith("Answer: "):
                    effective_prompts[i] = original[:-len("Answer: ")] + insertion + "Answer: "
                else:
                    effective_prompts[i] = original + insertion

    params = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=k)
    outputs = llm.generate(effective_prompts, params, use_tqdm=False)

    rows = []
    yes = no = abstain = agree = 0
    committed_indices: set[int] = set()

    for idx, (prompt, gold, out) in enumerate(zip(prompts, gold_ratings, outputs)):
        # Collect k votes.
        sample_texts = [s.text for s in out.outputs]
        votes = [_judge_label(t) for t in sample_texts]
        agg = aggregate_votes(votes)

        pred = agg["pred"]
        completion = sample_texts[0]  # first sample text for display

        rows.append({
            "gold": gold,
            "pred": pred,
            "completion": completion,
            "votes_yes": agg["votes_yes"],
            "votes_no": agg["votes_no"],
            "votes_abstain": agg["votes_abstain"],
            "conf_yes": agg["conf_yes"],
        })

        if pred == 1:
            yes += 1
        if pred == -1:
            no += 1
        if pred == 0:
            abstain += 1
        if pred != 0:
            committed_indices.add(idx)
        if pred == gold and gold != 0:
            agree += 1

    n = len(rows)
    attempted = n - abstain
    coverage = attempted / n if n else 0.0

    # Delegate AUROC scoring over committed rows to the canonical judge helper.
    scored = score_condition(rows)

    return {
        "n": n,
        "yes": yes, "no": no, "abstain": abstain,
        "attempt_rate": coverage,
        "coverage": coverage,
        "agreement_overall":      agree / n if n else 0.0,
        "agreement_on_attempted": agree / attempted if attempted else 0.0,
        "auroc_incorrect": scored["auroc_incorrect"],
        "auroc_ci_lo": scored["auroc_ci_lo"],
        "auroc_ci_hi": scored["auroc_ci_hi"],
        "f1_incorrect": scored["f1_incorrect"],
        "n_pos": scored["n_pos"],
        "n_neg": scored["n_neg"],
        "n_attempted": scored["n_attempted"],
        # committed_indices kept for matched-coverage computation in main().
        "_committed_indices": scored["committed_indices"],
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2_5_1_5b",
                   help="key into MODEL_DIRS for the base model weights")
    p.add_argument("--conditions", default="S0,S1,S2,S3,S4,C1,C2",
                   help="comma-separated condition codes; each loads "
                        "outputs/grpo/{model}_{cond}_s{seed}")
    p.add_argument("--seed", default="42",
                   help="adapter seed suffix (matches outputs/grpo/{model}_{cond}_s{seed})")
    p.add_argument("--n-rows", type=int, default=20)
    p.add_argument("--dataset", default=str(PROCESSED_DIR / "phase2_test.jsonl"))
    p.add_argument("--rollout-device", default="cuda:0")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                   help="vLLM upfront VRAM fraction; lower to fit alongside other tenants")
    p.add_argument("--max-tokens", type=int, default=16,
                   help="judge output is at most '\\boxed{NO}' + EOS, keep short")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="judge sampling temperature; 0.0 = greedy (default, backward-compatible). "
                        "Use 1.0 for a temperature-sensitivity judging run.")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--out-path",
                   default=str(REPO_ROOT / "outputs" / "phase5" / "smoke_judge.json"))
    # P1 AUROC flags (new; all default-off / default-1 so the existing smoke
    # invocation is behaviourally identical).
    p.add_argument("--k", type=int, default=1,
                   help="number of judge samples per prompt for vote aggregation "
                        "(k=1 = greedy, backward-compatible default)")
    p.add_argument("--balanced", action="store_true", default=False,
                   help="iterate ALL steps per row and yield up to "
                        "--max-pos-per-row / --max-neg-per-row per row "
                        "(default: first-step-only, unchanged)")
    p.add_argument("--max-pos-per-row", type=int, default=1,
                   help="max positive (correct) steps yielded per row in balanced mode")
    p.add_argument("--max-neg-per-row", type=int, default=1,
                   help="max negative (incorrect) steps yielded per row in balanced mode")
    p.add_argument("--attempt-floor", type=float, default=0.2,
                   help="conditions with coverage below this fraction are excluded "
                        "from the matched-coverage computation (treated as degenerate)")
    # P3 flag.
    p.add_argument("--reference", action="store_true", default=False,
                   help="P3: before judging each step, generate K=8 self-consistency "
                        "attempts at the underlying problem and inject the mode answer "
                        "as a reference block into the judge prompt")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples = list(_iter_judge_examples(
        Path(args.dataset), args.n_rows,
        balanced=args.balanced,
        max_pos_per_row=args.max_pos_per_row,
        max_neg_per_row=args.max_neg_per_row,
    ))
    if not args.balanced and len(examples) < args.n_rows:
        print(f"[warn] only found {len(examples)} usable rows in {args.dataset} "
              f"(needed {args.n_rows}); proceeding with what we have")

    problems = [p for p, _s, _g in examples]
    prompts = [JUDGE_PROMPT_TEMPLATE.format(problem=p, step=s) for p, s, _ in examples]
    gold_ratings = [g for _, _, g in examples]
    print(f"loaded {len(examples)} examples; +1 count={gold_ratings.count(1)}, "
          f"-1 count={gold_ratings.count(-1)}")

    model_path = str(MODEL_DIRS[args.model])
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    results: dict[str, dict] = {}

    # Per-condition committed index sets (needed for matched-coverage).
    per_condition_committed: dict[str, set[int]] = {}

    for cond in conditions:
        adapter_dir = REPO_ROOT / "outputs" / "grpo" / f"{args.model}_{cond}_s{args.seed}"
        if not adapter_dir.exists():
            print(f"[skip] {cond}: adapter not found at {adapter_dir}")
            continue
        print(f"\n=== {cond} ({adapter_dir.name}) ===")
        cond_result = _run_condition(
            prompts=prompts,
            gold_ratings=gold_ratings,
            adapter_path=str(adapter_dir),
            model_path=model_path,
            rollout_device=args.rollout_device,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            temperature=args.temperature,
            k=args.k,
            reference=args.reference,
            problems=problems,
        )

        # Stash committed indices before stripping the private key.
        committed_set = set(cond_result.pop("_committed_indices"))
        per_condition_committed[cond] = committed_set

        print(
            f"  attempt={cond_result['coverage']:.2f}  "
            f"agree_overall={cond_result['agreement_overall']:.2f}  "
            f"agree_attempted={cond_result['agreement_on_attempted']:.2f}  "
            f"YES={cond_result['yes']} NO={cond_result['no']} ABS={cond_result['abstain']}  "
            f"AUROC={cond_result['auroc_incorrect']:.3f}  "
            f"CI=[{cond_result['auroc_ci_lo']:.3f},{cond_result['auroc_ci_hi']:.3f}]"
        )
        results[cond] = cond_result

    # ------------------------------------------------------------------
    # Matched-coverage AUROC (N3): computed AFTER all conditions finish.
    # ------------------------------------------------------------------
    per_condition_rows = {cond: results[cond]["rows"] for cond in results}
    mc_result = matched_coverage_report(
        per_condition_committed=per_condition_committed,
        per_condition_rows=per_condition_rows,
        attempt_floor=args.attempt_floor,
        tiny_n=_MATCHED_TINY_N,
    )

    if mc_result["warning"]:
        print(mc_result["warning"])

    # Print matched-coverage summary for each included condition.
    for cond, mc_cond in mc_result["per_condition"].items():
        print(
            f"  matched-cov {cond}: AUROC={mc_cond['auroc_incorrect']:.3f} "
            f"CI=[{mc_cond['auroc_ci_lo']:.3f},{mc_cond['auroc_ci_hi']:.3f}] "
            f"n={mc_cond['n']}"
        )

    matched_coverage_results: dict[str, object] = {
        "n_matched": mc_result["n_matched"],
        "conditions_included": mc_result["conditions_included"],
        "conditions_excluded": mc_result["conditions_excluded"],
        "attempt_floor": args.attempt_floor,
        "fallback": mc_result["fallback"],
        "warning": mc_result["warning"],
        "per_condition": mc_result["per_condition"],
    }

    payload = {
        "config": {
            "model": args.model,
            "seed": args.seed,
            "n_rows": args.n_rows,
            "dataset": args.dataset,
            "dtype": args.dtype,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "k": args.k,
            "balanced": args.balanced,
            "max_pos_per_row": args.max_pos_per_row,
            "max_neg_per_row": args.max_neg_per_row,
            "reference": args.reference,
            "attempt_floor": args.attempt_floor,
        },
        "examples": [{"gold": g, "problem_prefix": p[:120], "step_prefix": s[:120]}
                     for p, s, g in examples],
        "results": results,
        "matched_coverage": matched_coverage_results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
