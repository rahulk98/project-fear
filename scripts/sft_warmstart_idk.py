#!/usr/bin/env python
"""IDK-channel SFT warm-start for FD1xsil GRPO (P6).

Problem: Qwen-2.5-1.5B emits \\boxed{IDK} essentially never from the base model
(observed ~1/80k on MATH prompts). The FD1xsil GRPO condition needs a non-zero
IDK signal to amplify; with zero emissions the wise_idk and wasteful_idk reward
channels are never activated and the model has no gradient path to learn IDK.

Solution: a short SFT pre-pass that installs a low-but-nonzero IDK probability
(target 1-10%) using a curated bilingual exemplar set. After SFT the adapter is
consumed by a GRPO run with --condition FD1xsil which CONTINUES training from the
warm-started LoRA (same base, same LoRA rank/alpha, same target modules).

How the FD1xsil GRPO run consumes this adapter
-----------------------------------------------
The warm-started adapter is saved to --out-dir (e.g. outputs/sft/idk_warmstart).
Pass it to train_grpo.py via --warm-start-adapter:

    python scripts/train_grpo.py \\
        --model qwen2_5_1_5b \\
        --condition FD1xsil \\
        --idk-prompt \\
        --warm-start-adapter outputs/sft/idk_warmstart \\
        --out-root outputs/grpo

train_grpo loads this directory as a TRAINABLE PeftModel at model-init
(PeftModel.from_pretrained(..., is_trainable=True)) and passes peft_config=None so
the trainer does NOT re-wrap with a fresh random adapter; GRPO then continues the
SAME LoRA from step 0 with a fresh optimizer. Do NOT use --resume-from-checkpoint:
that path expects a full TRL checkpoint (trainer_state.json + optimizer state),
which a bare adapter dir does not have, and it raises FileNotFoundError. The LoRA
config must match (r=16, alpha=32, same target_modules); this script hardcodes the
same LoRA config as configs/training.yaml to guarantee compatibility.

IDK format specification (must match FD1xsil reward detection)
---------------------------------------------------------------
reward.py detects IDK via is_idk_marker(extracted_boxed_content):
    extracted = last \\boxed{...} contents, stripped/lowercased/rstrip('.')
    IDK if extracted in {"idk", "i.d.k", "?"}
So the target completion for IDK exemplars must end with \\boxed{IDK}.
Normal (solvable) exemplars end with \\boxed{<numeric answer>}.
The FD1xsil extend_silent_forced_decode=True flag means silent samples (no box)
are bifurcated by forced decode -- the SFT warm-start does NOT need to teach the
model to avoid silence specifically; that is the GRPO's job. We only need to seed
the explicit IDK token.

Exemplar recipe (N6 compliance: no eval benchmark leakage)
-----------------------------------------------------------
- Solvable exemplars: drawn from MATH-train levels 1-3 (easy/medium problems
  excluded from the GRPO training set, which uses levels 4-5 only). These teach
  the model to COMMIT when it can solve the problem.
- Unanswerable / trick exemplars: entirely synthetic, not drawn from any eval
  split (MATH-test, GSM8K-test, AIME, MMLU, TruthfulQA are all off-limits).
  The synthetic set uses deliberately ill-posed, contradictory, or context-free
  prompts where IDK is the CORRECT epistemic response.
- Mix: ~80% solvable (normal answer), ~20% unanswerable (IDK answer). This ratio
  seeds IDK without collapsing the model to always-IDK.

Post-SFT gate (hard constraint)
--------------------------------
After SFT, before handing the adapter to GRPO, this script measures IDK emission
on TWO disjoint held-out probe sets SEPARATELY (never blended -- they are
opposite criteria):
    - SOLVABLE probes:    IDK rate <= SOLVABLE_IDK_MAX   (collapse ceiling)
    - UNANSWERABLE probes: IDK rate >= UNANSWERABLE_IDK_MIN (reachability floor)
The gate passes iff BOTH hold (low abstention on answerable problems AND a
reachable IDK channel on unanswerable ones). A single blended rate cannot
express this: a successful warm-start (e.g. ~2% solvable, ~50% unanswerable)
blends to ~0.16 and would fail a naive [0.01, 0.10] band. On failure the script
exits non-zero with which side failed; the GRPO launcher checks the exit code.

Usage
-----
    python scripts/sft_warmstart_idk.py \\
        --model qwen2_5_1_5b \\
        --out-dir outputs/sft/idk_warmstart \\
        --n-exemplars 120 \\
        --epochs 2 \\
        --device cuda:0

    # Then launch FD1xsil GRPO:
    python scripts/train_grpo.py \\
        --model qwen2_5_1_5b \\
        --condition FD1xsil \\
        --idk-prompt \\
        --warm-start-adapter outputs/sft/idk_warmstart
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Repo root so src/ is importable on the server
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# LoRA config -- must match configs/training.yaml exactly
# (r=16, alpha=32, dropout=0.05, target_modules same as GRPO run)
# ---------------------------------------------------------------------------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ---------------------------------------------------------------------------
# IDK prompt template: matches grpo.py PROMPT_TEMPLATE_IDK
# (the FD1xsil run uses --idk-prompt, so SFT must use the same instruction)
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE_IDK = (
    "Solve the following problem step by step. "
    "Put your final answer in \\boxed{{}}. "
    "If you cannot solve the problem, write \\boxed{{IDK}} as your answer.\n\n"
    "Problem: {problem}"
)

# Two-sided post-SFT gate thresholds (measured SEPARATELY, never blended):
#  - SOLVABLE set: the model must NOT collapse to abstaining on answerable
#    problems -> IDK rate must stay at/below SOLVABLE_IDK_MAX.
#  - UNANSWERABLE set: the IDK channel must be REACHABLE -> IDK rate must clear
#    UNANSWERABLE_IDK_MIN. A successful warm-start has LOW solvable + HIGH
#    unanswerable IDK; a single blended threshold cannot express that.
SOLVABLE_IDK_MAX = 0.10
UNANSWERABLE_IDK_MIN = 0.10

# ---------------------------------------------------------------------------
# Synthetic unanswerable/trick exemplar set (N6: no eval benchmark leakage)
# These prompts are deliberately ill-posed, contradictory, or context-free so
# that \\boxed{IDK} is the epistemically correct response.
# ---------------------------------------------------------------------------
SYNTHETIC_IDK_PROMPTS = [
    # Genuinely undefined / context-free
    "What is the value of x?",
    "Compute f(3).",
    "Find the root of the polynomial.",
    "What is the sum?",
    "Evaluate the integral.",
    # Self-contradictory
    "Find a real number x such that x > x + 1.",
    "Find integers a, b > 0 with a + b = 5 and a + b = 7.",
    "A triangle has angles 70, 80, and 90 degrees. Find the third angle.",
    "A fair coin is flipped. The probability of heads is 0.3 and the probability of tails is 0.8. Find P(heads).",
    "Let S be the set of all sets that do not contain themselves. Is S in S?",
    # Division by zero / ill-defined
    "Compute 5 / 0.",
    "What is the largest prime number?",
    "What is the square root of -9 in the real numbers expressed as a real number?",
    # Underspecified (missing data)
    "A rectangle has area 24. What is its perimeter?",
    "A car travels at average speed v. How long does the trip take?",
    "Given that P(A) = 0.4, find P(A and B).",
    "The sequence has a sum of 100. What is the 10th term?",
    # Contradictory constraints
    "Find all positive integers n such that n < 2 and n > 3.",
    "A function f is both strictly increasing and strictly decreasing everywhere. Give an example.",
    "Find a matrix A such that A^2 = -I (where I is the 2x2 identity matrix) with all real entries.",
]

# Completion template for IDK exemplars.
# NOTE: this string is used RAW as the SFT completion (never passed through
# .format()), so braces must be SINGLE (literal text).  The PROMPT_TEMPLATE_IDK
# above is passed through .format(problem=...) and therefore needs double-braces
# to produce a single brace in the formatted output -- keep them as-is there.
IDK_COMPLETION_TEMPLATE = (
    "This problem is either ill-posed, self-contradictory, or missing necessary "
    "information to determine a unique numerical answer. "
    "I cannot solve this problem as stated. "
    "\\boxed{IDK}"
)


def _make_synthetic_idk_exemplars() -> list[dict[str, str]]:
    """Return a list of {prompt, completion} dicts for unanswerable problems."""
    return [
        {
            "prompt": PROMPT_TEMPLATE_IDK.format(problem=p),
            "completion": IDK_COMPLETION_TEMPLATE,
        }
        for p in SYNTHETIC_IDK_PROMPTS
    ]


def _load_solvable_exemplars(n: int) -> list[dict[str, str]]:
    """Load n solvable exemplars from MATH-train levels 1-3 (non-GRPO split).

    The GRPO training set uses MATH levels 4-5 only (hard problems). This
    function reads from levels 1-3, which are NEVER used in GRPO training, so
    there is no data leakage into the GRPO reward distribution.

    Falls back to a small hardcoded set if the MATH-train benchmark is not yet
    downloaded to the server (e.g., in offline unit-test mode).
    """
    try:
        from asym_rewards.data import load_benchmark  # type: ignore[import]
        df = load_benchmark("math_train")
        if "level" in df.columns:
            df = df[df["level"].isin([1, 2, 3])].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("no level 1-3 samples found")
        df = df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)
        exemplars = []
        for _, row in df.iterrows():
            prompt = PROMPT_TEMPLATE_IDK.format(problem=str(row["prompt"]))
            answer = str(row["answer"]).strip()
            completion = (
                f"Let me solve this step by step.\n"
                f"After careful calculation, the answer is {answer}.\n"
                f"\\boxed{{{answer}}}"
            )
            exemplars.append({"prompt": prompt, "completion": completion})
        return exemplars
    except Exception as exc:
        print(f"[sft_warmstart] load_benchmark fallback: {exc}")
        # Minimal hardcoded set so the script does not crash in offline mode
        fallback = [
            {"prompt": PROMPT_TEMPLATE_IDK.format(problem="What is 2 + 2?"),
             "completion": "2 + 2 = 4. \\boxed{4}"},
            {"prompt": PROMPT_TEMPLATE_IDK.format(problem="What is 3 * 5?"),
             "completion": "3 * 5 = 15. \\boxed{15}"},
            {"prompt": PROMPT_TEMPLATE_IDK.format(problem="What is 10 - 7?"),
             "completion": "10 - 7 = 3. \\boxed{3}"},
            {"prompt": PROMPT_TEMPLATE_IDK.format(problem="Simplify: 6/2."),
             "completion": "6/2 = 3. \\boxed{3}"},
        ]
        return fallback[:n]


def build_exemplar_dataset(n_exemplars: int) -> list[dict[str, str]]:
    """Compose the SFT exemplar set: ~80% solvable, ~20% IDK.

    The split is set so the model learns WHEN to IDK (not always-IDK).
    """
    n_idk = min(len(SYNTHETIC_IDK_PROMPTS), max(1, round(0.20 * n_exemplars)))
    n_solvable = n_exemplars - n_idk

    idk_all = _make_synthetic_idk_exemplars()
    # Truncate or repeat if n_idk > available synthetic prompts
    idk_exemplars = (idk_all * math.ceil(n_idk / len(idk_all)))[:n_idk]

    solvable_exemplars = _load_solvable_exemplars(n_solvable)

    # Interleave so IDK samples are distributed across the epoch
    exemplars: list[dict[str, str]] = []
    # 4 solvable then 1 IDK pattern
    i_sol = 0
    i_idk = 0
    while i_sol < len(solvable_exemplars) or i_idk < len(idk_exemplars):
        for _ in range(4):
            if i_sol < len(solvable_exemplars):
                exemplars.append(solvable_exemplars[i_sol])
                i_sol += 1
        if i_idk < len(idk_exemplars):
            exemplars.append(idk_exemplars[i_idk])
            i_idk += 1

    return exemplars


def _is_idk_completion(text: str) -> bool:
    """Mirror reward.py is_idk_marker logic on a full completion string.

    reward.py extracts the LAST \\boxed{} and checks if its content (after
    strip/lower/rstrip('.')) is in {idk, i.d.k, ?}. We replicate that here
    for the post-SFT gate.
    """
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return False
    i = idx + len("\\boxed{")
    depth = 1
    start = i
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return False
    extracted = text[start : i - 1].strip().lower().rstrip(".").strip()
    return extracted in {"idk", "i.d.k", "?"}


def measure_idk_rate(
    model,
    tokenizer,
    probe_prompts: list[str],
    device: str,
    max_new_tokens: int = 256,
) -> float:
    """Fraction of probe_prompts whose sampled completion emits \\boxed{IDK}.

    A single, semantics-free measurement. The pass/fail policy lives in the
    caller, which measures the SOLVABLE set (collapse check: should be LOW) and
    the UNANSWERABLE set (reachability check: should clear a floor) SEPARATELY.
    Blending the two (as a single 70/30 mix) conflates two opposite criteria:
    a successful warm-start has high IDK on unanswerable + low on solvable, and
    the blended rate could exceed any single ceiling while being correct.
    """
    import torch

    model.eval()
    n_idk = 0
    with torch.no_grad():
        for prompt in probe_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            if _is_idk_completion(generated):
                n_idk += 1

    model.train()
    return n_idk / max(len(probe_prompts), 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SFT warm-start to seed IDK channel for FD1xsil GRPO."
    )
    p.add_argument(
        "--model", default="qwen2_5_1_5b",
        help="Model key in grpo.py HF_BASE_MODEL_IDS (default: qwen2_5_1_5b)",
    )
    p.add_argument(
        "--out-dir", required=True,
        help="Directory to save the LoRA adapter. Pass to train_grpo --warm-start-adapter.",
    )
    p.add_argument(
        "--n-exemplars", type=int, default=120,
        help="Total SFT exemplar count (~80%% solvable, ~20%% IDK). Default 120.",
    )
    p.add_argument(
        "--epochs", type=int, default=2,
        help="Number of SFT training epochs. Default 2.",
    )
    p.add_argument(
        "--batch-size", type=int, default=4,
        help="Per-device train batch size. Default 4 (A100). Use 1-2 on an 11GB "
             "2080 Ti (batch 4 fp16 needs ~10.4GB and OOMs).",
    )
    p.add_argument(
        "--device", default="cuda:0",
        help="Device for training (e.g. cuda:0). Default cuda:0.",
    )
    p.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
        help="Training dtype. Default bfloat16 (A100). Use float16 on Turing/Volta "
             "(2080 Ti / V100 have no bf16 tensor cores -> bf16 runs CPU-bound).",
    )
    p.add_argument(
        "--lr", type=float, default=2e-4,
        help="SFT learning rate. Default 2e-4 (higher than GRPO: short warm-start).",
    )
    p.add_argument(
        "--max-length", type=int, default=512,
        help="Max token length (prompt+completion). Default 512.",
    )
    p.add_argument(
        "--n-gate", type=int, default=50,
        help="Probe completions per set (solvable + unanswerable) for the gate. Default 50 each.",
    )
    p.add_argument(
        "--solvable-idk-max", type=float, default=SOLVABLE_IDK_MAX,
        help=f"Max IDK rate on SOLVABLE probes (collapse ceiling; default {SOLVABLE_IDK_MAX}).",
    )
    p.add_argument(
        "--unanswerable-idk-min", type=float, default=UNANSWERABLE_IDK_MIN,
        help=f"Min IDK rate on UNANSWERABLE probes (reachability floor; default {UNANSWERABLE_IDK_MIN}).",
    )
    p.add_argument(
        "--skip-gate", action="store_true",
        help="Skip the post-SFT gate check (for debugging only).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Lazy imports so py_compile passes locally (no torch on laptop)
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from asym_rewards.grpo import HF_BASE_MODEL_IDS  # type: ignore[import]

    # Resolve model path
    hf_model_id = HF_BASE_MODEL_IDS.get(args.model)
    if hf_model_id is None:
        # Try local MODEL_DIRS
        try:
            from asym_rewards.paths import MODEL_DIRS  # type: ignore[import]
            model_path = MODEL_DIRS.get(args.model, args.model)
        except ImportError:
            model_path = args.model
    else:
        # Check if a local copy exists first (server-side downloaded model)
        try:
            from asym_rewards.paths import MODEL_DIRS  # type: ignore[import]
            model_path = MODEL_DIRS.get(args.model, hf_model_id)
        except ImportError:
            model_path = hf_model_id

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sft_warmstart] model      = {model_path}")
    print(f"[sft_warmstart] out_dir    = {out_dir}")
    print(f"[sft_warmstart] n_exemplars= {args.n_exemplars}")
    print(f"[sft_warmstart] epochs     = {args.epochs}")
    print(f"[sft_warmstart] device     = {args.device}")
    print(f"[sft_warmstart] lr         = {args.lr}")

    # Build exemplar dataset
    print("[sft_warmstart] building exemplar dataset...")
    exemplars = build_exemplar_dataset(args.n_exemplars)
    n_idk_actual = sum(1 for e in exemplars if _is_idk_completion(e["completion"]))
    n_solvable_actual = len(exemplars) - n_idk_actual
    print(
        f"[sft_warmstart] exemplars: {len(exemplars)} total "
        f"({n_solvable_actual} solvable, {n_idk_actual} IDK = "
        f"{100*n_idk_actual/max(len(exemplars),1):.1f}%)"
    )

    # Format as single text field (SFTTrainer dataset_text_field mode)
    def _fmt(e: dict[str, str]) -> str:
        return e["prompt"] + "\n" + e["completion"]

    hf_dataset = Dataset.from_dict({"text": [_fmt(e) for e in exemplars]})

    # Load tokenizer + base model
    print(f"[sft_warmstart] loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[sft_warmstart] loading base model from {model_path} (dtype={args.dtype})...")
    _dtype_map = {
        "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32,
    }
    torch_dtype = _dtype_map[args.dtype]
    # device_map must be a strategy name ("auto") or an explicit {"": device}
    # mapping -- a bare "cuda:0" string is rejected by transformers.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map={"": args.device},
    )

    # LoRA config -- MUST match configs/training.yaml (r=16, alpha=32, same modules)
    # so the GRPO run can warm-start from this adapter without a config mismatch.
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # SFT training arguments. trl==0.15.2: dataset_text_field + max_seq_length
    # live on SFTConfig (not SFTTrainer kwargs), and the tokenizer is passed as
    # processing_class (transformers renamed tokenizer -> processing_class).
    # save_strategy="no": the GRPO run later loads this adapter via
    # --warm-start-adapter (PeftModel.from_pretrained), which expects the adapter
    # files directly in out_dir, NOT inside a checkpoint-N subdirectory that
    # save_strategy="epoch" would create. We save the adapter explicitly after
    # trainer.train() below.
    train_args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        bf16=(args.dtype == "bfloat16"),
        fp16=(args.dtype == "float16"),
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=42,
        dataloader_drop_last=True,
        dataset_text_field="text",
        max_seq_length=args.max_length,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=hf_dataset,
        processing_class=tokenizer,
    )

    print("[sft_warmstart] starting SFT training...")
    trainer.train()
    print("[sft_warmstart] SFT training complete.")

    # Save the adapter (NOT merged weights -- the GRPO run continues LoRA training)
    print(f"[sft_warmstart] saving LoRA adapter to {out_dir}...")
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Write a meta file documenting the warm-start so the GRPO run can verify it
    meta = {
        "warmstart_type": "sft_idk",
        "base_model": str(model_path),
        "n_exemplars": len(exemplars),
        "n_idk": n_idk_actual,
        "n_solvable": n_solvable_actual,
        "idk_fraction": n_idk_actual / max(len(exemplars), 1),
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_target_modules": LORA_TARGET_MODULES,
        "grpo_continue_command": (
            f"python scripts/train_grpo.py "
            f"--model {args.model} "
            f"--condition FD1xsil "
            f"--idk-prompt "
            f"--warm-start-adapter {out_dir}"
        ),
    }
    (out_dir / "sft_warmstart_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[sft_warmstart] meta written to {out_dir}/sft_warmstart_meta.json")

    # Post-SFT gate
    if args.skip_gate:
        print("[sft_warmstart] gate skipped (--skip-gate). CAUTION: IDK rate unchecked.")
        return 0

    print(f"[sft_warmstart] running post-SFT gate on {args.n_gate} solvable + {args.n_gate} unanswerable probes...")
    # Two DISJOINT probe sets, measured SEPARATELY (held-out, not in training).
    gate_solvable = [
        PROMPT_TEMPLATE_IDK.format(problem="What is 12 * 7?"),
        PROMPT_TEMPLATE_IDK.format(problem="Solve for x: 2x + 4 = 10."),
        PROMPT_TEMPLATE_IDK.format(problem="What is the area of a circle with radius 3?"),
        PROMPT_TEMPLATE_IDK.format(problem="Simplify: (x^2 - 4) / (x - 2)."),
        PROMPT_TEMPLATE_IDK.format(problem="What is 15% of 200?"),
    ]
    gate_unanswerable = [
        PROMPT_TEMPLATE_IDK.format(problem="What is the value of y?"),
        PROMPT_TEMPLATE_IDK.format(problem="Find x such that x > x."),
        PROMPT_TEMPLATE_IDK.format(problem="A circle has area 10 and area 15. Find the radius."),
    ]
    solvable_probes = (gate_solvable * math.ceil(args.n_gate / len(gate_solvable)))[: args.n_gate]
    unanswerable_probes = (gate_unanswerable * math.ceil(args.n_gate / len(gate_unanswerable)))[: args.n_gate]

    solvable_idk = measure_idk_rate(model, tokenizer, solvable_probes, args.device)
    unanswerable_idk = measure_idk_rate(model, tokenizer, unanswerable_probes, args.device)

    # Collapse check (solvable must stay low) AND reachability check (unanswerable
    # must clear the floor). Both must hold -- they are opposite criteria, so a
    # single blended rate cannot express a successful warm-start.
    collapse_ok = solvable_idk <= args.solvable_idk_max
    reachable_ok = unanswerable_idk >= args.unanswerable_idk_min
    gate_passed = collapse_ok and reachable_ok

    print(
        f"[sft_warmstart] gate: solvable IDK = {solvable_idk:.3f} (<= {args.solvable_idk_max:.2f}? "
        f"{'OK' if collapse_ok else 'FAIL-collapse'}); "
        f"unanswerable IDK = {unanswerable_idk:.3f} (>= {args.unanswerable_idk_min:.2f}? "
        f"{'OK' if reachable_ok else 'FAIL-unreachable'})"
    )

    if gate_passed:
        print("[sft_warmstart] GATE PASSED. Adapter is ready for FD1xsil GRPO.")
        print(f"[sft_warmstart] Launch command:\n  {meta['grpo_continue_command']}")
        return 0
    if not reachable_ok:
        print(
            f"[sft_warmstart] GATE FAILED (unreachable): unanswerable IDK {unanswerable_idk:.3f} "
            f"< {args.unanswerable_idk_min} -- IDK channel not seeded. "
            "Try --epochs 3 or --n-exemplars 160 to increase IDK exposure."
        )
    if not collapse_ok:
        print(
            f"[sft_warmstart] GATE FAILED (collapse): solvable IDK {solvable_idk:.3f} "
            f"> {args.solvable_idk_max} -- model over-abstains on answerable problems. "
            "Try --epochs 1 or fewer synthetic IDK exemplars."
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
