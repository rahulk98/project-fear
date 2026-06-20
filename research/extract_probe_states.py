#!/usr/bin/env python3
"""Memory-lean all-sample hidden-state extraction for the probe-as-verifier study.

Isolated research variant of scripts/extract_hidden_states.py. The stock script
calls the FULL causal-LM forward, which materialises lm_head logits over the whole
sequence (vocab 151936 -> a 5-6 GB allocation per batch) even though probing only
needs hidden states. On a 32 GB V100 with float32 + all layers that OOMs at the
first long-sequence batch.

This version calls the BASE transformer (model.model) with output_hidden_states,
skipping the sequence-wide lm_head entirely (~6 GB and ~half the FLOPs saved), and
applies lm_head to ONLY the final token to recover the exact same logit_max_prob /
logit_entropy baseline the stock path stored. Everything else -- prompt
reconstruction, left-padding, last-real-token gather, label codes, NPZ format -- is
reused verbatim from asym_rewards so the output is a drop-in for probe_bestofn.py.

No live-pipeline files are modified; this only imports them.

Usage:
  research/extract_probe_states.py --benchmark math_test \\
      --rollout /tmp/s0_sub_math_test.jsonl --condition S0bo16 \\
      --dtype float32 --device cuda:0 --batch-size 8 --out-dir outputs/probing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asym_rewards.paths import MODEL_DIRS  # noqa: E402
from asym_rewards import probing as P  # noqa: E402

# Prompt templates copied verbatim from scripts/extract_hidden_states.py so the
# reconstructed prompt string matches the one used at rollout-generation time.
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
    """Reconstruct pid -> (prompt_text, gold, scorer). Mirrors the stock script."""
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


def extract_no_lmhead(model, tokenizer, items, batch_size, max_len):
    """Same contract as probing.extract_last_token_states, but routes the forward
    through model.model (no sequence-wide lm_head) and applies lm_head only to the
    last token for the logit-confidence signals. Returns (H, y, meta)."""
    import torch

    assert tokenizer.padding_side == "left", "need left padding for last-token gather"
    base = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)
    if base is None or lm_head is None:
        raise AttributeError("model lacks .model / .lm_head (expected a Qwen2ForCausalLM)")
    device = next(model.parameters()).device

    all_H: list[np.ndarray] = []
    all_y: list[int] = []
    all_meta: list[dict] = []

    n = len(items)
    for start in range(0, n, batch_size):
        batch_items = items[start: start + batch_size]
        texts = [it["prompt_text"] + it["completion"] for it in batch_items]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len).to(device)
        with torch.no_grad():
            out = base(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                       output_hidden_states=True, use_cache=False, return_dict=True)
            hs = out.hidden_states                          # tuple (L+1) of (B,S,d)
            stacked = torch.stack(hs, dim=0)[:, :, -1, :]   # (L+1, B, d)
            last = stacked.permute(1, 0, 2)                 # (B, L+1, d)
            # exact-match logit signals: lm_head on the final post-norm token only
            last_tok = out.last_hidden_state[:, -1, :]      # (B, d)
            logits = lm_head(last_tok).float()              # (B, vocab)
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1)
            max_prob = probs.max(dim=-1).values

        last_np = last.float().cpu().numpy()
        entropy_np = entropy.cpu().numpy()
        max_prob_np = max_prob.cpu().numpy()
        for i, it in enumerate(batch_items):
            all_H.append(last_np[i])
            all_y.append(P.LABEL_MAP.get(it["label"], -1))
            m = dict(it)
            m["logit_entropy"] = float(entropy_np[i])
            m["logit_max_prob"] = float(max_prob_np[i])
            all_meta.append(m)
        if (start // batch_size) % 50 == 0:
            print(f"    {min(start + batch_size, n)}/{n} done", flush=True)

    H = np.stack(all_H, axis=0)
    y = np.array(all_y, dtype=np.int32)
    finite = np.isfinite(H.reshape(H.shape[0], -1)).all(axis=1)
    n_bad = int((~finite).sum())
    if n_bad:
        print(f"  WARNING: {n_bad}/{H.shape[0]} non-finite rows dropped", flush=True)
        if not finite.any():
            raise ValueError("all hidden states non-finite -- refusing to save")
        H, y = H[finite], y[finite]
        all_meta = [m for m, ok in zip(all_meta, finite) if ok]
    return H, y, all_meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2_5_1_5b", choices=list(MODEL_DIRS))
    ap.add_argument("--condition", default="S0bo16")
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "probing"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  reconstructing prompts for '{args.benchmark}' ...", flush=True)
    prompts_by_pid = _prompts_by_pid(args.benchmark)
    items = P.iter_rollout_records(Path(args.rollout), prompts_by_pid, max_per_label=None)
    counts: dict[str, int] = {}
    for it in items:
        counts[it["label"]] = counts.get(it["label"], 0) + 1
    print(f"  {len(items)} items after join: {counts}", flush=True)
    if not items:
        print("  ERROR: no items matched rollout<->prompt table", flush=True)
        return 1

    print(f"  loading {args.model} (dtype={args.dtype}, device={args.device}) ...", flush=True)
    model, tokenizer = P.load_probing_model(model_name=args.model, adapter_path=None,
                                            dtype=args.dtype, device=args.device)
    print("  extracting (base-model forward, no sequence lm_head) ...", flush=True)
    H, y, meta = extract_no_lmhead(model, tokenizer, items, args.batch_size, args.max_len)
    print(f"  H shape: {H.shape}", flush=True)

    n_layers = H.shape[1]
    layer_names = ["embed"] + [f"layer_{i}" for i in range(n_layers - 1)]
    stem = f"{args.model}_{args.condition}_{args.dtype}_{args.benchmark}_hidden"
    out_path = out_dir / f"{stem}.npz"
    config = {"model": args.model, "condition": args.condition, "dtype": args.dtype,
              "benchmark": args.benchmark, "batch_size": args.batch_size,
              "max_len": args.max_len, "all_layers": True, "forward": "base_model_no_lmhead"}
    P.save_probing_npz(out_path=out_path, H=H, y=y, meta=meta, layer_names=layer_names,
                       config=config, source_rollout=str(Path(args.rollout).resolve()))
    print(f"  saved -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
