"""Experiment 2: internal-representation probing utilities.

Extracts last-token hidden states from a (possibly adapter-merged) causal LM
for rollout samples categorised as correct / incorrect / abstained, then saves
a compact NPZ + companion JSON for downstream probe_analysis.py.

GPU host only: load_probing_model and extract_last_token_states require torch
and transformers. iter_rollout_records and save_probing_npz are pure
numpy/stdlib and can run anywhere.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Label encoding (stable across all functions in this module)
# ---------------------------------------------------------------------------

LABEL_MAP: dict[str, int] = {
    "correct": 0,
    "incorrect": 1,
    "abstained": 2,
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_probing_model(
    model_name: str,
    adapter_path: str | None = None,
    dtype: str = "float16",
    device: str = "cuda:0",
):
    """Load a base or adapter-merged causal LM for hidden-state extraction.

    adapter_path: PEFT LoRA dir; merged via merge_and_unload() if supplied.
    Returns (model.eval(), tokenizer) with padding_side="left".
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .paths import MODEL_DIRS

    _dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in _dtype_map:
        raise ValueError(f"Unknown dtype '{dtype}'. Known: {list(_dtype_map)}")
    if model_name not in MODEL_DIRS:
        raise KeyError(f"Unknown model '{model_name}'. Known: {list(MODEL_DIRS)}")
    model_path = MODEL_DIRS[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Model dir {model_path} not found. Run `make models` first.")

    torch_dtype = _dtype_map[dtype]
    base = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        device_map={"": device},
        attn_implementation="eager",
    )
    if adapter_path is not None:
        from peft import PeftModel
        base = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return base.eval(), tokenizer


# ---------------------------------------------------------------------------
# Rollout record joining
# ---------------------------------------------------------------------------

def iter_rollout_records(
    rollout_path: str | Path,
    prompts_by_pid: dict[str, tuple[str, str, str]],
    max_per_label: int | None = None,
    labels: tuple[str, ...] = ("correct", "incorrect", "abstained"),
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Join a *_rollouts.jsonl with reconstructed prompts; derive and balance labels.

    Label derivation: "abstained" if abstained, "correct" if correct, else "incorrect".
    prompts_by_pid: pid -> (prompt_text, gold_answer, scorer) from eval_calibration.
    If max_per_label is set, each label bucket is subsampled independently (seeded).
    Returns list of {problem_id, sample_idx, label, prompt_text, completion}.
    """
    rollout_path = Path(rollout_path)
    rng = random.Random(seed)

    buckets: dict[str, list[dict[str, Any]]] = {lbl: [] for lbl in labels}
    with rollout_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid = str(row["problem_id"])
            if pid not in prompts_by_pid:
                continue
            prompt_text, _, _ = prompts_by_pid[pid]
            abstained = bool(row.get("abstained", False))
            correct = bool(row.get("correct", False))
            if abstained:
                lbl = "abstained"
            elif correct:
                lbl = "correct"
            else:
                lbl = "incorrect"
            if lbl not in buckets:
                continue
            buckets[lbl].append({
                "problem_id": pid,
                "sample_idx": int(row["sample_idx"]),
                "label": lbl,
                "prompt_text": prompt_text,
                "completion": str(row.get("completion", "")),
            })

    out: list[dict[str, Any]] = []
    for lbl in labels:
        items = buckets[lbl]
        if max_per_label is not None and len(items) > max_per_label:
            items = rng.sample(items, max_per_label)
        out.extend(items)
    return out


# ---------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------

def extract_last_token_states(
    model,
    tokenizer,
    items: list[dict[str, Any]],
    batch_size: int = 8,
    all_layers: bool = True,
    max_len: int = 2048,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Run forward passes; gather last-real-token hidden states per layer.

    LEFT-PADDING ensures column -1 is always the last real token in every
    mixed-length batch, making "hidden[:, -1, :]" the correct gather index.

    Returns (H, y, meta):
        H     float32 (N, L+1, d_model) if all_layers else (N, d_model).
        y     int32   (N,) label codes from LABEL_MAP.
        meta  list of item dicts augmented with logit_entropy, logit_max_prob.
    """
    import torch

    assert tokenizer.padding_side == "left", (
        "tokenizer.padding_side must be 'left' for last-token gather to be correct"
    )

    all_H: list[np.ndarray] = []
    all_y: list[int] = []
    all_meta: list[dict[str, Any]] = []

    n = len(items)
    for start in range(0, n, batch_size):
        batch_items = items[start: start + batch_size]
        texts = [it["prompt_text"] + it["completion"] for it in batch_items]

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(next(model.parameters()).device)

        with torch.no_grad():
            out = model(
                **enc,
                output_hidden_states=True,
                return_dict=True,
            )

        # hidden_states: tuple of (batch, seq, d), one per layer (L+1 entries).
        # Left-padding means column -1 is always the last real token.
        hidden_states = out.hidden_states  # tuple length L+1
        if all_layers:
            # Stack -> (L+1, batch, d) then transpose -> (batch, L+1, d)
            stacked = torch.stack(hidden_states, dim=0)  # (L+1, B, S, d)
            last = stacked[:, :, -1, :]                   # (L+1, B, d)
            last = last.permute(1, 0, 2)                  # (B, L+1, d)
        else:
            last = hidden_states[-1][:, -1, :]            # (B, d)

        # Logit-space confidence signals from the last token position.
        logits = out.logits[:, -1, :]                     # (B, vocab)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)        # (B,)
        max_prob = probs.max(dim=-1).values               # (B,)

        last_np = last.float().cpu().numpy()
        entropy_np = entropy.cpu().numpy()
        max_prob_np = max_prob.cpu().numpy()

        for i, it in enumerate(batch_items):
            all_H.append(last_np[i])
            all_y.append(LABEL_MAP.get(it["label"], -1))
            meta_entry = dict(it)
            meta_entry["logit_entropy"] = float(entropy_np[i])
            meta_entry["logit_max_prob"] = float(max_prob_np[i])
            all_meta.append(meta_entry)

    H = np.stack(all_H, axis=0)  # (N, L+1, d) or (N, d)
    y = np.array(all_y, dtype=np.int32)

    # Guard against non-finite hidden states. Models with massive activations
    # (e.g. Qwen2.5) overflow the float16 range *inside* the forward pass, so
    # the post-hoc .float() cast above cannot recover finite values -- the rows
    # are already NaN. Drop offending rows and report the count; abort loudly if
    # the whole tensor is non-finite (a float16 dtype misconfiguration), since a
    # silently-saved all-NaN NPZ only fails much later in probe_analysis.
    finite_rows = np.isfinite(H.reshape(H.shape[0], -1)).all(axis=1)
    n_bad = int((~finite_rows).sum())
    if n_bad:
        print(
            f"  WARNING: {n_bad}/{H.shape[0]} extracted rows are non-finite "
            f"(float16 overflow?) -- dropping them. Re-run with "
            f"--dtype bfloat16 or float32 to avoid this."
        )
        if not finite_rows.any():
            raise ValueError(
                "All extracted hidden states are non-finite -- refusing to save. "
                "Re-run with --dtype bfloat16 or float32 (float16 overflows on "
                "this model's activations)."
            )
        H = H[finite_rows]
        y = y[finite_rows]
        all_meta = [m for m, ok in zip(all_meta, finite_rows) if ok]
    return H, y, all_meta


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_probing_npz(
    out_path: str | Path,
    H: np.ndarray,
    y: np.ndarray,
    meta: list[dict[str, Any]],
    layer_names: list[str],
    config: dict[str, Any] | None = None,
    source_rollout: str = "",
) -> None:
    """Save H (float32) and y (int32) to a compressed NPZ + a .meta.json companion."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), H=H.astype(np.float32), y=y.astype(np.int32))

    label_counts: dict[str, int] = {}
    for code, name in {v: k for k, v in LABEL_MAP.items()}.items():
        label_counts[name] = int((y == code).sum())

    # Strip any non-JSON-serialisable values from meta before saving.
    serialisable_meta: list[dict[str, Any]] = []
    for m in meta:
        row: dict[str, Any] = {}
        for k, v in m.items():
            try:
                json.dumps(v)
                row[k] = v
            except (TypeError, ValueError):
                row[k] = str(v)
        serialisable_meta.append(row)

    companion = {
        "label_map": LABEL_MAP,
        "layer_names": layer_names,
        "per_label_counts": label_counts,
        "n_total": int(y.size),
        "h_shape": list(H.shape),
        "config": config or {},
        "source_rollout": source_rollout,
        "items": serialisable_meta,
    }
    meta_path = out_path.with_suffix("").with_suffix(".meta.json")
    meta_path.write_text(json.dumps(companion, indent=2))
