#!/usr/bin/env python
"""Smoke test: load each training model in bf16 + one-token gen, and tokenize one sample per prepared dataset."""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asym_rewards.data import list_benchmarks, load_benchmark  # noqa: E402
from asym_rewards.models import load_model  # noqa: E402
from asym_rewards.paths import MODEL_DIRS  # noqa: E402

DEVICE = "cuda:0"
DTYPE = "bf16"


def smoke_model(name: str) -> None:
    print(f">>> Loading {name} ({DTYPE}) on {DEVICE}")
    model, tok = load_model(name, dtype=DTYPE, device=DEVICE)
    inputs = tok("What is 2+2?", return_tensors="pt").to(DEVICE)
    assert inputs["input_ids"].numel() >= 1, "tokenizer produced empty input"

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    assert out.shape[1] >= inputs["input_ids"].shape[1] + 1

    mem_gb = torch.cuda.memory_allocated(DEVICE) / 1e9
    print(f"    OK {name} on {DEVICE}, allocated={mem_gb:.2f} GB")

    del model, tok, inputs, out
    gc.collect()
    torch.cuda.empty_cache()


def smoke_benchmark(name: str, tokenizer) -> None:
    df = load_benchmark(name)
    assert len(df) > 0, f"benchmark {name} is empty"
    assert "prompt" in df.columns, f"benchmark {name} missing 'prompt' column"
    sample = df.iloc[0]
    n_tokens = len(tokenizer(str(sample["prompt"]), add_special_tokens=False)["input_ids"])
    print(f"    OK {name}: {len(df)} rows, sample prompt tokens={n_tokens}")


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1

    for name in MODEL_DIRS:
        smoke_model(name)

    print(">>> Tokenizing one sample per processed benchmark (using qwen tokenizer)")
    qwen_path = MODEL_DIRS["qwen2_5_3b"]
    if not qwen_path.exists():
        print(f"    SKIP benchmarks: {qwen_path} missing (run `make models` first)")
        return 0

    tok = AutoTokenizer.from_pretrained(str(qwen_path), use_fast=True)
    bench_names = list_benchmarks()
    if not bench_names:
        print("    SKIP benchmarks: none found (run `make data` first)")
        return 0

    for name in bench_names:
        try:
            smoke_benchmark(name, tok)
        except Exception as e:  # noqa: BLE001
            print(f"    FAIL {name}: {type(e).__name__}: {e}")

    print(">>> Smoke test done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
