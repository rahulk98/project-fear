from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .paths import MODEL_DIRS

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def load_model(name: str, dtype: str = "bf16", device: str = "cuda:0"):
    if name not in MODEL_DIRS:
        raise KeyError(f"Unknown model '{name}'. Known: {list(MODEL_DIRS)}")
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{dtype}'. Known: {list(_DTYPE_MAP)}")

    path = MODEL_DIRS[name]
    if not path.exists():
        raise FileNotFoundError(
            f"Model dir {path} not found. Run `make models` first."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True)
    # eager attention bypasses Triton kernels that assert Ampere-only MMA on Volta/Turing.
    # SDPA on V100/2080 Ti crashes with "mma -> mma layout conversion is only supported on Ampere".
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=_DTYPE_MAP[dtype],
        device_map={"": device},
        attn_implementation="eager",
    )
    return model, tokenizer
