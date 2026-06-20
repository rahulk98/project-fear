#!/usr/bin/env python
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

MODELS = [
    ("Qwen/Qwen2.5-0.5B", "qwen2_5_0_5b"),
    ("Qwen/Qwen2.5-1.5B", "qwen2_5_1_5b"),
    ("Qwen/Qwen2.5-Math-1.5B", "qwen2_5_math_1_5b"),
    ("Qwen/Qwen2.5-3B", "qwen2_5_3b"),
    # Non-Qwen base for fear-v2 cross-architecture probe (2026-05-29).
    # Llama-3.2-1B is a gated HF repo; downloader needs an HF token with
    # meta-llama access. snapshot_download will raise if access is missing.
    ("meta-llama/Llama-3.2-1B", "llama3_2_1b"),
]


def folder_size(path: Path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return f"{total / 1e9:.2f} GB"


def main() -> int:
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "ERROR: HF_TOKEN missing. Create a .env in the repo root with HF_TOKEN=hf_...",
            file=sys.stderr,
        )
        return 1

    models_root = Path.home() / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    for repo_id, subdir in MODELS:
        local_dir = models_root / subdir
        print(f">>> Downloading {repo_id} -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            token=token,
            max_workers=4,
        )
        print(f"    size: {folder_size(local_dir)}")

        print(f">>> Sanity: loading tokenizer from {local_dir}")
        AutoTokenizer.from_pretrained(str(local_dir), use_fast=True)
        print(f"    tokenizer OK")

    print(">>> All models downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
