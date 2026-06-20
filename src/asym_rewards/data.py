from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .paths import PROCESSED_DIR


def load_benchmark(name: str) -> pd.DataFrame:
    parquet_path = PROCESSED_DIR / f"{name}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    jsonl_path = PROCESSED_DIR / f"{name}.jsonl"
    if jsonl_path.exists():
        with jsonl_path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return pd.DataFrame(rows)

    raise FileNotFoundError(
        f"No processed benchmark '{name}' at {parquet_path} or {jsonl_path}. "
        "Run `make data` first."
    )


def list_benchmarks() -> list[str]:
    names = []
    if PROCESSED_DIR.exists():
        for p in sorted(PROCESSED_DIR.iterdir()):
            if p.suffix in {".parquet", ".jsonl"}:
                names.append(p.stem)
    return names
