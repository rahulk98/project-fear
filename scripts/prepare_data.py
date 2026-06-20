#!/usr/bin/env python
"""Download and normalize all eval/training benchmarks to data/processed/*.parquet.

Adds PRM800K from openai/prm800k GitHub repo (not on HF) at the end.
"""
from __future__ import annotations

import gzip
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_DIR = REPO_ROOT / "data" / "raw"

MMLU_STEM_SUBJECTS = {
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning",
}


def _extract_boxed(text: str) -> str:
    """Pull the LAST \\boxed{...} content; handles balanced braces."""
    if not text:
        return ""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out).strip()


def _parse_level(level: str) -> Optional[int]:
    if not isinstance(level, str):
        return None
    m = re.search(r"(\d+)", level)
    return int(m.group(1)) if m else None


def _write(df: pd.DataFrame, name: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{name}.parquet"
    if out.exists():
        print(f"  skip {name} (exists, {len(pd.read_parquet(out))} rows)")
        return
    df.to_parquet(out, index=False)
    print(f"  wrote {name} ({len(df)} rows) -> {out.relative_to(REPO_ROOT)}")


def _try(name: str, fn: Callable[[], None]) -> None:
    print(f">>> {name}")
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print(f"  WARN: {name} failed: {type(e).__name__}: {e}")


def prep_math(split: str, out_name: str) -> None:
    # hendrycks/competition_math was removed from HF Hub; use the lighteval mirror,
    # which preserves the original 7500/5000 train/test split and schema.
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split=split)
    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "id": f"math_{split}_{i}",
            "prompt": ex["problem"],
            "answer": _extract_boxed(ex["solution"]) or ex["solution"],
            "level": _parse_level(ex.get("level")),
            "subject": ex.get("type"),
            "split": split,
            "source": "DigitalLearningGmbH/MATH-lighteval",
        })
    _write(pd.DataFrame(rows), out_name)


def prep_math_hard() -> None:
    ds = load_dataset("lighteval/MATH-Hard", split="test")
    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "id": f"math_hard_{i}",
            "prompt": ex["problem"],
            "answer": _extract_boxed(ex["solution"]) or ex["solution"],
            "level": _parse_level(ex.get("level")),
            "subject": ex.get("type"),
            "split": "test",
            "source": "lighteval/MATH-Hard",
        })
    _write(pd.DataFrame(rows), "math_hard")


def prep_math_500() -> None:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "id": f"math500_{i}",
            "prompt": ex["problem"],
            "answer": ex.get("answer") or _extract_boxed(ex.get("solution", "")),
            "level": _parse_level(ex.get("level")),
            "subject": ex.get("subject"),
            "split": "test",
            "source": "HuggingFaceH4/MATH-500",
        })
    _write(pd.DataFrame(rows), "math_500")


def prep_gsm8k(split: str, out_name: str) -> None:
    ds = load_dataset("gsm8k", "main", split=split)
    rows = []
    for i, ex in enumerate(ds):
        # gsm8k answer format: "...#### 42"
        ans = ex["answer"]
        m = re.search(r"####\s*(.+)$", ans)
        gold = m.group(1).strip() if m else ans.strip()
        rows.append({
            "id": f"gsm8k_{split}_{i}",
            "prompt": ex["question"],
            "answer": gold,
            "split": split,
            "source": "gsm8k",
        })
    _write(pd.DataFrame(rows), out_name)


def prep_aime_2024() -> None:
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "id": f"aime2024_{i}",
            "prompt": ex.get("Problem") or ex.get("problem"),
            "answer": str(ex.get("Answer") or ex.get("answer")),
            "split": "test",
            "source": "Maxwell-Jia/AIME_2024",
        })
    _write(pd.DataFrame(rows), "aime_2024")


def prep_aime_2025() -> None:
    # opencompass/AIME2025 ships two configs (AIME2025-I, AIME2025-II), 15 problems each.
    rows = []
    for config in ("AIME2025-I", "AIME2025-II"):
        ds = load_dataset("opencompass/AIME2025", config, split="test")
        for i, ex in enumerate(ds):
            rows.append({
                "id": f"aime2025_{config}_{i}",
                "prompt": ex.get("question") or ex.get("problem"),
                "answer": str(ex.get("answer")),
                "split": "test",
                "source": f"opencompass/AIME2025:{config}",
            })
    _write(pd.DataFrame(rows), "aime_2025")


def prep_olympiad_bench() -> None:
    # english math text-only subset
    ds = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
    rows = []
    for i, ex in enumerate(ds):
        prompt = ex.get("question") or ex.get("problem")
        ans = ex.get("final_answer") or ex.get("answer")
        if isinstance(ans, list):
            ans = " ; ".join(str(a) for a in ans)
        rows.append({
            "id": f"olympiad_{i}",
            "prompt": prompt,
            "answer": str(ans) if ans is not None else "",
            "subject": ex.get("subject"),
            "split": "test",
            "source": "Hothan/OlympiadBench",
        })
    _write(pd.DataFrame(rows), "olympiad_bench")


def prep_mmlu_stem() -> None:
    ds = load_dataset("cais/mmlu", "all", split="test")
    rows = []
    for i, ex in enumerate(ds):
        subj = ex.get("subject")
        if subj not in MMLU_STEM_SUBJECTS:
            continue
        choices = ex.get("choices") or []
        rows.append({
            "id": f"mmlu_{subj}_{i}",
            "prompt": ex["question"],
            "answer": str(ex.get("answer")),  # int index
            "choices": choices,
            "subject": subj,
            "split": "test",
            "source": "cais/mmlu",
        })
    _write(pd.DataFrame(rows), "mmlu_stem")


def prep_truthful_qa() -> None:
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    rows = []
    for i, ex in enumerate(ds):
        mc1 = ex.get("mc1_targets", {})
        rows.append({
            "id": f"truthfulqa_{i}",
            "prompt": ex["question"],
            "choices": mc1.get("choices", []),
            "answer": str(mc1.get("labels", [])),
            "split": "validation",
            "source": "truthful_qa",
        })
    _write(pd.DataFrame(rows), "truthful_qa_mc")


def prep_prm800k() -> None:
    print(">>> PRM800K (GitHub clone)")
    target = RAW_DIR / "prm800k"
    if not target.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  cloning openai/prm800k -> {target}")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/openai/prm800k", str(target)],
            check=True,
        )
    else:
        print(f"  skip clone ({target} exists)")

    src_dir = target / "prm800k" / "data"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        # Upstream typically ships gzipped JSONL; handle both shapes.
        plain = src_dir / f"phase2_{split}.jsonl"
        gz = src_dir / f"phase2_{split}.jsonl.gz"
        dst = PROCESSED_DIR / f"phase2_{split}.jsonl"
        if dst.exists():
            print(f"  skip phase2_{split} (exists)")
            continue
        if plain.exists():
            shutil.copy2(plain, dst)
            print(f"  copied {plain.name} -> {dst.relative_to(REPO_ROOT)}")
        elif gz.exists():
            with gzip.open(gz, "rb") as fin, dst.open("wb") as fout:
                shutil.copyfileobj(fin, fout)
            print(f"  decompressed {gz.name} -> {dst.relative_to(REPO_ROOT)}")
        else:
            print(f"  WARN: phase2_{split} not found under {src_dir}")


def main() -> int:
    print(f"REPO_ROOT={REPO_ROOT}")
    print(f"PROCESSED_DIR={PROCESSED_DIR}")

    _try("math_train", lambda: prep_math("train", "math_train"))
    _try("math_test", lambda: prep_math("test", "math_test"))
    _try("math_hard", prep_math_hard)
    _try("math_500", prep_math_500)
    _try("gsm8k_train", lambda: prep_gsm8k("train", "gsm8k_train"))
    _try("gsm8k_test", lambda: prep_gsm8k("test", "gsm8k_test"))
    _try("aime_2024", prep_aime_2024)
    _try("aime_2025", prep_aime_2025)
    _try("olympiad_bench", prep_olympiad_bench)
    _try("mmlu_stem", prep_mmlu_stem)
    _try("truthful_qa_mc", prep_truthful_qa)
    _try("prm800k", prep_prm800k)

    print(">>> Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
