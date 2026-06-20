#!/usr/bin/env python
"""Generate reasoning traces for Exp-3 reference data.

Backends:
  - ollama (default): local qwen3:30b on the lab GPUs.
  - opencode: OpenCode Zen OpenAI-compatible API; reads OPENCODE_API from .env.
    Use a `*-free` model to stay within the Go-tier $0 quota
    (e.g. deepseek-v4-flash-free).

Produces k samples per problem with stable per-sample seeds, scores correctness
with math_verify, and is resumable (per-50 flush, dedup on (problem_id, sample_idx)).

Generator and judge must differ: judges in Exp 3 are the trained
Qwen2.5-3B / Llama-3.2-3B, neither of which appears as a generator here.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asym_rewards.data import load_benchmark  # noqa: E402
from asym_rewards.reward import extract_boxed_answer, is_correct  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="benchmark name (e.g. math_train, gsm8k_train)")
    p.add_argument("--levels", default="", help="comma-separated level filter (ignored if dataset has no 'level' col)")
    p.add_argument("--backend", choices=["ollama", "opencode"], default="ollama")
    p.add_argument("--model", default="qwen3:30b",
                   help="model tag - ollama format (qwen3:30b) or opencode bare id (deepseek-v4-flash-free)")
    p.add_argument("--k", type=int, default=4, help="samples per problem")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--num-ctx", type=int, default=0,
                   help="ollama num_ctx; 0 = auto (max_tokens + 1024). qwen3:30b defaults "
                        "to 262144 which fails to allocate; cap it to keep KV cache sane. "
                        "Ignored for opencode backend.")
    p.add_argument("--limit", type=int, default=0, help="0 = no limit (use small int for smoke)")
    p.add_argument("--n-problems", type=int, default=0,
                   help="random subsample of N problems (after --levels filter, before --limit). "
                        "0 = use all. Deterministic via --sample-seed.")
    p.add_argument("--sample-seed", type=int, default=42, help="seed for --n-problems subsample")
    p.add_argument("--out", required=True, help="output parquet path")
    p.add_argument("--host", default="http://127.0.0.1:11434", help="ollama host")
    p.add_argument("--opencode-base-url", default="https://opencode.ai/zen/v1",
                   help="OpenCode Zen OpenAI-compatible base URL")
    p.add_argument("--flush-every", type=int, default=50)
    p.add_argument("--no-think", action="store_true",
                   help="disable thinking mode (ollama only; opencode reasoning is determined by model)")
    p.add_argument("--rate-delay", type=float, default=0.0,
                   help="sleep this many seconds between requests (helps avoid 429s on free tiers)")
    p.add_argument("--max-retries", type=int, default=5,
                   help="retry attempts on 429/5xx (exponential backoff 4s, 8s, 16s, 32s, 64s)")
    return p.parse_args()


def _say(msg: str) -> None:
    """Print + flush so output is visible through tee/pipes."""
    print(msg, flush=True)


def build_prompt(problem: str) -> str:
    return (
        "Solve the following problem step by step. "
        "Put your final answer in \\boxed{}.\n\n"
        f"Problem: {problem}"
    )


def seed_for(problem_id: str, sample_idx: int) -> int:
    # Stable across Python processes (unlike hash(), which is PYTHONHASHSEED-randomized).
    digest = hashlib.sha256(f"{problem_id}:{sample_idx}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


def load_existing(out_path: Path) -> set[tuple[str, int]]:
    if not out_path.exists():
        return set()
    existing = pd.read_parquet(out_path)
    return set(zip(existing["problem_id"].astype(str), existing["sample_idx"].astype(int)))


def flush(buffer: list[dict[str, Any]], out_path: Path) -> None:
    if not buffer:
        return
    new_df = pd.DataFrame(buffer)
    if out_path.exists():
        old_df = pd.read_parquet(out_path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["problem_id", "sample_idx"], keep="last")
        combined.to_parquet(out_path, index=False)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        new_df.to_parquet(out_path, index=False)
    buffer.clear()


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)

    load_dotenv()
    ollama_client = None
    opencode_session = None
    if args.backend == "ollama":
        try:
            import ollama
        except ImportError as e:
            print(f"ERROR: ollama python package missing: {e}", file=sys.stderr)
            return 1
        ollama_client = ollama.Client(host=args.host)
    else:  # opencode
        token = os.environ.get("OPENCODE_API")
        if not token:
            print("ERROR: OPENCODE_API missing from .env", file=sys.stderr)
            return 1
        opencode_session = requests.Session()
        opencode_session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })

    df = load_benchmark(args.dataset)
    _say(f"loaded {len(df)} rows from benchmark '{args.dataset}'")

    if args.levels and "level" in df.columns:
        wanted = {int(x) for x in args.levels.split(",") if x.strip()}
        before = len(df)
        df = df[df["level"].isin(wanted)].reset_index(drop=True)
        _say(f"filtered to levels {sorted(wanted)}: {before} -> {len(df)} rows")
    elif args.levels:
        _say(f"NOTE: --levels {args.levels} ignored (dataset has no 'level' column)")

    if args.n_problems and args.n_problems > 0 and len(df) > args.n_problems:
        before = len(df)
        df = df.sample(n=args.n_problems, random_state=args.sample_seed).reset_index(drop=True)
        _say(f"subsampled to {args.n_problems} problems (seed={args.sample_seed}): {before} -> {len(df)} rows")

    if args.limit and args.limit > 0:
        df = df.head(args.limit).reset_index(drop=True)
        _say(f"limited to first {len(df)} rows (--limit {args.limit})")

    done = load_existing(out_path)
    _say(f"resuming: {len(done)} (problem_id, sample_idx) pairs already in {out_path}")

    total_planned = len(df) * args.k
    _say(f"plan: {len(df)} problems x k={args.k} = {total_planned} total samples")
    _say(f"backend: {args.backend} | model: {args.model}")
    if args.backend == "ollama":
        _say(f"think mode: {'OFF' if args.no_think else 'ON'} (Qwen3 thinking field captured in trace)")

    buffer: list[dict[str, Any]] = []
    n_new = 0
    n_correct_new = 0
    t_start = time.time()

    for row_idx, row in df.iterrows():
        problem_id = str(row["id"])
        problem = str(row["prompt"])
        gold = str(row["answer"])
        level = row.get("level") if "level" in df.columns else None

        for sample_idx in range(args.k):
            if (problem_id, sample_idx) in done:
                continue

            seed = seed_for(problem_id, sample_idx)
            prompt = build_prompt(problem)

            t0 = time.time()
            try:
                if args.backend == "ollama":
                    num_ctx = args.num_ctx if args.num_ctx > 0 else args.max_tokens + 1024
                    resp = ollama_client.generate(
                        model=args.model,
                        prompt=prompt,
                        think=(not args.no_think),
                        options={
                            "temperature": args.temperature,
                            "num_predict": args.max_tokens,
                            "num_ctx": num_ctx,
                            "seed": seed,
                        },
                    )
                    thinking = resp.get("thinking") or ""
                    response_text = resp.get("response") or ""
                    n_tokens = int(resp.get("eval_count") or 0)
                else:  # opencode
                    payload = {
                        "model": args.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": args.max_tokens,
                        "temperature": args.temperature,
                        "seed": seed,
                    }
                    last_err = None
                    data = None
                    for attempt in range(args.max_retries + 1):
                        try:
                            r = opencode_session.post(
                                f"{args.opencode_base_url}/chat/completions",
                                json=payload,
                                timeout=600,
                            )
                            if r.status_code in (429, 500, 502, 503, 504):
                                wait = 4.0 * (2 ** attempt)
                                _say(f"  HTTP {r.status_code} on ({problem_id}, {sample_idx}) attempt {attempt+1}/{args.max_retries+1}; sleeping {wait:.0f}s")
                                time.sleep(wait)
                                continue
                            r.raise_for_status()
                            data = r.json()
                            break
                        except requests.exceptions.RequestException as e:
                            last_err = e
                            wait = 4.0 * (2 ** attempt)
                            _say(f"  {type(e).__name__} on ({problem_id}, {sample_idx}) attempt {attempt+1}/{args.max_retries+1}; sleeping {wait:.0f}s")
                            time.sleep(wait)
                    if data is None:
                        raise RuntimeError(f"opencode exhausted retries: {last_err}")
                    msg = data["choices"][0]["message"]
                    thinking = msg.get("reasoning_content") or ""
                    response_text = msg.get("content") or ""
                    n_tokens = int(data.get("usage", {}).get("completion_tokens") or 0)
                    if args.rate_delay > 0:
                        time.sleep(args.rate_delay)
            except Exception as e:  # noqa: BLE001
                _say(f"  ERROR generating ({problem_id}, {sample_idx}): {type(e).__name__}: {e}")
                continue
            gen_seconds = time.time() - t0

            # Reasoning models put CoT in a separate field (ollama: 'thinking',
            # opencode: 'reasoning_content'); concatenate with the final answer so
            # downstream Exp-3 judges see the full trace.
            if thinking and response_text:
                trace = f"<think>{thinking}</think>\n{response_text}"
            else:
                trace = thinking + response_text
            extracted = extract_boxed_answer(response_text) or extract_boxed_answer(trace)
            correct = is_correct(extracted, gold)

            buffer.append({
                "problem_id": problem_id,
                "sample_idx": sample_idx,
                "seed": seed,
                "problem": problem,
                "level": int(level) if level is not None and pd.notna(level) else None,
                "gold_answer": gold,
                "trace": trace,
                "extracted_answer": extracted,
                "correct": correct,
                "n_tokens": n_tokens,
                "model": args.model,
                "gen_seconds": gen_seconds,
            })
            n_new += 1
            n_correct_new += int(correct)
            mark = "Y" if correct else "n"
            elapsed = time.time() - t_start
            _say(
                f"  [{n_new}/{total_planned - len(done)}] "
                f"pid={problem_id} s={sample_idx} ok={mark} "
                f"toks={n_tokens} gen={gen_seconds:.1f}s "
                f"cum={elapsed/60:.1f}m"
            )

            if len(buffer) >= args.flush_every:
                flush(buffer, out_path)
                acc = (n_correct_new / n_new) if n_new else 0.0
                _say(
                    f"  flushed @ row {row_idx + 1}/{len(df)} "
                    f"sample {sample_idx + 1}/{args.k} | "
                    f"new={n_new} correct%={acc:.1%} elapsed={elapsed/60:.1f}m"
                )

    flush(buffer, out_path)

    _say(">>> Summary")
    if out_path.exists():
        final = pd.read_parquet(out_path)
        _say(f"  total rows in {out_path}: {len(final)}")
        if len(final):
            _say(f"  overall correct%: {final['correct'].mean():.1%}")
            if "level" in final.columns and final["level"].notna().any():
                by_level = final.dropna(subset=["level"]).groupby("level")["correct"].mean()
                for lvl, acc in by_level.items():
                    _say(f"    level {int(lvl)}: {acc:.1%}")
            _say(f"  mean n_tokens: {final['n_tokens'].mean():.0f}")
            class_balance = (
                final.groupby("problem_id")["correct"]
                .agg(lambda s: (s.sum() >= 1) and ((~s).sum() >= 1))
                .sum()
            )
            _say(f"  problems with both correct AND incorrect samples (Exp-3 ready): {class_balance}")
    elapsed = time.time() - t_start
    _say(f"  wall: {elapsed/60:.1f} min, new samples this run: {n_new}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
