#!/usr/bin/env python3
"""Pivot 1 (Abstaining Verifier) -- step-level Monte-Carlo rollout generation.

Self-supervision idea under test: a process step is "UNVERIFIABLE" when the
Monte-Carlo success rate of completing the solution from that step has a Wilson
confidence interval that straddles 0.5 (the model is genuinely on the fence),
as opposed to "verifiable-correct" (CI above 0.5) or "verifiable-wrong" (CI
below 0.5). This script generates the raw MC rollouts; av_analyze.py does the
3-part de-risk on the output.

For each PRM800K problem we build several step-prefixes (at fractional positions
through the human pre_generated_steps), and for each prefix sample K completions
from the math completer model, grading each final boxed answer against the gold
answer with math_verify. The per-prefix MC value = (# correct completions) / K.

Reads:  data/raw/prm800k/prm800k/data/phase2_test.jsonl  (on alias)
Writes: one JSONL of per-prefix records (shardable across GPUs).

Standalone vLLM, no training, no adapter (uses the base math model as completer).
Designed for a single 2080 Ti (fp16). Shard across cards with --shard/--num-shards.

Usage:
  av_gen_rollouts.py --device cuda:1 --num-shards 2 --shard 0 \
      --n-problems 600 --k 8 --out research/data/av_rollouts.s0.jsonl
"""
import argparse
import json
import os
import sys
import hashlib
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

PROMPT_TEMPLATE = (
    "Solve the following problem step by step. "
    "Put your final answer in \\boxed{{}}.\n\n"
    "Problem: {problem}"
)

FRACS = [0.0, 0.25, 0.5, 0.75]  # prefix boundaries through the reference steps


def seed_for(key: str) -> int:
    d = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(d[:4], "big") % (2**31)


def load_unique_problems(path, limit):
    """Dedupe PRM800K phase2 by problem text; keep first record with usable steps."""
    seen = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            q = d.get("question", {})
            prob = q.get("problem")
            gold = q.get("ground_truth_answer")
            steps = q.get("pre_generated_steps") or []
            if not prob or not gold or len(steps) < 2:
                continue
            if prob in seen:
                continue
            seen[prob] = {"problem": prob, "gold": gold, "steps": steps}
            if len(seen) >= limit:
                break
    return list(seen.values())


def build_prefixes(steps):
    """Return list of (prefix_len, frac) prefix boundaries, deduped, capped."""
    n = len(steps)
    out = []
    seen = set()
    for fr in FRACS:
        pl = int(round(n * fr))
        pl = max(0, min(pl, n - 1))  # leave at least 1 step for the model to do
        if pl in seen:
            continue
        seen.add(pl)
        out.append((pl, round(pl / n, 3)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--model", default=os.path.expanduser("~/models/qwen2_5_math_1_5b"))
    ap.add_argument("--prm", default=os.path.join(
        REPO, "data/raw/prm800k/prm800k/data/phase2_test.jsonl"))
    ap.add_argument("--n-problems", type=int, default=600)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=2560)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device_idx = args.device.split(":")[-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from asym_rewards.reward import is_correct, extract_boxed_answer
    from vllm import LLM, SamplingParams

    # Global problem id (gid) is assigned BEFORE sharding so a train/test split by
    # gid is deterministic and collision-free across shards.
    all_problems = load_unique_problems(args.prm, args.n_problems)
    problems = [(gid, p) for gid, p in enumerate(all_problems) if gid % args.num_shards == args.shard]
    print(f"[shard {args.shard}/{args.num_shards}] {len(problems)} problems on {args.device}",
          flush=True)

    # Build the full work list: one (problem, prefix) -> K sampling requests.
    # `prompt` (problem + step-prefix) is stored so the verifier training/eval has
    # its exact input without re-deriving it from PRM800K.
    prompts, metas, seeds = [], [], []
    for gid, p in problems:
        for (pl, frac) in build_prefixes(p["steps"]):
            base = PROMPT_TEMPLATE.format(problem=p["problem"])
            if pl > 0:
                base = base + "\n\n" + "\n".join(p["steps"][:pl])
            for k in range(args.k):
                prompts.append(base)
                metas.append({"gid": gid, "prefix_len": pl, "frac": frac,
                              "n_steps": len(p["steps"]), "gold": p["gold"],
                              "prompt": base, "k": k})
                seeds.append(seed_for(f"{p['problem'][:64]}:{pl}:{k}"))

    print(f"[shard {args.shard}] {len(prompts)} total generations", flush=True)

    llm = LLM(model=args.model, dtype="float16", gpu_memory_utilization=0.85,
              max_model_len=args.max_model_len)
    sps = [SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens,
                          seed=s, n=1) for s in seeds]
    outs = llm.generate(prompts, sps)

    # Aggregate K completions back per (pidx, prefix_len).
    agg = {}
    for meta, o in zip(metas, outs):
        comp = o.outputs[0].text
        ext = extract_boxed_answer(comp)
        ok = bool(is_correct(ext, meta["gold"])) if ext else False
        key = (meta["gid"], meta["prefix_len"])
        rec = agg.setdefault(key, {"gid": meta["gid"], "prefix_len": meta["prefix_len"],
                                   "frac": meta["frac"], "n_steps": meta["n_steps"],
                                   "gold": meta["gold"], "prompt": meta["prompt"],
                                   "K": 0, "n_correct": 0, "exts": []})
        rec["K"] += 1
        rec["n_correct"] += int(ok)
        rec["exts"].append(ext)

    # Store the modal-answer correctness label at generation time (math_verify is
    # available here) so av_analyze.py never has to re-grade -- it can run on the
    # laptop without math_verify and still use a real gold-graded label.
    for rec in agg.values():
        exts = [e for e in rec["exts"] if e]
        if exts:
            modal = Counter(exts).most_common(1)[0][0]
            rec["modal_correct"] = int(bool(is_correct(modal, rec["gold"])))
        else:
            rec["modal_correct"] = 0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for rec in agg.values():
            f.write(json.dumps(rec) + "\n")
    print(f"[shard {args.shard}] wrote {len(agg)} prefix records -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
