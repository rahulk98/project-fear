#!/usr/bin/env python
"""Post-hoc selective-prediction analysis across conditions.

Reads `outputs/eval/{model}_{cond}_{dtype}_{bench}_rollouts.jsonl` files
(already produced by eval_calibration.py) and emits:

  1. Per-condition selective metrics:
       attempt_rate, acc_on_attempted, acc_overall,
       attempt_right / attempt_wrong / abstain_on_right (E) / abstain_on_wrong (E)

  2. Neighbour-wise transition matrices on matched (problem_id, sample_idx):
       3x3 outcome transitions {correct,wrong,abstain} -> same,
       plus named counts (correct->abstain, wrong->abstain, ...).

No GPU needed - this is pure JSONL analysis and runs anywhere.

    python scripts/analyze_calibration.py \\
        --model qwen2_5_1_5b --dtype bfloat16 \\
        --conditions base,S0,S1,S2,S3,S4,C1,C2 \\
        --benchmarks math_test,gsm8k_test,aime_2024,aime_2025,mmlu_stem,truthful_qa_mc \\
        --pairs S0-S1,S1-S2,S2-S3,S3-S4,S0-C1,C1-C2,S2-C2,base-S2,base-C2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asym_rewards import metrics as M  # noqa: E402
from asym_rewards.reward import family_for_rollout_path, rescore_change_counts, rescore_rollouts  # noqa: E402


def _load_rollouts_with_changes(path: Path) -> tuple[list[dict], dict[str, int]]:
    with path.open() as f:
        rollouts = [json.loads(line) for line in f]
    rescored = rescore_rollouts(rollouts, family=family_for_rollout_path(path))
    return rescored, rescore_change_counts(rollouts, rescored)


def _rollouts_to_records(rollouts: list[dict]) -> list[dict]:
    by_pid: dict[str, list[dict]] = {}
    for r in rollouts:
        by_pid.setdefault(r["problem_id"], []).append(
            {"correct": r["correct"], "abstained": r["abstained"]}
        )
    return [{"problem_id": pid, "samples": ss} for pid, ss in by_pid.items()]


def _per_condition_metrics(rollouts: list[dict]) -> dict:
    records = _rollouts_to_records(rollouts)
    confusion = M.selective_confusion(records)
    return {
        "n_problems": len(records),
        "n_samples": sum(len(r["samples"]) for r in records),
        "attempt_rate":     M.attempt_rate(records),
        "abstention_rate":  M.abstention_rate(records),
        "acc_overall":      M.accuracy(records),
        "acc_on_attempted": M.accuracy_on_attempted(records),
        "confusion":        confusion,
    }


def _format_confusion(conf: dict) -> str:
    return (
        f"    attempt_right={conf['attempt_right']:5d}  "
        f"attempt_wrong={conf['attempt_wrong']:5d}  "
        f"abstain_on_right(E)={conf['abstain_on_right_expected']:7.2f}  "
        f"abstain_on_wrong(E)={conf['abstain_on_wrong_expected']:7.2f}  "
        f"abstain_unknown={conf['abstain_unknown']:4d}"
    )


def _format_transition_named(named: dict, n_pairs: int) -> str:
    return (
        f"    n_pairs={n_pairs}  "
        f"correct->abstain={named['correct_to_abstain']}  "
        f"wrong->abstain={named['wrong_to_abstain']}  "
        f"abstain->correct={named['abstain_to_correct']}  "
        f"abstain->wrong={named['abstain_to_wrong']}  "
        f"correct->wrong={named['correct_to_wrong']}  "
        f"wrong->correct={named['wrong_to_correct']}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--conditions", default="base,S0,S1,S2,S3,S4,C1,C2")
    p.add_argument("--benchmarks",
                   default="math_test,gsm8k_test,aime_2024,aime_2025,mmlu_stem,truthful_qa_mc")
    p.add_argument("--pairs", default="S0-S1,S1-S2,S2-S3,S3-S4,S0-C1,C1-C2,S2-C2",
                   help="comma-separated condition pairs A-B for transition matrices")
    p.add_argument("--in-dir",  default=str(REPO_ROOT / "outputs" / "eval"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "eval" / "analysis"))
    p.add_argument("--greedy", action="store_true",
                   help="read the paired-mode `_greedy` rollout files instead of the K=16 stochastic ones")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    benches    = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    pairs      = [tuple(p.strip().split("-")) for p in args.pairs.split(",") if p.strip()]
    suffix     = "_greedy" if args.greedy else ""

    # ---- 1. Per-condition selective metrics --------------------------------
    selective: dict[str, dict[str, dict]] = {}
    rollout_cache: dict[tuple[str, str], list[dict]] = {}
    for cond in conditions:
        selective[cond] = {}
        for bench in benches:
            path = in_dir / f"{args.model}_{cond}_{args.dtype}{suffix}_{bench}_rollouts.jsonl"
            if not path.exists():
                continue
            rollouts, rescore_changes = _load_rollouts_with_changes(path)
            rollout_cache[(cond, bench)] = rollouts
            metrics = _per_condition_metrics(rollouts)
            metrics["rescore_changes"] = rescore_changes
            selective[cond][bench] = metrics
            print(f"[{cond:>4} {bench:<16}] "
                  f"attempt={metrics['attempt_rate']:.3f}  "
                  f"acc_attempted={metrics['acc_on_attempted']:.3f}  "
                  f"acc_overall={metrics['acc_overall']:.3f}")
            print(_format_confusion(metrics["confusion"]))

    selective_out = out_dir / f"{args.model}_{args.dtype}{suffix}_selective.json"
    selective_out.write_text(json.dumps(selective, indent=2))
    print(f"\nwrote {selective_out}")

    # ---- 2. Neighbour-wise transition matrices -----------------------------
    transitions: dict[str, dict[str, dict]] = {}
    if args.greedy:
        interp = ("PAIRED CAUSAL (greedy, T=0, K=1): same problem, only weights "
                  "differ. Each per-sample cell is a clean per-problem outcome flip.")
    else:
        interp = ("STOCHASTIC AGGREGATE (T=0.7, K=16): matched (pid, sample_idx) "
                  "pairs share the RNG seed only - different weights produce different "
                  "multinomial draws from step 0, so trajectories diverge. Transition "
                  "counts are valid as a problem-level paired aggregate, NOT as "
                  "per-sample causal flips. For per-sample causal claims, rerun with --greedy.")
    print(f"\nTransitions (matched on (problem_id, sample_idx))")
    print(f"  Interpretation: {interp}")
    for cond_a, cond_b in pairs:
        key = f"{cond_a}->{cond_b}"
        transitions[key] = {}
        for bench in benches:
            ra = rollout_cache.get((cond_a, bench))
            rb = rollout_cache.get((cond_b, bench))
            if ra is None or rb is None:
                continue
            transition = M.condition_transition(ra, rb)
            transitions[key][bench] = transition
            print(f"  [{key} on {bench}]")
            print(_format_transition_named(transition["named"], transition["n_pairs"]))

    transitions_out = out_dir / f"{args.model}_{args.dtype}{suffix}_transitions.json"
    payload = {
        "_mode": "greedy" if args.greedy else "stochastic_K16",
        "_interpretation": interp,
        "transitions": transitions,
    }
    transitions_out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {transitions_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
