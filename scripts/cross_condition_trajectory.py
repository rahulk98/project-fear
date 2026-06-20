"""Per-problem cross-condition analyses on existing rollouts JSONLs.

Three analyses (offline, no GPU):
1. Trajectory / stacking: classify each problem's abstention pattern across the
   EV-ordered conditions (S0 < C1 < S1 < C2 < S2 < S3 < S4) and count perfect-
   chain ("stacked") problems. Tests whether the asymmetric reward sweep induces
   a controllable threshold along a shared difficulty axis.
2. Difficulty-axis recovery: correlate abstain_count (= number of conditions
   that majority-abstain on a problem) against gold-truth difficulty signals
   (MATH level, subject; GSM8K prompt length).
3. Consistent-attempter subsets: on problems where no condition abstains, does
   asymmetric pressure still hurt accuracy? On problems where all conditions
   abstain, are these in fact unsolvable, or does S0 Pass@16 reveal latent
   capability?

Writes a JSON summary to `outputs/eval/trajectory_summary.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from asym_rewards.data import load_benchmark  # noqa: E402
from asym_rewards.reward import family_for_rollout_path, rescore_rollouts  # noqa: E402

EV_ORDER = ["S0", "C1", "S1", "C2", "S2", "S3", "S4"]
EV_THRESHOLDS = {"S0": 0.50, "C1": 0.60, "S1": 0.67, "C2": 0.75, "S2": 0.80, "S3": 0.91, "S4": 1.00}


def load_per_problem(eval_dir: Path, model: str, dtype: str, bench: str) -> dict:
    """For each (condition, problem), record abst_rate, p_per_sample, p_per_attempt, correct_any.

    p_per_sample = sum(correct) / K        (counts abstentions as wrong)
    p_per_attempt = sum(correct) / (K - n_abst)  (excludes abstentions; None if all abstained)
    """
    out: dict[str, dict] = {}
    for c in EV_ORDER:
        p = eval_dir / f"{model}_{c}_{dtype}_{bench}_rollouts.jsonl"
        if not p.exists():
            return {}
        by_pid: dict[str, list] = defaultdict(list)
        rollouts = rescore_rollouts(
            [json.loads(line) for line in p.open()], family=family_for_rollout_path(p),
        )
        for r in rollouts:
            by_pid[r["problem_id"]].append((r["abstained"], r["correct"]))
        cond_out: dict = {}
        for pid, v in by_pid.items():
            K = len(v)
            n_abst = sum(a for a, _ in v)
            n_correct = sum(co for _, co in v)
            n_attempts = K - n_abst
            cond_out[pid] = {
                "K": K,
                "abst_rate": n_abst / K,
                "p_per_sample": n_correct / K,
                "p_per_attempt": (n_correct / n_attempts) if n_attempts > 0 else None,
                "correct_any": bool(n_correct > 0),
                "n_attempts": n_attempts,
                "n_correct": n_correct,
            }
        out[c] = cond_out
    return out


def trajectory(per_problem: dict, abst_threshold: float = 0.5) -> dict:
    pids = sorted(set.intersection(*(set(per_problem[c].keys()) for c in EV_ORDER)))
    n = len(pids)
    stack = np.array([
        [per_problem[c][pid]["abst_rate"] >= abst_threshold for c in EV_ORDER] for pid in pids
    ])
    abst_count = stack.sum(axis=1)

    # Stacking: row is 0...01...1 (a chain in EV order)
    stacked = 0
    for row in stack:
        if not row.any() or row.all():
            stacked += 1
            continue
        first_true = int(np.argmax(row))
        if row[first_true:].all():
            stacked += 1

    abst_distribution = {int(k): int((abst_count == k).sum()) for k in range(8)}
    border_first = {c: 0 for c in EV_ORDER}
    for row in stack:
        if row.all() or not row.any():
            continue
        border_first[EV_ORDER[int(np.argmax(row))]] += 1
    return {
        "n_problems": n,
        "stacked": int(stacked),
        "stacked_pct": float(stacked / n) if n else 0.0,
        "abst_count_distribution": abst_distribution,
        "border_first_abstainer": border_first,
        "_pids": pids,
        "_abst_count": abst_count.tolist(),
    }


def difficulty_axis_math(per_problem: dict, traj: dict) -> dict:
    pids = traj["_pids"]
    abst_count = np.array(traj["_abst_count"])
    bench = load_benchmark("math_test")[["id", "level", "subject"]]
    bench["problem_id"] = bench["id"].astype(str)
    bench["level"] = bench["level"].astype(int)
    df = pd.DataFrame({"problem_id": pids, "abst_count": abst_count}).merge(bench, on="problem_id", how="left")

    rho, p = spearmanr(df["level"], df["abst_count"])
    by_level = df.groupby("level")["abst_count"].agg(["mean", "std", "count"]).reset_index()
    always_att_pct = df.groupby("level").apply(lambda x: float((x["abst_count"] == 0).mean()))
    always_abst_pct = df.groupby("level").apply(lambda x: float((x["abst_count"] == 7).mean()))
    by_subject = df.groupby("subject")["abst_count"].agg(["mean", "count"]).reset_index()
    by_subject = by_subject[by_subject["count"] >= 100].sort_values("mean")

    return {
        "spearman_level_vs_abst_count": {"rho": float(rho), "p": float(p)},
        "by_level": [
            {
                "level": int(row["level"]),
                "mean_abst_count": float(row["mean"]),
                "std": float(row["std"]),
                "n": int(row["count"]),
                "pct_always_attempted": float(always_att_pct.get(row["level"], 0)),
                "pct_always_abstained": float(always_abst_pct.get(row["level"], 0)),
            }
            for _, row in by_level.iterrows()
        ],
        "by_subject_sorted_easiest_to_hardest": [
            {"subject": row["subject"], "mean_abst_count": float(row["mean"]), "n": int(row["count"])}
            for _, row in by_subject.iterrows()
        ],
    }


def difficulty_axis_gsm8k(per_problem: dict, traj: dict) -> dict:
    pids = traj["_pids"]
    abst_count = np.array(traj["_abst_count"])
    bench = load_benchmark("gsm8k_test")[["id", "prompt"]].copy()
    bench["problem_id"] = bench["id"].astype(str)
    bench["prompt_len"] = bench["prompt"].str.len()
    df = pd.DataFrame({"problem_id": pids, "abst_count": abst_count}).merge(bench, on="problem_id", how="left")
    rho, p = spearmanr(df["prompt_len"], df["abst_count"])
    return {"spearman_prompt_len_vs_abst_count": {"rho": float(rho), "p": float(p)}}


def consistent_attempters(per_problem: dict) -> dict:
    pids = sorted(set.intersection(*(set(per_problem[c].keys()) for c in EV_ORDER)))
    always_att = [
        pid for pid in pids if all(per_problem[c][pid]["abst_rate"] < 0.5 for c in EV_ORDER)
    ]
    unanimous = [
        pid for pid in pids if all(per_problem[c][pid]["abst_rate"] == 0 for c in EV_ORDER)
    ]
    always_abst = [
        pid for pid in pids if all(per_problem[c][pid]["abst_rate"] >= 0.5 for c in EV_ORDER)
    ]

    def acc_on(subset: list, c: str) -> dict:
        if not subset:
            return {"pass_at_K": None, "mean_p_per_sample": None, "mean_p_per_attempt": None, "n": 0}
        # Per-attempt rate aggregated correctly: sum(correct) / sum(attempts) across the subset,
        # not the simple mean of per-problem ratios (which would weight problems with few attempts equally).
        total_correct = sum(per_problem[c][pid]["n_correct"] for pid in subset)
        total_attempts = sum(per_problem[c][pid]["n_attempts"] for pid in subset)
        total_samples = sum(per_problem[c][pid]["K"] for pid in subset)
        return {
            "pass_at_K": float(np.mean([per_problem[c][pid]["correct_any"] for pid in subset])),
            "mean_p_per_sample": float(total_correct / total_samples) if total_samples else None,
            "mean_p_per_attempt": float(total_correct / total_attempts) if total_attempts else None,
            "n": len(subset),
            "total_attempts": total_attempts,
            "total_samples": total_samples,
        }

    return {
        "subset_sizes": {
            "n_total": len(pids),
            "always_attempted": len(always_att),
            "unanimous": len(unanimous),
            "always_abstained": len(always_abst),
        },
        "always_attempted_acc": {c: acc_on(always_att, c) for c in EV_ORDER},
        "unanimous_acc": {c: acc_on(unanimous, c) for c in EV_ORDER},
        "always_abstained_acc": {c: acc_on(always_abst, c) for c in EV_ORDER},
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2_5_1_5b")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--eval-dir", default="outputs/eval")
    p.add_argument("--benchmarks", default="math_test,gsm8k_test")
    p.add_argument("--out", default="outputs/eval/trajectory_summary.json")
    args = p.parse_args()

    eval_dir = Path(args.eval_dir)
    summary: dict = {}
    for bench in args.benchmarks.split(","):
        per_p = load_per_problem(eval_dir, args.model, args.dtype, bench)
        if not per_p:
            summary[bench] = {"missing": True}
            continue
        traj = trajectory(per_p)
        cons = consistent_attempters(per_p)
        s = {
            "trajectory": {k: v for k, v in traj.items() if not k.startswith("_")},
            "consistent_attempters": cons,
        }
        if bench == "math_test":
            s["difficulty_axis"] = difficulty_axis_math(per_p, traj)
        elif bench == "gsm8k_test":
            s["difficulty_axis"] = difficulty_axis_gsm8k(per_p, traj)
        summary[bench] = s
        print(f"  {bench}: n={traj['n_problems']} stacked={traj['stacked']}/{traj['n_problems']} ({traj['stacked_pct']*100:.1f}%)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"OK trajectory -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
