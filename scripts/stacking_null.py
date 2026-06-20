"""Empirical null test for per-problem stacking across EV-ordered conditions.

For each benchmark, compares the observed fraction of "stacked" problems (abstention
pattern is a zeros-then-ones chain in EV order) against two nulls:
  Null A: independent Bernoulli per condition with the observed marginal abst rate.
  Null B: column-shuffle that preserves per-condition marginals but breaks per-problem
          alignment across conditions.

Both nulls answer the codex peer-review critique that the naive 6.25% baseline
(uniform-random over 128 patterns) ignores the correlated-training prior. The
marginal-preserving null sets the proper baseline (~59% on math/gsm8k given
the observed marginals are skewed by S4's high abst rate); observed 92-94%
remains 25-99 sigma above this null.

Writes the comparison to outputs/eval/stacking_null.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from asym_rewards.reward import family_for_rollout_path, rescore_rollouts  # noqa: E402

EV_ORDER = ["S0", "C1", "S1", "C2", "S2", "S3", "S4"]


def load_per_problem_abst(eval_dir: Path, model: str, dtype: str, bench: str,
                          thresh: float = 0.5) -> dict:
    out: dict[str, dict[str, bool]] = {}
    for c in EV_ORDER:
        p = eval_dir / f"{model}_{c}_{dtype}_{bench}_rollouts.jsonl"
        if not p.exists():
            return {}
        by_pid: dict[str, list[bool]] = defaultdict(list)
        rollouts = rescore_rollouts(
            [json.loads(line) for line in p.open()], family=family_for_rollout_path(p),
        )
        for r in rollouts:
            by_pid[r["problem_id"]].append(r["abstained"])
        out[c] = {pid: sum(v) / len(v) >= thresh for pid, v in by_pid.items()}
    return out


def stack_count(matrix: np.ndarray) -> int:
    """Rows where pattern is 0...01...1 (zeros then ones, no interleave)."""
    n = 0
    for row in matrix:
        if not row.any() or row.all():
            n += 1
            continue
        first_true = int(np.argmax(row))
        if row[first_true:].all():
            n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2_5_1_5b")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--eval-dir", default="outputs/eval")
    p.add_argument("--benchmarks", default="math_test,gsm8k_test")
    p.add_argument("--n-iter", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/eval/stacking_null.json")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    eval_dir = Path(args.eval_dir)
    summary: dict = {}
    for bench in args.benchmarks.split(","):
        data = load_per_problem_abst(eval_dir, args.model, args.dtype, bench)
        if not data:
            summary[bench] = {"missing": True}
            continue
        pids = sorted(set.intersection(*(set(data[c].keys()) for c in EV_ORDER)))
        n = len(pids)
        observed = np.array([[data[c][pid] for c in EV_ORDER] for pid in pids], dtype=bool)
        marg = observed.mean(axis=0)
        obs_stacked = stack_count(observed)

        nullA = np.array([
            stack_count(rng.random((n, 7)) < marg) / n for _ in range(args.n_iter)
        ])
        # column-shuffle: preserves marginals exactly, breaks alignment
        nullB = []
        for _ in range(args.n_iter):
            sim = observed.copy()
            for col in range(7):
                rng.shuffle(sim[:, col])
            nullB.append(stack_count(sim) / n)
        nullB = np.array(nullB)

        bench_summary = {
            "n_problems": n,
            "observed_stacked": int(obs_stacked),
            "observed_pct": float(obs_stacked / n),
            "marginal_abst": {c: float(m) for c, m in zip(EV_ORDER, marg)},
            "null_independent_bernoulli": {
                "mean_pct": float(nullA.mean()),
                "ci_2_5_pct": float(np.percentile(nullA, 2.5)),
                "ci_97_5_pct": float(np.percentile(nullA, 97.5)),
                "std_pct": float(nullA.std()),
                "z_ratio_obs_vs_null": float((obs_stacked / n - nullA.mean()) / max(nullA.std(), 1e-9)),
            },
            "null_column_shuffle": {
                "mean_pct": float(nullB.mean()),
                "ci_2_5_pct": float(np.percentile(nullB, 2.5)),
                "ci_97_5_pct": float(np.percentile(nullB, 97.5)),
                "std_pct": float(nullB.std()),
                "z_ratio_obs_vs_null": float((obs_stacked / n - nullB.mean()) / max(nullB.std(), 1e-9)),
            },
        }
        summary[bench] = bench_summary
        print(f"{bench}: obs {obs_stacked}/{n} = {obs_stacked/n*100:.2f}%; nullA {nullA.mean()*100:.2f}% (z {bench_summary['null_independent_bernoulli']['z_ratio_obs_vs_null']:.1f}); nullB {nullB.mean()*100:.2f}% (z {bench_summary['null_column_shuffle']['z_ratio_obs_vs_null']:.1f})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"OK stacking-null -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
