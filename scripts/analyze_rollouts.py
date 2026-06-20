"""Offline analyses on existing rollouts JSONLs.

Computes calibration metrics (corrected ECE), selective-prediction curves,
hard-vs-easy abstention, threshold-recovery, token efficiency, and the
C2-vs-S2 head-to-head from `outputs/eval/{model}_{COND}_{dtype}_{bench}_rollouts.jsonl`.

Outputs a single JSON summary keyed by (condition, benchmark, analysis)
to `outputs/eval/analysis/{model}_{dtype}_summary.json`.

Designed to consume the eval-script's existing rollout cache so it does
NOT need GPU time. Re-run anytime new conditions or seeds land.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from asym_rewards.data import load_benchmark  # noqa: E402
from asym_rewards.reward import family_for_rollout_path, rescore_change_counts, rescore_rollouts  # noqa: E402


def load_rollouts(path: Path) -> list[dict]:
    return rescore_rollouts(
        [json.loads(l) for l in path.open()], family=family_for_rollout_path(path),
    )


def load_rollouts_with_changes(path: Path) -> tuple[list[dict], dict[str, int]]:
    rollouts = [json.loads(l) for l in path.open()]
    rescored = rescore_rollouts(rollouts, family=family_for_rollout_path(path))
    return rescored, rescore_change_counts(rollouts, rescored)


def ece(p_hat: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width-bin ECE. Standard definition: weighted mean |conf - acc| per bin."""
    if len(p_hat) == 0:
        return 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    n = len(p_hat)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p_hat >= lo) & (p_hat <= hi)
        else:
            mask = (p_hat >= lo) & (p_hat < hi)
        if mask.sum() == 0:
            continue
        e += (mask.sum() / n) * abs(p_hat[mask].mean() - correct[mask].mean())
    return float(e)


def rollouts_to_problems(rollouts: list[dict]) -> dict[str, list[dict]]:
    by_pid: dict[str, list[dict]] = defaultdict(list)
    for r in rollouts:
        by_pid[r["problem_id"]].append(r)
    return by_pid


def selective_acc_curve(problems: dict, sort_by: str) -> tuple[np.ndarray, np.ndarray]:
    """Return coverage and accuracy arrays for a coverage-accuracy curve.

    sort_by: 'p_hat' (Pass@K fraction) or 'mode_conf' (majority-vote confidence).
    """
    items = []
    for pid, samples in problems.items():
        K = len(samples)
        correct = [s["correct"] for s in samples]
        p_hat = sum(correct) / K
        # Modal answer must be computed over samples that actually answered.
        # For D1/CWD1, `extracted="IDK"` with `abstained=True` would otherwise
        # be ranked as the modal "answer", giving an all-IDK problem
        # mode_conf=1.0 alongside abst_rate=1.0.
        non_abst = [s["extracted"] for s in samples if not s.get("abstained")]
        if non_abst:
            mode_ans, mode_count = Counter(non_abst).most_common(1)[0]
            mode_conf = mode_count / K
            mode_correct = any(
                s["correct"] for s in samples
                if not s.get("abstained") and s["extracted"] == mode_ans
            )
        else:
            mode_conf = 0.0
            mode_correct = False
        s = p_hat if sort_by == "p_hat" else mode_conf
        items.append((s, int(mode_correct if sort_by == "mode_conf" else any(correct))))
    items.sort(key=lambda x: -x[0])
    n = len(items)
    cov = np.arange(1, n + 1) / n
    acc = np.cumsum([c for _, c in items]) / np.arange(1, n + 1)
    return cov, acc


def per_problem_metrics(rollouts: list[dict]) -> pd.DataFrame:
    """One row per problem with K, p_hat, correct_any, mode_conf, mode_correct, abst_rate, mean_tokens."""
    by_pid = rollouts_to_problems(rollouts)
    rows = []
    for pid, samples in by_pid.items():
        K = len(samples)
        correct = [s["correct"] for s in samples]
        abstained = [s["abstained"] for s in samples]
        completions = [s["completion"] for s in samples]
        p_hat = sum(correct) / K
        # Drive modal-answer + entropy off the abstention flag, not truthy
        # `extracted`, so D1 `\boxed{IDK}` rows are excluded (their
        # abstained=True is set by rescore_rollouts under family="d1").
        non_abst = [s["extracted"] for s in samples if not s.get("abstained")]
        if non_abst:
            mode_ans, mode_count = Counter(non_abst).most_common(1)[0]
            mode_conf = mode_count / K
            mode_correct = any(
                s["correct"] for s in samples
                if not s.get("abstained") and s["extracted"] == mode_ans
            )
        else:
            mode_conf = 0.0
            mode_correct = False
        # answer-entropy among non-abstained samples
        if len(non_abst) > 1:
            ans_counts = np.array(list(Counter(non_abst).values()))
            ans_p = ans_counts / ans_counts.sum()
            ent = float(-(ans_p * np.log(ans_p + 1e-12)).sum())
        else:
            ent = 0.0
        rows.append({
            "problem_id": pid,
            "K": K,
            "p_hat": p_hat,
            "correct_any": any(correct),
            "mode_conf": mode_conf,
            "mode_correct": mode_correct,
            "abst_rate": sum(abstained) / K,
            "n_non_abstained": len(non_abst),
            "ans_entropy": ent,
            "mean_tokens": float(np.mean([len(c) for c in completions])),  # char count proxy
        })
    return pd.DataFrame(rows)


def compute_pass_at_k(rollouts: list[dict], k_values: list[int]) -> dict[int, float]:
    """Unbiased Pass@k estimator across the corpus."""
    by_pid = rollouts_to_problems(rollouts)
    out = {}
    for k in k_values:
        vals = []
        for pid, samples in by_pid.items():
            n = len(samples)
            c = sum(s["correct"] for s in samples)
            if k > n:
                vals.append(float(c >= 1))
            elif n - c < k:
                vals.append(1.0)
            else:
                # 1 - C(n-c, k) / C(n, k)
                from math import comb
                vals.append(1.0 - comb(n - c, k) / comb(n, k))
        out[k] = float(np.mean(vals))
    return out


def analyse_condition_benchmark(model: str, cond: str, dtype: str, bench: str, out_dir: Path) -> dict:
    rpath = out_dir / f"{model}_{cond}_{dtype}_{bench}_rollouts.jsonl"
    if not rpath.exists():
        return {"missing": True, "path": str(rpath)}
    rollouts, rescore_changes = load_rollouts_with_changes(rpath)
    df = per_problem_metrics(rollouts)
    n = len(df)

    # ECE variants
    p_hat = df["p_hat"].to_numpy()
    correct_any = df["correct_any"].astype(int).to_numpy()
    mode_conf = df["mode_conf"].to_numpy()
    mode_correct = df["mode_correct"].astype(int).to_numpy()

    # Selective accuracy at fixed coverages (mode-vote ranking)
    items = sorted(zip(mode_conf, mode_correct), key=lambda x: -x[0])
    sel_accs = {}
    for cov in [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        k = max(1, int(round(cov * n)))
        sel_accs[f"selacc_at_cov_{cov}"] = float(np.mean([c for _, c in items[:k]]))

    # AUACC on mode-vote ranking
    accs_at_k = np.cumsum([c for _, c in items]) / np.arange(1, n + 1)
    aucc_mode = float(np.trapz(accs_at_k, np.arange(1, n + 1) / n))

    # Pass@K curve
    K = int(df["K"].iloc[0])
    pass_at = compute_pass_at_k(rollouts, [k for k in [1, 4, 8, 16, K] if k <= K])

    # Threshold recovery: under EV-optimal policy, model should attempt iff p(correct|attempt) > threshold.
    # Estimate empirical p(correct|attempted) per problem; aggregate.
    attempted_mask = df["abst_rate"] < 1.0  # at least one non-abstained sample
    if attempted_mask.sum() > 0:
        p_correct_given_attempted = float(
            df.loc[attempted_mask, "p_hat"].sum() / (1 - df.loc[attempted_mask, "abst_rate"]).sum()
            if (1 - df.loc[attempted_mask, "abst_rate"]).sum() > 0 else 0.0
        )
    else:
        p_correct_given_attempted = 0.0

    # Oracle-adjusted selectivity: how much of the latent capability (Pass@K) does the model
    # selectively expose via majority-vote? Ratio: auacc_mode_vote / pass_at_K.
    pass_at_K = pass_at.get(K, 1.0)
    oracle_adj_selectivity = float(aucc_mode / pass_at_K) if pass_at_K > 0 else 0.0

    return {
        "n_problems": n,
        "K": K,
        "accuracy_any": float(correct_any.mean()),
        "accuracy_mode_vote": float(mode_correct.mean()),
        "abstention_rate_mean": float(df["abst_rate"].mean()),
        "abstention_rate_all_K": float((df["abst_rate"] == 1.0).mean()),
        "ece_passatk_vs_any": ece(p_hat, correct_any),
        "ece_modeconf_vs_modecorrect": ece(mode_conf, mode_correct),
        "selective_accuracy": sel_accs,
        "auacc_mode_vote": aucc_mode,
        "oracle_adj_selectivity": oracle_adj_selectivity,
        "reliability_bins": reliability_bins(rollouts),
        "pass_at_k": pass_at,
        "ans_entropy_mean": float(df["ans_entropy"].mean()),
        "mean_chars_per_completion": float(df["mean_tokens"].mean()),
        "p_correct_given_attempted": p_correct_given_attempted,
        "rescore_changes": rescore_changes,
    }


# EV-implied thresholds per condition: p_attempt iff p(correct) > (beta+abstain)/(alpha+beta).
EV_THRESHOLDS = {
    "S0": 0.50, "S1": 0.67, "S2": 0.80, "S3": 0.91, "S4": 1.00,
    "C1": 0.60, "C2": 0.75,
}


def reliability_bins(rollouts: list[dict], n_bins: int = 10) -> list[dict]:
    """Per-bin (count, mean confidence, mean accuracy) for the corrected-ECE definition.

    Bins by mode-vote confidence (fraction of K samples matching the modal answer).
    Empty bins are omitted. Useful for plotting reliability diagrams.
    """
    df = per_problem_metrics(rollouts)
    conf = df["mode_conf"].to_numpy()
    correct = df["mode_correct"].astype(int).to_numpy()
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        out.append({
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "count": int(mask.sum()),
            "bin_conf_mean": float(conf[mask].mean()),
            "bin_accuracy": float(correct[mask].mean()),
            "gap": float(abs(conf[mask].mean() - correct[mask].mean())),
        })
    return out


def cross_condition_jaccard(model: str, conditions: list[str], dtype: str, bench: str,
                            out_dir: Path, abst_threshold: float = 0.5) -> dict:
    """Jaccard similarity matrix on per-problem "is abstained" masks across conditions.

    A problem is "abstained" if abst_rate >= abst_threshold (default: majority abstained).
    Tests whether conditions agree on which problems trigger abstention.
    """
    masks = {}
    for cond in conditions:
        rpath = out_dir / f"{model}_{cond}_{dtype}_{bench}_rollouts.jsonl"
        if not rpath.exists():
            continue
        rollouts = load_rollouts(rpath)
        df = per_problem_metrics(rollouts)
        df = df.sort_values("problem_id").reset_index(drop=True)
        masks[cond] = (df["abst_rate"] >= abst_threshold).to_numpy()
    if len(masks) < 2:
        return {}
    conds = list(masks.keys())
    pids_count = len(next(iter(masks.values())))
    jaccard = {}
    for a in conds:
        jaccard[a] = {}
        for b in conds:
            if a == b:
                jaccard[a][b] = 1.0
                continue
            inter = (masks[a] & masks[b]).sum()
            union = (masks[a] | masks[b]).sum()
            jaccard[a][b] = float(inter / union) if union > 0 else 0.0
    # also report fraction abstained per condition for context
    per_cond_rates = {c: float(masks[c].mean()) for c in conds}
    return {
        "n_problems": pids_count,
        "abst_threshold": abst_threshold,
        "fraction_majority_abstained": per_cond_rates,
        "jaccard": jaccard,
    }


def cross_condition_summary(per_condition_bench: dict) -> dict:
    """Pull headline cross-condition metrics for the project write-up."""
    out: dict = {
        "ev_threshold_recovery": {},
        "discrimination_index_math": {},
        "capability_gap": {},
        "c2_vs_s2_deltas": {},
    }
    # EV-threshold recovery: empirical per-attempt acc vs configured EV threshold.
    for cond, ev in EV_THRESHOLDS.items():
        if cond not in per_condition_bench:
            continue
        ev_d = {}
        for bench in ["math_test", "gsm8k_test", "mmlu_stem", "truthful_qa_mc"]:
            x = per_condition_bench[cond].get(bench, {})
            if "missing" in x:
                continue
            acc_att = x.get("p_correct_given_attempted", 0.0)
            ev_d[bench] = {
                "ev_threshold": ev,
                "acc_given_attempted": acc_att,
                "shortfall": max(0.0, ev - acc_att),
                "meets_threshold": acc_att >= ev,
            }
        out["ev_threshold_recovery"][cond] = ev_d

    # C2-vs-S2 deltas on every metric we have.
    if "S2" in per_condition_bench and "C2" in per_condition_bench:
        for bench in ["math_test", "gsm8k_test", "mmlu_stem", "truthful_qa_mc", "aime_2024", "aime_2025"]:
            s2 = per_condition_bench["S2"].get(bench, {})
            c2 = per_condition_bench["C2"].get(bench, {})
            if "missing" in s2 or "missing" in c2:
                continue
            deltas = {}
            for m in ["accuracy_any", "accuracy_mode_vote", "abstention_rate_mean",
                      "ece_passatk_vs_any", "ece_modeconf_vs_modecorrect",
                      "auacc_mode_vote", "p_correct_given_attempted",
                      "ans_entropy_mean", "mean_chars_per_completion"]:
                if m in s2 and m in c2:
                    deltas[m] = {"S2": s2[m], "C2": c2[m], "C2_minus_S2": c2[m] - s2[m]}
            out["c2_vs_s2_deltas"][bench] = deltas
    return out


def analyse_math_by_level(model: str, cond: str, dtype: str, out_dir: Path) -> dict:
    """Hard-vs-easy abstention analysis on MATH-test using the `level` column."""
    rpath = out_dir / f"{model}_{cond}_{dtype}_math_test_rollouts.jsonl"
    if not rpath.exists():
        return {"missing": True}
    rollouts = load_rollouts(rpath)
    df = per_problem_metrics(rollouts)
    bench_df = load_benchmark("math_test")[["id", "level"]].rename(columns={"id": "problem_id"})
    bench_df["problem_id"] = bench_df["problem_id"].astype(str)
    df = df.merge(bench_df, on="problem_id", how="left")
    by_level = {}
    for lvl in sorted(df["level"].dropna().unique()):
        sub = df[df["level"] == lvl]
        by_level[str(lvl)] = {
            "n": int(len(sub)),
            "accuracy_any": float(sub["correct_any"].mean()),
            "abstention_rate": float(sub["abst_rate"].mean()),
            "mode_conf_mean": float(sub["mode_conf"].mean()),
        }
    return by_level


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2_5_1_5b")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--conditions", default="S0,S1,S2,S3,S4,C1,C2")
    p.add_argument("--benchmarks", default="math_test,gsm8k_test,aime_2024,aime_2025,mmlu_stem,truthful_qa_mc")
    p.add_argument("--eval-dir", default="outputs/eval")
    p.add_argument("--out", default="outputs/eval/analysis_summary.json")
    args = p.parse_args()

    eval_dir = Path(args.eval_dir)
    summary: dict = {"per_condition_bench": {}, "math_by_level": {}, "cross_condition": {}}

    for cond in args.conditions.split(","):
        cond_d: dict = {}
        for bench in args.benchmarks.split(","):
            cond_d[bench] = analyse_condition_benchmark(args.model, cond, args.dtype, bench, eval_dir)
        summary["per_condition_bench"][cond] = cond_d
        summary["math_by_level"][cond] = analyse_math_by_level(args.model, cond, args.dtype, eval_dir)
        # Discrimination index: abst(L5) / abst(L1) on MATH.
        bl = summary["math_by_level"][cond]
        if "1" in bl and "5" in bl and bl["1"].get("abstention_rate", 0) > 1e-6:
            summary.setdefault("discrimination_index_math", {})[cond] = (
                bl["5"]["abstention_rate"] / bl["1"]["abstention_rate"]
            )
        print(f"  {cond}: done")

    summary["cross_condition"] = cross_condition_summary(summary["per_condition_bench"])

    # Cross-condition Jaccard on abstention masks per benchmark.
    summary["jaccard_abstention"] = {}
    for bench in args.benchmarks.split(","):
        j = cross_condition_jaccard(args.model, args.conditions.split(","), args.dtype, bench, eval_dir)
        if j:
            summary["jaccard_abstention"][bench] = j

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"OK summary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
