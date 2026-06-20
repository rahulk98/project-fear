#!/usr/bin/env python3
"""Probe-gated adaptive test-time compute (the probe-reframe downstream task).

The Exp2 linear probe reads a single forward pass and predicts whether a model's
answer is correct (AUROC ~0.785 math / ~0.72 gsm8k -- it beats self-consistency
at K=2 and matches it at K~3). This script converts that selective-prediction
signal into a deployable WIN: use the probe as a per-problem difficulty estimate
to ALLOCATE test-time sampling budget. Easy problems (high probe confidence) get
1 sample; hard problems (low probe confidence) get K samples + majority vote.

We trace the accuracy-vs-average-compute Pareto frontier of this probe-routed
policy and compare it to:
  - uniform self-consistency at fixed K (the standard baseline),
  - logit-confidence routing (free 1-pass signal: mean max-token-prob),
  - random routing (null: routing carries no information),
  - oracle routing (ceiling: escalate exactly the 1-sample failures).

A positive result = the probe-routed frontier lies ABOVE uniform self-consistency
and above logit routing at matched average compute, i.e. the probe reaches a target
accuracy with strictly fewer samples than either baseline.

No GPU, no training, no inference. Pure numpy on existing local artifacts:
  research/data/probe_scores_{bench}.jsonl        (per-record OOF probe score + logit conf)
  research/data/uc_strip/K16_S0_{bench}.jsonl     (16-sample self-consistency rollouts)

Usage: python research/probe_adaptive_compute.py [out_json]
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
KMAX = 16
BENCHES = ["math_test", "gsm8k_test"]
# Fixed budgets at which we report a head-to-head accuracy comparison.
REPORT_BUDGETS = [2.0, 3.0, 4.0, 6.0, 8.0]
RANDOM_SEEDS = list(range(40))  # null-routing is averaged over these


# ----------------------------------------------------------------------------- data
def load_probe(bench):
    """pid -> {'scores': [...], 'lmp': [...]} from the OOF probe-score dump.

    y in this file: 1 = the single generation was correct, 0 = incorrect.
    score = OOF probe P(correct); lmp = free logit max-prob confidence.
    """
    by = defaultdict(lambda: {"scores": [], "lmp": [], "si": []})
    path = os.path.join(DATA, f"probe_scores_{bench}.jsonl")
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            by[d["pid"]]["scores"].append(float(d["score"]))
            by[d["pid"]]["lmp"].append(float(d["lmp"]))
            by[d["pid"]]["si"].append(int(d.get("si", -1)))
    return by


def load_k16(bench):
    """pid -> list of (si, c, a, e) sorted by si, from the K=16 SC rollouts."""
    by = defaultdict(list)
    path = os.path.join(DATA, "uc_strip", f"K16_S0_{bench}.jsonl")
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            by[d["pid"]].append((int(d["si"]), int(d["c"]), int(d["a"]), d.get("e")))
    for pid in by:
        by[pid].sort(key=lambda r: r[0])
    return by


def mode_correct_at(samples, m):
    """Majority-vote correctness over the first m samples (abstentions don't vote).

    samples: list of (si, c, a, e) sorted by si. Returns 1 if the modal committed
    answer is correct, else 0 (all-abstain -> 0). Mirrors research/sc_selectivity.py.
    """
    first = samples[:m]
    committed = [(c, e) for (_si, c, a, e) in first if not a and e is not None]
    if not committed:
        return 0
    counts = Counter(e for (_c, e) in committed)
    modal = counts.most_common(1)[0][0]
    return 1 if any(c for (c, e) in committed if e == modal) else 0


# ----------------------------------------------------------------------------- core
def build_problem_table(bench):
    """Return aligned arrays over the matched population (probe pids that have K16).

    mc[i, m-1] = majority-vote correctness of problem i at budget m (m=1..KMAX).
    probe_mean / probe_first / logit / null fields are per-problem routing signals.
    """
    probe = load_probe(bench)
    k16 = load_k16(bench)
    pids = [p for p in probe if p in k16]
    pids.sort()

    n = len(pids)
    mc = np.zeros((n, KMAX), dtype=float)
    probe_mean = np.zeros(n)
    probe_first = np.zeros(n)
    logit = np.zeros(n)
    for i, pid in enumerate(pids):
        samples = k16[pid]
        for m in range(1, KMAX + 1):
            mc[i, m - 1] = mode_correct_at(samples, m)
        sc = probe[pid]
        # CAVEAT (codex P2): probe_mean/logit average over ALL committed K16 records
        # and probe_first uses the lowest-si committed record (not strictly sample 0
        # when the first rollout abstained). So at budgets < K the router peeks at
        # confidence from samples it has not "paid" for -- an OPTIMISTIC signal for
        # the probe. The two-tier result is nonetheless NEGATIVE (probe ties random
        # and loses to uniform SC), so the negative conclusion holds a fortiori: even
        # given unfair extra information the probe is not a useful compute allocator.
        probe_mean[i] = float(np.mean(sc["scores"]))
        order = int(np.argmin(sc["si"])) if sc["si"] else 0
        probe_first[i] = float(sc["scores"][order])
        logit[i] = float(np.mean(sc["lmp"]))
    return {"pids": pids, "n": n, "mc": mc,
            "probe_mean": probe_mean, "probe_first": probe_first, "logit": logit}


def uniform_sc_curve(mc):
    """Standard baseline: every problem gets K samples. (compute=K, acc=mean mc[:,K-1])."""
    return [(float(K), float(mc[:, K - 1].mean())) for K in range(1, KMAX + 1)]


def two_tier_curve(mc, conf, k_high, n_points=None):
    """Escalate the least-confident fraction to k_high samples; rest get 1 sample.

    Returns list of (avg_compute, accuracy) as the escalated fraction grows 0->1,
    sorted by avg_compute. Lower conf -> escalate first.
    """
    n = len(conf)
    cheap = mc[:, 0]            # 1-sample correctness
    exp = mc[:, k_high - 1]     # k_high-sample correctness
    order = np.argsort(conf, kind="stable")  # ascending: least confident first
    cheap_o = cheap[order]
    exp_o = exp[order]
    # prefix sums so escalating the first j is O(1)
    csum_exp = np.concatenate([[0.0], np.cumsum(exp_o)])
    csum_cheap = np.concatenate([[0.0], np.cumsum(cheap_o)])
    total_cheap = csum_cheap[-1]
    curve = []
    js = range(0, n + 1) if n_points is None else \
        sorted(set(int(round(t)) for t in np.linspace(0, n, n_points)))
    for j in js:  # escalate first j (least confident)
        acc = (csum_exp[j] + (total_cheap - csum_cheap[j])) / n
        comp = (j * k_high + (n - j) * 1) / n
        curve.append((float(comp), float(acc)))
    curve.sort()
    return curve


def acc_at_budget(curve, budget):
    """Linear-interpolate a (compute, acc) curve at a target average compute."""
    xs = [c for c, _ in curve]
    ys = [a for _, a in curve]
    if budget <= xs[0]:
        return ys[0]
    if budget >= xs[-1]:
        return ys[-1]
    return float(np.interp(budget, xs, ys))


def compute_to_reach(curve, target_acc):
    """Smallest average compute on the curve that reaches target_acc (None if never)."""
    for c, a in curve:
        if a >= target_acc - 1e-9:
            return float(c)
    return None


def curve_auc(curve, lo=1.0, hi=8.0):
    """Mean accuracy over the compute window [lo,hi] (area / width). Higher = better."""
    xs = np.array([c for c, _ in curve])
    ys = np.array([a for _, a in curve])
    grid = np.linspace(lo, hi, 200)
    vals = np.interp(grid, xs, ys)
    return float(vals.mean())


# ----------------------------------------------------------------------------- stats
def bootstrap_budget_gap(mc, conf_a, conf_b, k_high, budget, kind_a, kind_b,
                         B=2000, seed=0):
    """Paired bootstrap over problems: accuracy(policy_a) - accuracy(policy_b) at a
    matched average-compute budget. kind in {'two_tier','uniform'}.

    For 'uniform', budget is the (integer-rounded) K; for 'two_tier', conf routes.
    Returns (mean_gap, lo, hi, frac_gap_gt_0).
    """
    rng = np.random.default_rng(seed)
    n = len(mc)
    gaps = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        mcb = mc[idx]
        a = _policy_acc(mcb, conf_a[idx] if conf_a is not None else None,
                        k_high, budget, kind_a)
        b = _policy_acc(mcb, conf_b[idx] if conf_b is not None else None,
                        k_high, budget, kind_b)
        gaps.append(a - b)
    gaps = np.array(gaps)
    return (float(gaps.mean()), float(np.percentile(gaps, 2.5)),
            float(np.percentile(gaps, 97.5)), float((gaps > 0).mean()))


def _policy_acc(mc, conf, k_high, budget, kind):
    if kind == "uniform":
        K = int(round(budget))
        K = max(1, min(KMAX, K))
        return float(mc[:, K - 1].mean())
    if kind == "two_tier":
        curve = two_tier_curve(mc, conf, k_high)
        return acc_at_budget(curve, budget)
    raise ValueError(kind)


# ----------------------------------------------------------------------------- main
def analyze(bench):
    t = build_problem_table(bench)
    mc, n = t["mc"], t["n"]
    pass1 = float(mc[:, 0].mean())
    scK = uniform_sc_curve(mc)
    acc_sc16 = scK[-1][1]

    # routing signals (higher = more confident -> NOT escalated)
    signals = {
        "probe_mean": t["probe_mean"],
        "probe_first": t["probe_first"],
        "logit": t["logit"],
        "oracle": mc[:, 0],  # confident == 1-sample already correct
    }
    rng = np.random.default_rng(12345)
    # null: average several random routings into one representative curve
    null_curves = [two_tier_curve(mc, rng.random(n), KMAX) for _ in RANDOM_SEEDS]

    k_high = KMAX
    curves = {name: two_tier_curve(mc, sig, k_high) for name, sig in signals.items()}
    # representative random curve = pointwise mean over seeds on a shared compute grid
    grid = np.linspace(1.0, float(k_high), 60)
    null_acc = np.mean([[acc_at_budget(c, g) for g in grid] for c in null_curves], axis=0)
    curves["random"] = list(zip([float(x) for x in grid], [float(y) for y in null_acc]))

    # headline 1: compute needed to match uniform SC@16 accuracy
    reach = {name: compute_to_reach(curve, acc_sc16) for name, curve in curves.items()}
    reach["uniform_sc"] = 16.0

    # headline 2: accuracy at fixed average-compute budgets
    at_budget = {}
    for b in REPORT_BUDGETS:
        row = {name: round(acc_at_budget(curve, b), 4) for name, curve in curves.items()}
        row["uniform_sc"] = round(_policy_acc(mc, None, k_high, b, "uniform"), 4)
        at_budget[f"B{b:g}"] = row

    # headline 3: area under accuracy-compute curve in [1,8]
    auc = {name: round(curve_auc(curve, 1.0, 8.0), 4) for name, curve in curves.items()}
    auc["uniform_sc"] = round(curve_auc(scK, 1.0, 8.0), 4)

    # significance: probe_mean vs uniform_sc and vs logit at the budgets, paired bootstrap
    sig_tests = {}
    for b in REPORT_BUDGETS:
        g1 = bootstrap_budget_gap(mc, t["probe_mean"], None, k_high, b,
                                  "two_tier", "uniform", seed=int(b * 7))
        g2 = bootstrap_budget_gap(mc, t["probe_mean"], t["logit"], k_high, b,
                                  "two_tier", "two_tier", seed=int(b * 13))
        sig_tests[f"B{b:g}"] = {
            "probe_minus_uniformSC": {"gap": round(g1[0], 4), "ci": [round(g1[1], 4), round(g1[2], 4)],
                                      "p_gap_gt_0": round(g1[3], 4)},
            "probe_minus_logitRoute": {"gap": round(g2[0], 4), "ci": [round(g2[1], 4), round(g2[2], 4)],
                                       "p_gap_gt_0": round(g2[3], 4)},
        }

    return {
        "bench": bench,
        "n_problems": n,
        "pass1_acc": round(pass1, 4),
        "uniform_sc_acc": {f"K{K}": round(a, 4) for (K, a) in scK},
        "acc_sc16": round(acc_sc16, 4),
        "k_high": k_high,
        "compute_to_match_sc16": {k: (round(v, 3) if v is not None else None)
                                  for k, v in reach.items()},
        "acc_at_budget": at_budget,
        "auc_1to8": auc,
        "significance_paired_bootstrap": sig_tests,
        "curves": {name: [[round(c, 4), round(a, 4)] for c, a in curve]
                   for name, curve in curves.items()},
    }


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(ROOT, "outputs", "probe_adaptive_compute.json")
    out = {}
    for bench in BENCHES:
        try:
            out[bench] = analyze(bench)
        except FileNotFoundError as e:
            out[bench] = {"error": str(e)}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # human summary
    for bench in BENCHES:
        r = out.get(bench, {})
        if "error" in r:
            print(f"[{bench}] {r['error']}")
            continue
        print(f"\n=== {bench}  (n={r['n_problems']}, pass@1={r['pass1_acc']}, "
              f"SC@16={r['acc_sc16']}) ===")
        print("  compute to MATCH uniform SC@16 accuracy:")
        for k, v in r["compute_to_match_sc16"].items():
            print(f"    {k:<14} {v}")
        print("  accuracy at fixed avg-compute budget:")
        hdr = sorted(next(iter(r["acc_at_budget"].values())).keys())
        print("    budget | " + " ".join(f"{h:>11}" for h in hdr))
        for b, row in r["acc_at_budget"].items():
            print(f"    {b:<6} | " + " ".join(f"{row[h]:>11}" for h in hdr))
        print("  paired bootstrap (probe_mean two-tier vs baselines):")
        for b, row in r["significance_paired_bootstrap"].items():
            u = row["probe_minus_uniformSC"]; l = row["probe_minus_logitRoute"]
            print(f"    {b}: vs uniformSC gap={u['gap']:+.4f} CI{u['ci']} P(>0)={u['p_gap_gt_0']}"
                  f" | vs logitRoute gap={l['gap']:+.4f} CI{l['ci']} P(>0)={l['p_gap_gt_0']}")
    print(f"\nwrote -> {out_path}")


if __name__ == "__main__":
    main()
