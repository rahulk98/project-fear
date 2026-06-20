#!/usr/bin/env python3
"""Single-shot cost-sensitive utility: where punishment-oriented (FEAR) training wins.

ISOLATED research analysis (no GPU, no training, no interference with the live fleet).
Reads compact stripped rollouts in research/data/uc_strip/ and answers:

  Q1 (single-shot win): among ONE-SAMPLE (K=1, greedy) policies -- the realistic
      cost-efficient deployment regime -- is a punishment-trained policy the
      expected-utility-optimal choice under asymmetric error cost
          U(c) = P(correct) - c * P(wrong)   (abstain = 0) ?
      We show each FEAR level tiles a cost band and overtakes the symmetric
      baseline S0 above a benchmark-specific crossover cost c*.

  Q2 (compute fairness): results.md:932 dismisses the FEAR utility win because
      "thresholding S0/P0 confidence dominates". That post-hoc selective policy
      needs K samples per problem (self-consistency). We compute its best-tau
      utility U_thr(c;K) and the UTILITY-PER-SAMPLE U_thr/K, and compare to the
      single-shot FEAR utility (1 sample). FEAR delivers more utility per unit
      compute; the post-hoc baseline is undefined at K=1.

Outputs: research/outputs/utility_summary.json, research/figures/*.png
"""
import json, os, glob
from collections import Counter
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "uc_strip")
OUT = os.path.join(ROOT, "outputs"); FIG = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

GREEDY_CONDS = ["S0", "S1", "S2", "S3", "S4", "C1", "C2"]
FEAR_CONDS = ["S1", "S2", "S3", "S4"]          # punishment-dominant asymmetry
BENCHES = ["math_test", "gsm8k_test", "mmlu_stem", "truthful_qa_mc", "aime_2024", "aime_2025"]
COSTS = np.round(np.linspace(0.0, 5.0, 501), 4)


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def greedy_stats(cond, bench):
    """K=1 per-sample probabilities + per-problem correct/wrong arrays (paired by pid order)."""
    path = os.path.join(DATA, f"greedy_{cond}_{bench}.jsonl")
    if not os.path.exists(path):
        return None
    rows = load_rows(path)
    corr = np.array([r["c"] for r in rows], dtype=float)
    abst = np.array([r["a"] for r in rows], dtype=float)
    wrong = 1.0 - corr - abst
    return {"n": len(rows), "pc": corr.mean(), "pa": abst.mean(), "pw": wrong.mean(),
            "corr": corr, "wrong": wrong}


def util_curve(pc, pw):
    return pc - COSTS * pw


def boot_util_ci(corr, wrong, cost, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(corr)
    u = corr - cost * wrong
    idx = rng.integers(0, n, size=(B, n))
    means = u[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def first_crossover(cost_grid, u_base, u_other):
    """smallest cost where u_other >= u_base (returns None if never)."""
    diff = u_other - u_base
    sign = diff >= 0
    for i in range(1, len(cost_grid)):
        if (not sign[i - 1]) and sign[i]:
            return float(cost_grid[i])
    if sign[0]:
        return float(cost_grid[0])
    return None


def posthoc_threshold_utility(path, K_sub, costs):
    """Best-tau post-hoc selective utility using mode-vote over the first K_sub samples.

    Per problem: take first K_sub samples; mode answer over non-abstained; conf=mode_count/K_sub;
    answer iff conf>=tau (mode answer), else abstain. For each cost, sweep tau, take max utility.
    Returns dict cost-> (best_utility, best_tau). Utility per *decision*; spends K_sub samples/problem.
    """
    rows = load_rows(path)
    byp = {}
    for r in rows:
        byp.setdefault(r["pid"], []).append(r)
    # per-problem: (mode_conf, mode_correct) using first K_sub samples
    confs, mcorr = [], []
    for pid, samples in byp.items():
        samples = sorted(samples, key=lambda r: r["si"])[:K_sub]
        non_abst = [s for s in samples if not s["a"]]
        if not non_abst:
            confs.append(0.0); mcorr.append(0); continue
        ans_counts = Counter(s["e"] for s in non_abst)
        mode_ans, mode_count = ans_counts.most_common(1)[0]
        confs.append(mode_count / K_sub)
        # mode correct if any sample with the mode answer is scored correct
        mc = any(s["c"] for s in non_abst if s["e"] == mode_ans)
        mcorr.append(1 if mc else 0)
    confs = np.array(confs); mcorr = np.array(mcorr, dtype=float)
    taus = np.unique(np.concatenate([[0.0], confs, [1.01]]))
    out = {}
    for c in costs:
        best_u, best_t = 0.0, 1.01  # abstain-all floor = 0
        for t in taus:
            answered = confs >= t
            u = (mcorr * answered).sum() - c * ((1 - mcorr) * answered).sum()
            u = u / len(confs)
            if u > best_u:
                best_u, best_t = u, float(t)
        out[float(c)] = (float(best_u), best_t)
    return out


def main():
    summary = {"description": "single-shot cost-sensitive utility; FEAR vs symmetric + post-hoc threshold",
               "costs_min": 0.0, "costs_max": 5.0, "n_costs": len(COSTS), "benchmarks": {}}

    # ---- Q1: single-shot utility per benchmark ----
    for bench in BENCHES:
        stats = {c: greedy_stats(c, bench) for c in GREEDY_CONDS}
        stats = {c: s for c, s in stats.items() if s is not None}
        if "S0" not in stats:
            continue
        curves = {c: util_curve(s["pc"], s["pw"]) for c, s in stats.items()}
        # optimal single-shot policy at each cost
        cond_list = list(curves.keys())
        M = np.vstack([curves[c] for c in cond_list])
        opt_idx = M.argmax(axis=0)
        opt_cond = [cond_list[i] for i in opt_idx]
        # crossover cost: S0 -> best fear condition
        best_fear = max([c for c in FEAR_CONDS if c in curves],
                        key=lambda c: curves[c][-1], default=None)  # highest U at max cost
        cstar = first_crossover(COSTS, curves["S0"], curves[best_fear]) if best_fear else None
        # operating points
        ops = {c: {"pc": round(stats[c]["pc"], 4), "pw": round(stats[c]["pw"], 4),
                   "pa": round(stats[c]["pa"], 4)} for c in stats}
        # where the optimum switches away from S0 the first time
        switch_cost = None
        for j in range(len(COSTS)):
            if opt_cond[j] != "S0":
                switch_cost = float(COSTS[j]); break
        # bootstrap CI of the S0 vs best_fear gap at cost=3 (if defined)
        ci = {}
        if best_fear and 3.0 <= 5.0:
            ci_s0 = boot_util_ci(stats["S0"]["corr"], stats["S0"]["wrong"], 3.0)
            ci_bf = boot_util_ci(stats[best_fear]["corr"], stats[best_fear]["wrong"], 3.0)
            ci = {"cost": 3.0, "S0_U_ci": [round(x, 4) for x in ci_s0],
                  f"{best_fear}_U_ci": [round(x, 4) for x in ci_bf]}
        summary["benchmarks"][bench] = {
            "n_problems": stats["S0"]["n"],
            "operating_points_K1": ops,
            "best_fear_cond": best_fear,
            "crossover_cost_S0_to_bestfear": cstar,
            "first_cost_optimum_leaves_S0": switch_cost,
            "optimal_cond_at_costs": {str(round(float(COSTS[k]), 2)): opt_cond[k]
                                      for k in range(0, len(COSTS), 50)},
            "bootstrap_ci_cost3": ci,
        }

    # ---- Q2: compute-fairness vs post-hoc threshold (math + gsm8k) ----
    q2 = {}
    for bench in ["math_test", "gsm8k_test"]:
        stats = {c: greedy_stats(c, bench) for c in GREEDY_CONDS}
        best_fear = max(FEAR_CONDS, key=lambda c: util_curve(stats[c]["pc"], stats[c]["pw"])[-1])
        # single-shot reference utilities at sampled costs
        ss = {}
        for c in [1.0, 2.0, 3.0, 4.0]:
            i = int(np.argmin(np.abs(COSTS - c)))
            ss[c] = {
                "S0_K1": round(float(util_curve(stats["S0"]["pc"], stats["S0"]["pw"])[i]), 4),
                f"{best_fear}_K1": round(float(util_curve(stats[best_fear]["pc"], stats[best_fear]["pw"])[i]), 4),
                "best_singleshot_K1": round(float(max(util_curve(stats[cc]["pc"], stats[cc]["pw"])[i]
                                                      for cc in GREEDY_CONDS)), 4),
            }
        # post-hoc threshold baselines (need K samples)
        baselines = {}
        for name, fname, Ks in [("S0", f"K16_S0_{bench}.jsonl", [2, 4, 8, 16]),
                                ("P0", f"K32_P0_{bench}.jsonl", [2, 4, 8, 16, 32])]:
            p = os.path.join(DATA, fname)
            if not os.path.exists(p):
                continue
            per_K = {}
            for K in Ks:
                ph = posthoc_threshold_utility(p, K, [1.0, 2.0, 3.0, 4.0])
                per_K[K] = {str(c): {"U_thr": round(v[0], 4),
                                     "U_per_sample": round(v[0] / K, 4),
                                     "best_tau": round(v[1], 3)} for c, v in ph.items()}
            baselines[name] = per_K
        q2[bench] = {"best_fear_cond": best_fear, "single_shot": ss,
                     "posthoc_threshold": baselines}
    summary["compute_fairness"] = q2

    with open(os.path.join(OUT, "utility_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", os.path.join(OUT, "utility_summary.json"))

    # ---------- figures ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib unavailable, skipping figures:", e); return

    # fig1: U(cost) curves, math + gsm8k
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    palette = {"S0": "k", "S1": "#1f77b4", "S2": "#ff7f0e", "S3": "#d62728", "S4": "#9467bd",
               "C1": "#2ca02c", "C2": "#8c564b"}
    for ax, bench in zip(axes, ["math_test", "gsm8k_test"]):
        for c in GREEDY_CONDS:
            s = greedy_stats(c, bench)
            ax.plot(COSTS, util_curve(s["pc"], s["pw"]), label=c, color=palette[c],
                    lw=2 if c in ("S0",) else 1.4, ls="-" if c != "S0" else "--")
        ax.axhline(0, color="gray", lw=0.6)
        b = summary["benchmarks"][bench]
        if b["crossover_cost_S0_to_bestfear"]:
            ax.axvline(b["crossover_cost_S0_to_bestfear"], color="red", ls=":", lw=1)
            ax.text(b["crossover_cost_S0_to_bestfear"] + 0.05, ax.get_ylim()[1] * 0.9,
                    f"c*={b['crossover_cost_S0_to_bestfear']:.2f}", color="red", fontsize=9)
        ax.set_title(f"Single-shot utility U(c)=P(correct)-c*P(wrong)  [{bench}]")
        ax.set_xlabel("error cost c (abstain=0, correct=+1)"); ax.set_ylabel("expected utility / problem")
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_single_shot_utility.png"), dpi=130)
    print("wrote fig1")

    # fig2: utility-per-sample vs K at cost=3 (math)
    bench = "math_test"; cost = 3.0
    stats = {c: greedy_stats(c, bench) for c in GREEDY_CONDS}
    i = int(np.argmin(np.abs(COSTS - cost)))
    bf = q2[bench]["best_fear_cond"]
    fig2, ax = plt.subplots(figsize=(7.5, 5))
    for name, color in [("S0", "#1f77b4"), ("P0", "#ff7f0e")]:
        if name in q2[bench]["posthoc_threshold"]:
            d = q2[bench]["posthoc_threshold"][name]
            Ks = sorted(int(k) for k in d.keys())
            ups = [d[K][str(cost)]["U_per_sample"] for K in Ks]
            ax.plot(Ks, ups, "o-", color=color, label=f"post-hoc threshold {name} (U/sample)")
    ax.axhline(util_curve(stats[bf]["pc"], stats[bf]["pw"])[i], color="red", ls="--",
               label=f"single-shot FEAR {bf} (1 sample)")
    ax.axhline(util_curve(stats["S0"]["pc"], stats["S0"]["pw"])[i], color="k", ls=":",
               label="single-shot S0 (1 sample)")
    ax.set_xlabel("K samples per problem (self-consistency)"); ax.set_ylabel(f"utility per generated sample (cost c={cost})")
    ax.set_title(f"Compute-normalized utility [{bench}, c={cost}]\nFEAR needs 1 sample; post-hoc threshold needs K")
    ax.legend(fontsize=9)
    fig2.tight_layout(); fig2.savefig(os.path.join(FIG, "fig2_utility_per_sample.png"), dpi=130)
    print("wrote fig2")

    # fig3: crossover cost c* per benchmark
    fig3, ax = plt.subplots(figsize=(8, 5))
    bb = [b for b in BENCHES if b in summary["benchmarks"]]
    cs = [summary["benchmarks"][b]["crossover_cost_S0_to_bestfear"] or np.nan for b in bb]
    ax.bar(range(len(bb)), cs, color="#d62728")
    ax.set_xticks(range(len(bb))); ax.set_xticklabels(bb, rotation=30, ha="right")
    ax.set_ylabel("crossover cost c* (S0 -> best FEAR)")
    ax.set_title("Cost at which single-shot FEAR overtakes S0\n(lower = FEAR wins even at modest error cost)")
    for k, v in enumerate(cs):
        if not np.isnan(v):
            ax.text(k, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
    fig3.tight_layout(); fig3.savefig(os.path.join(FIG, "fig3_crossover_by_benchmark.png"), dpi=130)
    print("wrote fig3")


if __name__ == "__main__":
    main()
