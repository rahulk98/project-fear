#!/usr/bin/env python3
"""VOI-GRPO offline pre-check (pivot #2 de-risk AND a Goal-1 test).

The Finding-O failure mode: under GRPO z-score normalization, a scalar reward that is constant
within a group (e.g. all-correct or all-wrong groups) collapses to zero advantage. VOI-GRPO claims
a value-of-information signal carries WITHIN-GROUP variance that survives. This checks, on cached
rollouts, whether that is true and how much pivotal signal exists.

Per group (committed rollouts, grouped by problem):
  - classify all-correct / all-wrong / mixed
  - gold = extracted answer of correct rollouts (math_verify already applied via the 'c' flag)
  - answer-flip: does leave-one-out change the plurality answer? (pivotal mass)
  - VOI_i = change in group's fraction-on-gold when rollout i is removed (LOO); report within-group
    std of VOI by group type (the z-score-survival precondition)

No GPU. Reads research/data/uc_strip/{K16_S0,K32_P0}_{math_test,gsm8k_test}.jsonl.
"""
import json, os
from collections import Counter, defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "uc_strip")


def analyze(path):
    byp = defaultdict(list)
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line); byp[d["pid"]].append(d)
    n_all_corr = n_all_wrong = n_mixed = n_groups = 0
    flips = 0
    voi_std_mixed, voi_std_unanimous = [], []
    pivotal_rollout_frac = []
    for pid, ss in byp.items():
        com = [s for s in ss if not s["a"]]
        K = len(com)
        if K < 2:
            continue
        n_groups += 1
        cs = np.array([s["c"] for s in com])
        ans = [s["e"] for s in com]
        if cs.all():
            n_all_corr += 1; gtype = "unanimous"
        elif not cs.any():
            n_all_wrong += 1; gtype = "unanimous"
        else:
            n_mixed += 1; gtype = "mixed"
        # gold = modal extracted among correct rollouts (if any)
        corr_ans = [ans[i] for i in range(K) if cs[i] == 1]
        gold = Counter(corr_ans).most_common(1)[0][0] if corr_ans else None
        # plurality + LOO flip
        full_plur = Counter(ans).most_common(1)[0][0]
        flipped = False; pivotal = 0
        for i in range(K):
            loo = Counter(ans[:i] + ans[i + 1:]).most_common(1)[0][0]
            if loo != full_plur:
                flipped = True; pivotal += 1
        if flipped:
            flips += 1
        pivotal_rollout_frac.append(pivotal / K)
        # VOI_i toward gold (fraction-on-gold LOO change); only if gold known
        if gold is not None:
            on_gold = np.array([1.0 if ans[i] == gold else 0.0 for i in range(K)])
            base = on_gold.mean()
            voi = np.array([base - (on_gold.sum() - on_gold[i]) / (K - 1) for i in range(K)])
            (voi_std_mixed if gtype == "mixed" else voi_std_unanimous).append(float(voi.std()))
    return {
        "file": path.split("/")[-1], "n_groups": n_groups,
        "pct_all_correct": round(100 * n_all_corr / n_groups, 1),
        "pct_all_wrong": round(100 * n_all_wrong / n_groups, 1),
        "pct_mixed": round(100 * n_mixed / n_groups, 1),
        "loo_plurality_flip_rate": round(flips / n_groups, 4),
        "mean_pivotal_rollout_frac": round(float(np.mean(pivotal_rollout_frac)), 4),
        "voi_within_group_std_mixed": round(float(np.mean(voi_std_mixed)), 4) if voi_std_mixed else None,
        "voi_within_group_std_unanimous": round(float(np.mean(voi_std_unanimous)), 4) if voi_std_unanimous else None,
    }


def main():
    files = ["K16_S0_math_test.jsonl", "K16_S0_gsm8k_test.jsonl",
             "K32_P0_math_test.jsonl", "K32_P0_gsm8k_test.jsonl"]
    out = [analyze(os.path.join(DATA, f)) for f in files if os.path.exists(os.path.join(DATA, f))]
    print(json.dumps(out, indent=2))
    json.dump(out, open(os.path.join(ROOT, "outputs", "voi_precheck.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
