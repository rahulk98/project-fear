#!/usr/bin/env python3
"""K-sample self-consistency selectivity, for the #1 K*(c) comparison vs the single-pass probe.

For S0 K=16 rollouts (stripped), compute per-problem mode-vote confidence at K in {2,4,8,16}
and the AUROC of that confidence for predicting mode-answer correctness. This is the
selective-prediction quality of self-consistency as a function of the sampling budget K,
to be contrasted with the single forward-pass probe (AUROC ~0.78-0.83) and the free logit
confidence (~0.59-0.64). At K=1 SC confidence is degenerate (no vote), so SC needs K>=2.

No GPU. Reads research/data/uc_strip/K16_S0_{bench}.jsonl.
"""
import json, os
from collections import Counter
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "uc_strip")


def auroc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # tie-aware Mann-Whitney
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    R_pos = ranks[labels == 1].sum()
    n_pos = len(pos); n_neg = len(neg)
    U = R_pos - n_pos * (n_pos + 1) / 2.0
    return float(U / (n_pos * n_neg))


def sc_auroc_at_K(path, K):
    byp = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            byp.setdefault(d["pid"], []).append(d)
    confs, corrects = [], []
    for pid, samples in byp.items():
        samples = sorted(samples, key=lambda r: r["si"])[:K]
        non_abst = [s for s in samples if not s["a"]]
        if not non_abst:
            continue
        counts = Counter(s["e"] for s in non_abst)
        mode_ans, mode_count = counts.most_common(1)[0]
        conf = mode_count / K
        mode_correct = 1 if any(s["c"] for s in non_abst if s["e"] == mode_ans) else 0
        confs.append(conf); corrects.append(mode_correct)
    return auroc(confs, corrects), len(confs), float(np.mean(corrects))


def main():
    out = {}
    for bench in ["math_test", "gsm8k_test"]:
        p = os.path.join(DATA, f"K16_S0_{bench}.jsonl")
        if not os.path.exists(p):
            continue
        out[bench] = {}
        for K in [2, 4, 8, 16]:
            a, n, mc = sc_auroc_at_K(p, K)
            out[bench][f"K{K}"] = {"sc_mode_conf_auroc": (round(a, 4) if a else None),
                                    "n_problems": n, "mode_correct_rate": round(mc, 4)}
    print(json.dumps(out, indent=2))
    json.dump(out, open(os.path.join(ROOT, "outputs", "sc_selectivity.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
