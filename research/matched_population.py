#!/usr/bin/env python3
"""Matched-population, matched-compute selective-prediction comparison (the key publishability fix).

On the SAME set of problems (the Exp2 probe subset), compare per-problem selective signals:
  - probe (1 forward pass): mean OOF probe score over the problem's committed records
  - free logit confidence (1 pass): mean logit_max_prob
  - self-consistency at K in {2,4,8,16}: mode-vote agreement over the first K rollouts
Label = is the problem's self-consistency mode answer correct (from K16). Report AUROC per method
with bootstrap CIs over problems, and K* = smallest K at which SC AUROC matches the 1-pass probe.

No GPU. Reads research/data/probe_scores_{bench}.jsonl + research/data/uc_strip/K16_S0_{bench}.jsonl.
"""
import json, os
from collections import Counter, defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")


def auroc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None
    order = np.argsort(scores); ranks = np.empty(len(scores), float); s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1; i = j + 1
    n1 = labels.sum(); n0 = len(labels) - n1
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def boot_auroc_ci(scores, labels, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    n = len(scores); vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        a = auroc(scores[idx], labels[idx])
        if a is not None:
            vals.append(a)
    return (round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4))


def sc_per_pid(path, K):
    byp = defaultdict(list)
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line); byp[d["pid"]].append(d)
    conf, correct = {}, {}
    for pid, ss in byp.items():
        ss = sorted(ss, key=lambda r: r["si"])[:K]
        na = [s for s in ss if not s["a"]]
        if not na:
            continue
        ans, cnt = Counter(s["e"] for s in na).most_common(1)[0]
        conf[pid] = cnt / K
        correct[pid] = 1 if any(s["c"] for s in na if s["e"] == ans) else 0
    return conf, correct


def main():
    out = {}
    for bench in ["math_test", "gsm8k_test"]:
        ps = os.path.join(DATA, f"probe_scores_{bench}.jsonl")
        k16 = os.path.join(DATA, "uc_strip", f"K16_S0_{bench}.jsonl")
        if not (os.path.exists(ps) and os.path.exists(k16)):
            continue
        # probe + logit per pid (mean over committed records)
        probe_by, logit_by = defaultdict(list), defaultdict(list)
        with open(ps) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    probe_by[d["pid"]].append(d["score"]); logit_by[d["pid"]].append(d["lmp"])
        scK16, lblK16 = sc_per_pid(k16, 16)
        # common population = pids in probe subset that also have a K16 mode label
        pids = [p for p in probe_by if p in lblK16]
        label = np.array([lblK16[p] for p in pids])
        probe = np.array([np.mean(probe_by[p]) for p in pids])
        logit = np.array([np.mean(logit_by[p]) for p in pids])
        res = {"n_problems": len(pids), "mode_correct_rate": round(float(label.mean()), 4),
               "probe_auroc": round(auroc(probe, label), 4), "probe_ci": boot_auroc_ci(probe, label),
               "logit_auroc": round(auroc(logit, label), 4), "logit_ci": boot_auroc_ci(logit, label),
               "sc": {}}
        for K in [2, 4, 8, 16]:
            cK, lK = sc_per_pid(k16, K)
            sc = np.array([cK.get(p, 0.0) for p in pids])
            a = auroc(sc, label)
            res["sc"][f"K{K}"] = {"auroc": round(a, 4) if a else None, "ci": boot_auroc_ci(sc, label)}
        # K* on common population: interpolate where SC auroc crosses probe auroc
        Ks = [2, 4, 8, 16]; scs = [res["sc"][f"K{k}"]["auroc"] for k in Ks]
        kstar = None
        if scs[0] is not None and scs[0] >= res["probe_auroc"]:
            kstar = 2.0
        else:
            for i in range(1, len(Ks)):
                if scs[i - 1] is not None and scs[i] is not None and scs[i - 1] < res["probe_auroc"] <= scs[i]:
                    kstar = Ks[i - 1] + (Ks[i] - Ks[i - 1]) * (res["probe_auroc"] - scs[i - 1]) / (scs[i] - scs[i - 1])
        res["K_star_probe_equiv"] = round(kstar, 2) if kstar else ">16"
        out[bench] = res
    print(json.dumps(out, indent=2))
    json.dump(out, open(os.path.join(ROOT, "outputs", "matched_population.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
