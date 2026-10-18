#!/usr/bin/env python3
"""Q7: S0 probe-guided / SC-guided abstention vs reward-trained abstention at MATCHED answer rate.

Answers: S0 GREEDY completions (outputs/eval/qwen2_5_1_5b_S0_float16_greedy_{bench}_rollouts.jsonl).
S0's own greedy abstentions count as abstained first; the selector then ranks the remaining answers.
Probe score: per-problem mean of the nested-CV, question-grouped OUT-OF-FOLD probe score (E1 bench json,
oof_nested) over that problem's probed sampled rollouts (committed rows). It is a problem-level confidence
from other samples of the same question, not from the greedy completion itself.
Population: problems that have >=1 scored probe row (cannot score others: no hidden states for them).
SC K=4: agreement = top-vote count / 4 over the first 4 non-abstained K16 samples (ties broken by fixed noise).
Conditions: S1,S2,S3,S4,C2 greedy with their natural abstention. Bootstrap: 2000 paired problem resamples;
within each resample the S0 selector's coverage = the condition's answer rate in that resample.
Runs on CPU. Usage: q7_probe_matched_coverage.py OUT.json
"""
import json, os, sys, collections
import numpy as np
REPO = os.environ.get("FEAR_REPO_DIR", ".")
E1 = os.environ.get("FEAR_WORK_DIR", "outputs/followups") + "/e1"
EV = os.environ.get("FEAR_EVAL_DIR", "outputs/eval") + "/qwen2_5_1_5b_%s_float16_greedy_%s_rollouts.jsonl"
K16 = REPO + "/research/data/uc_strip/K16_S0_%s.jsonl"
CONDS = ["S1", "S2", "S3", "S4", "C2"]
COVS = [0.5, 0.6, 0.7, 0.8, 0.9]
B = 2000


def greedy(c, b):
    r = {}
    for l in open(EV % (c, b)):
        d = json.loads(l); r[d["problem_id"]] = (0 if d["abstained"] else 1, 1 if (d["correct"] and not d["abstained"]) else 0)
    return r


def sc4(b):
    byp = collections.defaultdict(list)
    for l in open(K16 % b):
        d = json.loads(l); byp[d["pid"]].append(d)
    out = {}
    for p, ss in byp.items():
        ss = sorted(ss, key=lambda r: r["si"])[:4]
        na = [s["e"] for s in ss if not s["a"]]
        out[p] = collections.Counter(na).most_common(1)[0][1] / 4 if na else 0.0
    return out


def topk_acc(score, cand, corr, k):
    s = np.where(cand, score, -np.inf)
    idx = np.argsort(-s, kind="stable")[:k]
    return corr[idx].mean()


def run(b):
    bj = json.load(open(f"{E1}/bench_{b}.json"))
    pb = collections.defaultdict(list)
    for p, s in zip(bj["problem_id"], bj["oof_nested"]): pb[p].append(s)
    S0 = greedy("S0", b); C = {c: greedy(c, b) for c in CONDS}; sc = sc4(b)
    pids = [p for p in pb if p in S0 and all(p in C[c] for c in CONDS) and p in sc]
    n = len(pids); rng = np.random.default_rng(0)
    tb = rng.random(n) * 1e-6  # fixed tie-break noise
    probe = np.array([np.mean(pb[p]) for p in pids]) + tb
    scs = np.array([sc[p] for p in pids]) + tb
    cand = np.array([S0[p][0] for p in pids], bool); corr0 = np.array([S0[p][1] for p in pids], float)
    ans = {c: np.array([C[c][p][0] for p in pids], bool) for c in CONDS}
    cor = {c: np.array([C[c][p][1] for p in pids], float) for c in CONDS}
    res = {"n_problems": n, "s0_greedy_answer_rate": round(float(cand.mean()), 4),
           "s0_greedy_acc_when_answering": round(float(corr0[cand].mean()), 4)}
    res["conditions"] = {c: {"answer_rate": round(float(ans[c].mean()), 4),
                             "acc_when_answering": round(float(cor[c][ans[c]].sum() / ans[c].sum()), 4)} for c in CONDS}
    sel = {"probe": probe, "sc4": scs}
    k = lambda a: int(round(a * n))
    # point estimates, matched coverage
    res["matched"] = {}
    boot_idx = rng.integers(0, n, (B, n))
    for c in CONDS:
        a = ans[c].mean(); kk = k(a)
        r = {"coverage": round(float(a), 4), "cond_acc": res["conditions"][c]["acc_when_answering"],
             "s0_reachable": bool(kk <= cand.sum())}
        for name, sc_ in sel.items():
            if kk > cand.sum():
                r[name] = None; continue
            pt = topk_acc(sc_, cand, corr0, kk) - res["conditions"][c]["acc_when_answering"]
            d = []
            for ii in boot_idx:
                ai = ans[c][ii]; ki = int(round(ai.mean() * n)); ci = cand[ii]
                if ki == 0 or ki > ci.sum() or ai.sum() == 0: continue
                d.append(topk_acc(sc_[ii], ci, corr0[ii], ki) - cor[c][ii][ai].sum() / ai.sum())
            r[name] = {"s0_sel_acc": round(float(topk_acc(sc_, cand, corr0, kk)), 4), "diff": round(float(pt), 4),
                       "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                       "n_boot_used": len(d)}
        res["matched"][c] = r
    # risk-coverage curves (coverage over all problems)
    res["rc_curve"] = {}
    for name, sc_ in sel.items():
        row = {}
        for cv in COVS:
            kk = k(cv)
            if kk > cand.sum(): row[str(cv)] = None; continue
            v = [topk_acc(sc_[ii], cand[ii], corr0[ii], kk) for ii in boot_idx[:500] if int(cand[ii].sum()) >= kk]
            row[str(cv)] = {"acc": round(float(topk_acc(sc_, cand, corr0, kk)), 4),
                            "ci95_boot500": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]}
        res["rc_curve"][name] = row
    # sanity: probe AUROC for S0 greedy correctness among committed
    from scipy.stats import rankdata
    m = cand; y = corr0[m]; r = rankdata(probe[m]); n1 = y.sum(); n0 = len(y) - n1
    res["probe_auroc_vs_greedy_correct"] = round(float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), 4)
    r = rankdata(scs[m]); res["sc4_auroc_vs_greedy_correct"] = round(float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), 4)
    return res


out = {b: run(b) for b in ["math_test", "gsm8k_test"]}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
