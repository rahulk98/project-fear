#!/usr/bin/env python3
"""Probe-as-verifier downstream tasks (the probe-reframe positive test).

The Exp2 linear probe scores a single generation's hidden state for P(correct).
This tests whether that per-sample signal yields a deployable WIN over plain
self-consistency at MATCHED compute, via three aggregation methods:

  1. probe-reranked best-of-N : among the first K samples, submit the answer of the
     single highest-probe sample (probe used as a verifier / reranker).
  2. probe-weighted vote      : majority vote where each sample's vote is weighted
     by its probe P(correct) instead of 1 (down-weights likely-wrong samples).
  3. probe early-stop         : sample sequentially, stop at the first sample whose
     probe >= tau and submit it; trace accuracy vs the AVERAGE samples drawn.

Baselines at the same K: plain majority vote (standard self-consistency), the free
logit-confidence reranker / weighted vote, a random reranker (null), and the
oracle best-of-K (ceiling = any correct among the K).

A positive result = a probe method beats plain majority vote at matched K, or
reaches majority-vote@16 accuracy at strictly fewer samples (compute saving).

Needs per-sample probe scores for ALL K samples, so it consumes a fresh all-layers
extraction (research launch `probe_bo16`): outputs/probing/qwen2_5_1_5b_S0bo16_*.
Probe is trained on a disjoint TRAIN pid split and evaluated only on TEST pids.

No training, no GPU inference (sklearn logreg only). Run on alias (npz lives there).
Usage: probe_bestofn.py <bench> [npz_dir] [rollout_path] [out_json]
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

KS = [1, 2, 3, 4, 6, 8, 12, 16]
LAYERS = list(range(8, 25))
TEST_FRAC = 0.4
SEED = 0


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


# ---------------------------------------------------------------------------- load
def load_rollout(path):
    """(pid,si) -> dict(extracted, correct, abstained)."""
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out[(d["problem_id"], int(d["sample_idx"]))] = {
                "ext": d.get("extracted"),
                "correct": int(bool(d.get("correct"))),
                "abst": int(bool(d.get("abstained"))),
            }
    return out


def load_npz(npz_dir, bench):
    stem = os.path.join(npz_dir, f"qwen2_5_1_5b_S0bo16_float32_{bench}_hidden")
    d = np.load(stem + ".npz")
    H, y = d["H"], d["y"]
    meta = json.load(open(stem + ".meta.json"))
    items = meta["items"]
    pid = np.array([it["problem_id"] for it in items])
    si = np.array([int(it.get("sample_idx", -1)) for it in items])
    lmp = np.array([float(it.get("logit_max_prob", np.nan)) for it in items])
    return H, y, pid, si, lmp


# ---------------------------------------------------------------------------- probe
def fit_probe_scores(H, correct, committed, pid, train_mask):
    """Pick best layer by GroupKFold AUROC on TRAIN committed samples; train final
    probe there; return per-record probe P(correct) for ALL records + diagnostics.
    """
    tr = train_mask & committed
    yc = correct[tr]
    groups = pid[tr]
    best = {"auc": -1, "L": None}
    for L in LAYERS:
        X = H[tr][:, L, :]
        if len(np.unique(yc)) < 2:
            continue
        oof = np.zeros(len(yc))
        for a, b in GroupKFold(n_splits=5).split(X, yc, groups):
            sc = StandardScaler().fit(X[a])
            clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
            clf.fit(sc.transform(X[a]), yc[a])
            oof[b] = clf.predict_proba(sc.transform(X[b]))[:, 1]
        auc = roc_auc_score(yc, oof)
        if auc > best["auc"]:
            best = {"auc": float(auc), "L": int(L)}
    L = best["L"]
    sc = StandardScaler().fit(H[tr][:, L, :])
    clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
    clf.fit(sc.transform(H[tr][:, L, :]), yc)
    probe_all = clf.predict_proba(sc.transform(H[:, L, :]))[:, 1]
    return probe_all, best


# ---------------------------------------------------------------------------- agg
def per_problem(test_pids, byp):
    """byp[pid] = list of per-sample dicts sorted by si (committed, ext, correct,
    probe, logit). Yields (pid, samples)."""
    for pid in test_pids:
        yield pid, byp[pid]


def _committed_first_k(samples, k):
    return [s for s in samples[:k] if s["committed"]]


def acc_majority(samples, k):
    c = _committed_first_k(samples, k)
    if not c:
        return 0
    w = defaultdict(int)
    for s in c:
        w[s["ext"]] += 1
    modal = max(w, key=w.get)
    return 1 if any(s["correct"] for s in c if s["ext"] == modal) else 0


def acc_weighted(samples, k, key):
    c = _committed_first_k(samples, k)
    if not c:
        return 0
    w = defaultdict(float)
    for s in c:
        w[s["ext"]] += s[key]
    modal = max(w, key=w.get)
    return 1 if any(s["correct"] for s in c if s["ext"] == modal) else 0


def acc_rerank(samples, k, key):
    c = _committed_first_k(samples, k)
    if not c:
        return 0
    best = max(c, key=lambda s: s[key])
    return best["correct"]


def acc_oracle(samples, k):
    c = _committed_first_k(samples, k)
    return 1 if any(s["correct"] for s in c) else 0


def acc_rerank_random(samples, k, rng):
    c = _committed_first_k(samples, k)
    if not c:
        return 0
    return c[rng.integers(0, len(c))]["correct"]


def early_stop(samples, tau):
    """Sequential: stop at first committed sample with probe>=tau, submit it.
    Fall back to best-probe-of-16 if none reach tau. Returns (correct, n_drawn)."""
    drawn = 0
    for s in samples:
        drawn += 1
        if s["committed"] and s["probe"] >= tau:
            return s["correct"], drawn
    c = [s for s in samples if s["committed"]]
    if not c:
        return 0, drawn
    best = max(c, key=lambda s: s["probe"])
    return best["correct"], drawn


# ---------------------------------------------------------------------------- main
def analyze(bench, npz_dir, rollout_path):
    H, y, pid, si, lmp = load_npz(npz_dir, bench)
    roll = load_rollout(rollout_path)
    # correctness/abstain/extracted aligned to NPZ rows via (pid,si)
    correct = np.zeros(len(pid), int)
    abst = np.zeros(len(pid), int)
    ext = np.empty(len(pid), object)
    keep = np.ones(len(pid), bool)
    for i in range(len(pid)):
        r = roll.get((pid[i], int(si[i])))
        if r is None:
            keep[i] = False
            continue
        correct[i] = r["correct"]; abst[i] = r["abst"]; ext[i] = r["ext"]
    committed = (abst == 0) & keep & np.array([e is not None for e in ext])

    # held-out split by problem
    uniq = sorted(set(pid[keep]))
    rng0 = np.random.default_rng(SEED)
    perm = rng0.permutation(len(uniq))
    n_test = int(round(len(uniq) * TEST_FRAC))
    test_set = set(uniq[i] for i in perm[:n_test])
    train_mask = np.array([p not in test_set for p in pid]) & keep

    probe_all, best = fit_probe_scores(H, correct, committed, pid, train_mask)

    # diagnostic: per-sample probe AUROC on TEST committed
    tmask = np.array([p in test_set for p in pid]) & committed
    test_auroc = auroc(probe_all[tmask], correct[tmask])
    logit_auroc = auroc(lmp[tmask], correct[tmask])

    # build per-problem sample lists (TEST only), sorted by si
    byp = defaultdict(list)
    for i in range(len(pid)):
        if not keep[i] or pid[i] not in test_set:
            continue
        byp[pid[i]].append({"si": int(si[i]), "committed": bool(committed[i]),
                            "ext": ext[i], "correct": int(correct[i]),
                            "probe": float(probe_all[i]), "logit": float(lmp[i])})
    for p in byp:
        byp[p].sort(key=lambda s: s["si"])
    test_pids = sorted(byp.keys())
    n = len(test_pids)

    rng = np.random.default_rng(123)
    methods = {}
    for K in KS:
        rows = {
            "majority": np.array([acc_majority(byp[p], K) for p in test_pids]),
            "probe_rerank": np.array([acc_rerank(byp[p], K, "probe") for p in test_pids]),
            "probe_weighted": np.array([acc_weighted(byp[p], K, "probe") for p in test_pids]),
            "logit_rerank": np.array([acc_rerank(byp[p], K, "logit") for p in test_pids]),
            "logit_weighted": np.array([acc_weighted(byp[p], K, "logit") for p in test_pids]),
            "random_rerank": np.array([acc_rerank_random(byp[p], K, rng) for p in test_pids]),
            "oracle_bestof": np.array([acc_oracle(byp[p], K) for p in test_pids]),
        }
        methods[K] = rows

    acc_by_k = {m: {f"K{K}": round(float(methods[K][m].mean()), 4) for K in KS}
                for m in methods[KS[0]]}

    # compute saving: smallest K where a probe method reaches majority@16 accuracy
    mv16 = methods[16]["majority"].mean()
    saving = {}
    for m in ["probe_rerank", "probe_weighted", "logit_rerank"]:
        hit = next((K for K in KS if methods[K][m].mean() >= mv16 - 1e-9), None)
        saving[m] = hit
    saving["majority_at16_acc"] = round(float(mv16), 4)

    # paired bootstrap: probe_rerank - majority and probe_weighted - majority at each K
    def boot_gap(a, b, B=2000, seed=7):
        r = np.random.default_rng(seed)
        d = a - b
        vals = [d[r.integers(0, n, n)].mean() for _ in range(B)]
        return (round(float(np.mean(d)), 4), round(float(np.percentile(vals, 2.5)), 4),
                round(float(np.percentile(vals, 97.5)), 4), round(float((np.array(vals) > 0).mean()), 4))

    sig = {}
    for K in KS:
        mv = methods[K]["majority"]
        sig[f"K{K}"] = {
            "rerank_minus_mv": boot_gap(methods[K]["probe_rerank"], mv, seed=K * 3),
            "weighted_minus_mv": boot_gap(methods[K]["probe_weighted"], mv, seed=K * 5),
            "rerank_minus_logitRerank": boot_gap(methods[K]["probe_rerank"],
                                                 methods[K]["logit_rerank"], seed=K * 11),
        }

    # early-stop curve (probe) vs uniform-majority curve
    taus = [round(x, 3) for x in np.linspace(0.0, 1.0, 41)]
    es = []
    for tau in taus:
        res = [early_stop(byp[p], tau) for p in test_pids]
        acc = float(np.mean([c for c, _ in res]))
        comp = float(np.mean([d for _, d in res]))
        es.append([round(comp, 4), round(acc, 4), tau])
    es.sort()
    mv_curve = [[float(K), round(float(methods[K]["majority"].mean()), 4)] for K in KS]

    return {
        "bench": bench,
        "n_test_problems": n,
        "n_train_records_committed": int((train_mask & committed).sum()),
        "probe_best_layer": best["L"],
        "probe_train_cv_auroc": round(best["auc"], 4),
        "per_sample_auroc_TEST": {"probe": round(test_auroc, 4) if test_auroc else None,
                                  "logit": round(logit_auroc, 4) if logit_auroc else None},
        "acc_by_k": acc_by_k,
        "compute_to_match_majority16": saving,
        "significance_paired_bootstrap": sig,
        "early_stop_probe_curve": es,
        "majority_curve": mv_curve,
    }


def main():
    bench = sys.argv[1] if len(sys.argv) > 1 else "math_test"
    npz_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs/probing"
    rollout = sys.argv[3] if len(sys.argv) > 3 else f"/tmp/s0_sub_{bench}.jsonl"
    out_path = sys.argv[4] if len(sys.argv) > 4 else \
        f"research/outputs/probe_bestofn_{bench}.json"
    r = analyze(bench, npz_dir, rollout)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(r, open(out_path, "w"), indent=2)

    print(f"\n=== {bench}  (n_test={r['n_test_problems']}, probe L{r['probe_best_layer']} "
          f"cv_auroc={r['probe_train_cv_auroc']}, "
          f"per-sample TEST auroc probe={r['per_sample_auroc_TEST']['probe']} "
          f"logit={r['per_sample_auroc_TEST']['logit']}) ===")
    print("  accuracy by K (matched compute):")
    methods = list(r["acc_by_k"].keys())
    print("    K     | " + " ".join(f"{m:>14}" for m in methods))
    for K in KS:
        print(f"    K{K:<4} | " + " ".join(f"{r['acc_by_k'][m]['K'+str(K)]:>14}" for m in methods))
    print(f"  majority@16 acc = {r['compute_to_match_majority16']['majority_at16_acc']}")
    print("  smallest K reaching majority@16 acc:")
    for m in ["probe_rerank", "probe_weighted", "logit_rerank"]:
        print(f"    {m:<16} -> K={r['compute_to_match_majority16'][m]}")
    print("  paired bootstrap probe_rerank - majority:")
    for K in KS:
        g = r["significance_paired_bootstrap"][f"K{K}"]["rerank_minus_mv"]
        print(f"    K{K:<3}: gap={g[0]:+.4f} CI[{g[1]},{g[2]}] P(>0)={g[3]}")
    print(f"\nwrote -> {out_path}")


if __name__ == "__main__":
    main()
