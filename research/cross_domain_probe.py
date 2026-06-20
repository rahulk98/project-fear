#!/usr/bin/env python3
"""Decisive Pivot-A experiment: does the single-pass correctness probe TRANSFER across domains?

Train the probe on MATH hidden states (S0), pick the best layer by group-CV on math, then apply
that fixed probe to GSM8K / MMLU-STEM / TruthfulQA-MC and report transfer AUROC vs each domain's
in-domain probe AUROC (the ceiling). If transfer survives a real domain shift, the contribution is
"one cheap forward pass is a TRANSFERABLE selective-prediction signal"; if not, that is an honest
boundary on the method.

Runs on alias (npz + sklearn there). Reads outputs/probing/qwen2_5_1_5b_S0_float32_{bench}_hidden.npz.
Usage: cross_domain_probe.py <out_json>
"""
import sys, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

DOMAINS = ["math_test", "gsm8k_test", "mmlu_stem", "truthful_qa_mc"]
TRAIN = "math_test"
BASE = "outputs/probing/qwen2_5_1_5b_S0_float32_%s_hidden"


def load(bench):
    d = np.load((BASE % bench) + ".npz")
    H, y = d["H"], d["y"]
    m = json.load(open((BASE % bench) + ".meta.json"))
    pid = np.array([it["problem_id"] for it in m["items"]])
    mask = (y == 0) | (y == 1)            # committed only (correct=0 / incorrect=1 in label_map)
    return H[mask], (y[mask] == 0).astype(int), pid[mask]   # y=1 means correct


def cv_auroc_at_layer(H, y, pid, L, C=0.5):
    X = H[:, L, :]
    oof = np.zeros(len(y))
    for a, b in GroupKFold(n_splits=5).split(X, y, pid):
        sc = StandardScaler().fit(X[a])
        clf = LogisticRegression(max_iter=2000, C=C).fit(sc.transform(X[a]), y[a])
        oof[b] = clf.predict_proba(sc.transform(X[b]))[:, 1]
    return roc_auc_score(y, oof)


def main():
    out_json = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cross_domain_probe.json"
    Htr, ytr, ptr = load(TRAIN)
    # pick best layer on math by group-CV
    best = {"auc": -1, "L": None}
    for L in range(8, 25):
        auc = cv_auroc_at_layer(Htr, ytr, ptr, L)
        if auc > best["auc"]:
            best = {"auc": float(auc), "L": int(L)}
    L = best["L"]
    # refit on ALL math at L
    sc = StandardScaler().fit(Htr[:, L, :])
    clf = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(Htr[:, L, :]), ytr)
    res = {"train_domain": TRAIN, "best_layer_idx": L,
           "train_cv_auroc": round(best["auc"], 4), "cross_domain": {}}
    for dom in DOMAINS:
        H, y, pid = load(dom)
        transfer = roc_auc_score(y, clf.predict_proba(sc.transform(H[:, L, :]))[:, 1])
        indom = cv_auroc_at_layer(H, y, pid, L)   # same layer, in-domain CV = ceiling
        res["cross_domain"][dom] = {
            "transfer_auroc_from_math": round(float(transfer), 4),
            "in_domain_auroc_same_layer": round(float(indom), 4),
            "n_committed": int(len(y)), "frac_correct": round(float(y.mean()), 4),
        }
    print(json.dumps(res, indent=2))
    json.dump(res, open(out_json, "w"), indent=2)


if __name__ == "__main__":
    main()
