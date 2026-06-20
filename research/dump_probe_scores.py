#!/usr/bin/env python3
"""Dump per-record OOF probe scores so the probe can be compared to self-consistency on the
SAME population (the matched-population fix). For S0 math+gsm8k: fit best-layer correctness
probe via group-CV, emit JSONL {pid, si, y, score, lmp, lent} for every committed record.

Runs on alias (npz + sklearn there). Usage: dump_probe_scores.py <bench> <out_jsonl>
"""
import sys, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

BASE = "outputs/probing/qwen2_5_1_5b_S0_float32_%s_hidden"


def main():
    bench, out = sys.argv[1], sys.argv[2]
    d = np.load((BASE % bench) + ".npz")
    H, y = d["H"], d["y"]
    m = json.load(open((BASE % bench) + ".meta.json"))
    items = m["items"]
    pid = np.array([it["problem_id"] for it in items])
    si = np.array([it.get("sample_idx", -1) for it in items])
    lmp = np.array([it.get("logit_max_prob", np.nan) for it in items], float)
    lent = np.array([it.get("logit_entropy", np.nan) for it in items], float)
    mask = (y == 0) | (y == 1)
    Hc, pidc, sic = H[mask], pid[mask], si[mask]
    yc = (y[mask] == 0).astype(int)
    lmpc, lentc = lmp[mask], lent[mask]
    # best layer by group-CV, collect OOF scores at that layer
    best = {"auc": -1, "L": None, "oof": None}
    for L in range(8, 25):
        X = Hc[:, L, :]
        oof = np.zeros(len(yc))
        for a, b in GroupKFold(n_splits=5).split(X, yc, pidc):
            sc = StandardScaler().fit(X[a])
            clf = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(X[a]), yc[a])
            oof[b] = clf.predict_proba(sc.transform(X[b]))[:, 1]
        auc = roc_auc_score(yc, oof)
        if auc > best["auc"]:
            best = {"auc": float(auc), "L": int(L), "oof": oof}
    with open(out, "w") as f:
        for i in range(len(yc)):
            f.write(json.dumps({"pid": str(pidc[i]), "si": int(sic[i]), "y": int(yc[i]),
                                "score": float(best["oof"][i]), "lmp": float(lmpc[i]),
                                "lent": float(lentc[i])}) + "\n")
    print(json.dumps({"bench": bench, "best_layer": best["L"], "probe_auroc": round(best["auc"], 4),
                      "n": int(len(yc))}))


if __name__ == "__main__":
    main()
