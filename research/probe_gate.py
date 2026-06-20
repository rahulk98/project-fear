#!/usr/bin/env python3
"""De-risk pivot idea #2: probe-gated SINGLE-SHOT abstention.

Question: from ONE forward pass, can a tiny linear probe on the hidden state (or the
free logit confidence) rank correct-vs-incorrect answers well enough to be a useful
single-shot abstention gate -- and how far below K-sample self-consistency does it sit?

Runs on alias CPU over the Exp2 hidden-state caches (no GPU, no interference).
Inputs per (condition, benchmark): the *_hidden.npz (H:[N,29,1536], y:[N] 0=correct/
1=incorrect/2=abstained) + .meta.json (items[] carry problem_id, logit_entropy,
logit_max_prob). Output: a compact JSON of AUROCs + a natural-base-rate risk-coverage curve.

Usage: probe_gate.py <npz> <meta> <natural_att> <out_json>
"""
import sys, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

LAYERS = list(range(8, 25))  # candidate layers (index into the 29-layer axis)


def natural_risk_coverage(score, correct, natural_att, n_pts=50):
    """Risk-coverage curve reweighted so the committed population has correct-rate = natural_att.
    score: higher = more likely correct. Returns coverage[], acc_among_kept[]."""
    correct = correct.astype(float)
    n_c = correct.sum(); n_w = (1 - correct).sum()
    w = np.where(correct == 1, natural_att / max(n_c, 1), (1 - natural_att) / max(n_w, 1))
    order = np.argsort(-score)
    w, corr = w[order], correct[order]
    cumw = np.cumsum(w); cum_correct = np.cumsum(w * corr)
    coverage = cumw / cumw[-1]
    acc = cum_correct / np.maximum(cumw, 1e-12)
    idx = np.unique(np.linspace(0, len(coverage) - 1, n_pts).astype(int))
    return coverage[idx].tolist(), acc[idx].tolist()


def main():
    npz, meta, natural_att, out_json = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    d = np.load(npz)
    H, y = d["H"], d["y"]
    m = json.load(open(meta))
    items = m["items"]
    pid = np.array([it["problem_id"] for it in items])
    lmp = np.array([it.get("logit_max_prob", np.nan) for it in items], dtype=float)
    lent = np.array([it.get("logit_entropy", np.nan) for it in items], dtype=float)

    committed = (y == 0) | (y == 1)
    Hc, pidc = H[committed], pid[committed]
    yc = (y[committed] == 0).astype(int)  # 1 = correct
    lmpc, lentc = lmp[committed], lent[committed]

    # probe: pick best layer by group-5-fold OOF AUROC
    best = {"auc": -1, "L": None, "oof": None}
    for L in LAYERS:
        X = Hc[:, L, :]
        oof = np.zeros(len(yc))
        for tr, te in GroupKFold(n_splits=5).split(X, yc, pidc):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(X[tr]), yc[tr])
            oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        auc = roc_auc_score(yc, oof)
        if auc > best["auc"]:
            best = {"auc": float(auc), "L": int(L), "oof": oof}

    def safe_auc(s):
        s = np.asarray(s, float)
        return float(roc_auc_score(yc, s)) if np.isfinite(s).all() and len(np.unique(s)) > 1 else None

    cov_p, acc_p = natural_risk_coverage(best["oof"], yc, natural_att)
    out = {
        "npz": npz, "n_committed": int(len(yc)), "natural_att": natural_att,
        "frac_correct_balanced": float(yc.mean()),
        "probe_best_layer": best["L"], "probe_auroc": round(best["auc"], 4),
        "logit_maxprob_auroc": (round(safe_auc(lmpc), 4) if safe_auc(lmpc) is not None else None),
        "neg_entropy_auroc": (round(safe_auc(-lentc), 4) if safe_auc(-lentc) is not None else None),
        "probe_risk_coverage": {"coverage": [round(c, 4) for c in cov_p],
                                 "acc_among_kept": [round(a, 4) for a in acc_p]},
    }
    json.dump(out, open(out_json, "w"), indent=2)
    print(json.dumps({k: out[k] for k in ["probe_best_layer", "probe_auroc",
          "logit_maxprob_auroc", "neg_entropy_auroc", "n_committed"]}))


if __name__ == "__main__":
    main()
