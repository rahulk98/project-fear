#!/usr/bin/env python3
"""Pivot 1 (Abstaining Verifier) -- de-risk analysis on MC rollouts.

Three checks the zero-training de-risk must pass before training a 3-way PRM:
  (a) the "unverifiable" bucket (Wilson CI of MC success rate straddles 0.5) is a
      non-trivial slice (target ~10-30%, not ~0 and not ~100);
  (b) 2x2 sanity: the unverifiable bucket holds real gold-correct mass (mean MC
      value near 0.5), clearly separated from the verifiable-wrong bucket (near 0)
      -- i.e. it is NOT just a renamed "incorrect" class;
  (c) the MC value (and CI width) traces a coverage-risk frontier: abstaining on
      the most-uncertain prefixes lowers error on the answered set (AUROC > 0.5),
      and a best-of-K deferral on the unverifiable band recovers correctness.

Two modes (single script):
  step   : reads the step-level rollouts from av_gen_rollouts.py (GPU output).
  answer : reads cached answer-level K-sample data (research/data/uc_strip/*.jsonl,
           records {pid,si,c,a,e}) as a no-GPU complementary pre-check.

modal_correct is precomputed per record in each loader: an int label, or None to
fall back to (MC value > 0.5). No GPU needed for the answer mode; the step mode
uses math_verify only if available, else falls back gracefully.

Writes outputs/abstaining_verifier_<mode>.json.

Usage:
  av_analyze.py step   research/data/av_rollouts.jsonl
  av_analyze.py answer research/data/uc_strip/K32_P0_math_test.jsonl [more files...]
"""
import sys
import os
import json
import math
from collections import Counter, defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))  # for the grader fallback
Z = 1.96


def wilson(k, n, z=Z):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def bucket(k, n):
    lo, hi = wilson(k, n)
    if lo < 0.5 < hi:
        return "unverifiable"
    return "ver_correct" if (k / n) > 0.5 else "ver_wrong"


def auroc(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None
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
    n1 = labels.sum()
    n0 = len(labels) - n1
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def risk_coverage(scores, labels):
    """Abstain on lowest-score items first; report risk (error) at coverage levels."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    order = np.argsort(-scores)  # most confident first
    lab = labels[order]
    n = len(lab)
    out = {}
    for cov in (0.5, 0.7, 0.9, 1.0):
        m = max(1, int(round(cov * n)))
        out[f"risk@cov{cov}"] = round(float(1 - lab[:m].mean()), 4)
    return out


def summarize(records, mode):
    """records: dicts with keys K, n_correct, modal_correct (int or None)."""
    counts = Counter()
    phat_by_bucket = defaultdict(list)
    scores, labels, neg_ciw = [], [], []
    defer_single, defer_bestofk = [], []
    for r in records:
        n, kc = r["K"], r["n_correct"]
        if n < 2:
            continue
        p_hat = kc / n
        b = bucket(kc, n)
        counts[b] += 1
        phat_by_bucket[b].append(p_hat)
        mc = r.get("modal_correct")
        label = int(p_hat > 0.5) if mc is None else int(mc)
        scores.append(p_hat)
        labels.append(label)
        lo, hi = wilson(kc, n)
        neg_ciw.append(-(hi - lo))
        if b == "unverifiable":
            defer_single.append(p_hat)                    # E[correct] for one sample
            defer_bestofk.append(1.0 if kc > 0 else 0.0)  # best-of-K: any correct

    total = sum(counts.values())
    a_phat = auroc(scores, labels)
    a_ciw = auroc(neg_ciw, labels)
    return {
        "mode": mode,
        "n_items": total,
        "bucket_counts": dict(counts),
        "bucket_frac": {b: round(counts[b] / total, 4) for b in counts} if total else {},
        "unverifiable_frac": round(counts["unverifiable"] / total, 4) if total else None,
        "mean_phat_by_bucket": {b: round(float(np.mean(v)), 4) for b, v in phat_by_bucket.items()},
        "gold_correct_mass_unverifiable": round(float(np.mean(phat_by_bucket["unverifiable"])), 4)
            if phat_by_bucket["unverifiable"] else None,
        "auroc_phat_for_modal_correct": round(a_phat, 4) if a_phat is not None else None,
        "auroc_ciwidth_for_modal_correct": round(a_ciw, 4) if a_ciw is not None else None,
        "risk_coverage_by_phat": risk_coverage(scores, labels) if scores else {},
        "deferral_on_unverifiable": {
            "n": len(defer_single),
            "single_sample_acc": round(float(np.mean(defer_single)), 4) if defer_single else None,
            "best_of_K_acc": round(float(np.mean(defer_bestofk)), 4) if defer_bestofk else None,
        },
    }


def _grader():
    try:
        from asym_rewards.reward import is_correct
        return is_correct
    except Exception:
        return None


def load_step(path):
    """Prefer the gold-graded modal_correct stored by the generator. If an older
    file lacks it, re-grade with math_verify when importable, else fall back to
    p_hat>0.5 (self-referential; flagged in the output by modal_correct=None)."""
    is_correct = _grader()
    recs = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if "modal_correct" in d:
                recs.append(d)
                continue
            exts = [e for e in (d.get("exts") or []) if e]
            modal = Counter(exts).most_common(1)
            if is_correct is not None and modal:
                d["modal_correct"] = int(bool(is_correct(modal[0][0], d["gold"])))
            else:
                d["modal_correct"] = None  # fall back to p_hat>0.5
            recs.append(d)
    return recs


def load_answer(path):
    byp = defaultdict(list)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            byp[d["pid"]].append(d)
    recs = []
    for pid, ss in byp.items():
        com = [s for s in ss if not s.get("a")]
        if len(com) < 2:
            continue
        kc = sum(int(s["c"]) for s in com)
        modal = Counter(s["e"] for s in com).most_common(1)[0][0]
        modal_correct = int(any(s["c"] for s in com if s["e"] == modal))
        recs.append({"K": len(com), "n_correct": kc, "modal_correct": modal_correct})
    return recs


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    mode = sys.argv[1]
    files = sys.argv[2:]
    if mode == "step":
        out = {os.path.basename(p): summarize(load_step(p), "step") for p in files}
        dest = os.path.join(ROOT, "outputs", "abstaining_verifier_step.json")
    elif mode == "answer":
        out = {os.path.basename(p): summarize(load_answer(p), "answer") for p in files}
        dest = os.path.join(ROOT, "outputs", "abstaining_verifier_answer.json")
    else:
        print("mode must be 'step' or 'answer'"); sys.exit(1)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
