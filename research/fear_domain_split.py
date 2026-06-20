#!/usr/bin/env python3
"""Goal-1 avenue: is there a DOMAIN where punishment (fear) genuinely improves
selective accuracy, rather than just dialing up abstention?

On standard math (math_test, gsm8k_test) more fear (S0 -> S4) lowers
accuracy-on-attempted (established negative). But on adversarial / misleading
benchmarks (truthful_qa_mc) and hard OOD (aime), the questions are built so the
tempting answer is often wrong. There, extra abstention may be POSITIVELY
selected -- the model abstains on items it would have gotten wrong -- so
acc_on_attempted could RISE with fear. If it does, that is a genuine, defensible,
domain-specific fear advantage (a safety-relevant "does not assert a falsehood"
gain), not just the abstention dial.

Uses cached greedy (T=0, K=1, paired-causal: only LoRA weights differ) rollouts.
No GPU. numpy-free. Records: {pid, si, c(correct), a(abstain), e(extracted)}.

Writes outputs/fear_domain_split.json.
"""
import os
import json
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "uc_strip")
CONDS = ["S0", "S1", "S2", "S3", "S4", "C1", "C2"]
BENCH = ["math_test", "gsm8k_test", "mmlu_stem", "truthful_qa_mc", "aime_2024", "aime_2025"]


def load(cond, bench):
    p = os.path.join(DATA, f"greedy_{cond}_{bench}.jsonl")
    if not os.path.exists(p):
        return None
    rows = []
    with open(p) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stats(rows):
    n = len(rows)
    committed = [r for r in rows if not r["a"]]
    nc = len(committed)
    correct = sum(int(r["c"]) for r in rows)
    corr_committed = sum(int(r["c"]) for r in committed)
    return {
        "n": n,
        "abst_rate": round(1 - nc / n, 4) if n else None,
        "overall_acc": round(correct / n, 4) if n else None,
        "acc_on_attempted": round(corr_committed / nc, 4) if nc else None,
        "confident_wrong_rate": round((nc - corr_committed) / n, 4) if n else None,
    }


def main():
    out = {}
    for bench in BENCH:
        out[bench] = {}
        for cond in CONDS:
            rows = load(cond, bench)
            if rows is None:
                continue
            out[bench][cond] = stats(rows)
    # Trend test: does acc_on_attempted rise from S0 to S4 on each benchmark?
    trends = {}
    for bench in BENCH:
        seq = [out[bench].get(c, {}).get("acc_on_attempted") for c in ["S0", "S1", "S2", "S3", "S4"]]
        seq_v = [(c, v) for c, v in zip(["S0", "S1", "S2", "S3", "S4"], seq) if v is not None]
        if len(seq_v) >= 2:
            s0 = next((v for c, v in seq_v if c == "S0"), None)
            s4 = next((v for c, v in seq_v if c == "S4"), None)
            best = max(seq_v, key=lambda x: x[1])
            trends[bench] = {
                "S0_acc_on_attempted": s0,
                "S4_acc_on_attempted": s4,
                "S4_minus_S0": round(s4 - s0, 4) if (s0 is not None and s4 is not None) else None,
                "best_cond": best[0], "best_acc_on_attempted": best[1],
                "fear_helps_attempted": (s0 is not None and s4 is not None and s4 > s0),
            }
    out["_trends_acc_on_attempted_S0_to_S4"] = trends
    dest = os.path.join(ROOT, "outputs", "fear_domain_split.json")
    json.dump(out, open(dest, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
