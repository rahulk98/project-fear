"""E4: abstention rate per condition x benchmark with problem-level bootstrap 95% CI."""
import json, sys
import numpy as np

EVAL = "/home/s2sadesh/master-thesis-asymmetric-training/outputs/eval/"
CONDS = ["S0", "S1", "S2", "S3", "S4", "C1", "C2"]
BENCHES = ["math_test", "gsm8k_test", "aime_2024", "aime_2025", "mmlu_stem", "truthful_qa_mc"]
K, NB = 16, 2000
rng = np.random.default_rng(0)

summary = json.load(open(sys.argv[1]))["per_condition_bench"]
out, mism = {}, []
for c in CONDS:
    for b in BENCHES:
        per = {}
        for l in open(f"{EVAL}qwen2_5_1_5b_{c}_float16_{b}_rollouts.jsonl"):
            r = json.loads(l)
            if r["sample_idx"] < K:
                per.setdefault(r["problem_id"], []).append(r["abstained"])
        f = np.array([np.mean(v) for v in per.values()])
        idx = rng.integers(0, len(f), (NB, len(f)))
        bs = f[idx].mean(1)
        est = float(f.mean())
        ref = summary[c][b]["abstention_rate_mean"]
        if round(est, 4) != round(ref, 4):
            mism.append((c, b, est, ref))
        out[f"{c}/{b}"] = dict(cond=c, bench=b, n_problems=len(f), rate=est,
                               ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5)),
                               summary_rate=ref)
json.dump(dict(bootstrap=NB, K=K, cells=out, mismatches=mism), open(sys.argv[2], "w"), indent=1)
print("mismatches:", mism)
print("max CI halfwidth:", max((v["ci_hi"] - v["ci_lo"]) / 2 for v in out.values()))
