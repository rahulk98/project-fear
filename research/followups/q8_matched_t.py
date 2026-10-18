#!/usr/bin/env python3
"""Q8: matched-t check. Do two reward triples with equal t give the same GRPO advantages?

Trainer formula (src/asym_rewards/grpo.py, z_score branch, std_floor default 1e-4):
  low = std < floor;  adv = where(low, 0, (r - mean) / (where(low, 1, std) + 1e-4)), std = torch.std (ddof=1).
Real groups: S0 cached K=16 rollouts (research/data/uc_strip/K16_S0_{bench}.jsonl), split into
groups of 8 (si 0-7, 8-15). Outcome: a==1 abstain, else c==1 correct, else wrong. Runs on CPU.
"""
import json, os, sys, collections
import numpy as np, torch
torch.set_num_threads(1)
REPO = os.environ.get("FEAR_REPO_DIR", ".")
OUT = sys.argv[1]
FLOOR = 1e-4
TRI = {"A_t.75 (1,-3,0)": (1, -3, 0), "B_t.75 (1,-1,.5)": (1, -1, .5),
       "C2_t.75 (1,-1,.5)": (1, -1, .5), "S2like_t.80 (.5,-2,0)": (.5, -2, 0)}


def adv(R):  # R: (n_groups, 8) float32, exact trainer branch
    mean = R.mean(1, keepdim=True); std = R.std(1, keepdim=True)
    low = std < FLOOR
    safe = torch.where(low, torch.ones_like(std), std)
    return torch.where(low, torch.zeros_like(R), (R - mean) / (safe + 1e-4))


def run(bench):
    byp = collections.defaultdict(dict)
    for l in open(f"{REPO}/research/data/uc_strip/K16_S0_{bench}.jsonl"):
        d = json.loads(l); byp[d["pid"]][d["si"]] = 2 if d["a"] else (0 if d["c"] else 1)  # 0 corr,1 wrong,2 abst
    O = []
    for p, s in byp.items():
        for h in (0, 8):
            O.append([s[h + i] for i in range(8)])
    O = torch.tensor(O)
    kind = np.array([len(set(r)) for r in O.tolist()])
    names = {1: "all-same", 2: "two-outcome", 3: "three-outcome"}
    def rew(t):
        return torch.tensor(t, dtype=torch.float32)[O]
    adv_ = {k: adv(rew(t)) for k, t in TRI.items()}
    res = {"n_groups": len(O), "n_samples": int(O.numel()),
           "group_counts": {names[k]: int((kind == k).sum()) for k in (1, 2, 3)}}
    def cmp(a, b):
        out = {}
        for tag, m in [("all", np.ones(len(O), bool))] + [(names[k], kind == k) for k in (1, 2, 3)]:
            if m.sum() == 0: continue
            A, B = adv_[a][m], adv_[b][m]
            d = (A - B).abs()
            out[tag] = {"groups": int(m.sum()), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                        "mean_abs_adv_ref": float(A.abs().mean()),
                        "mean_rel_diff": float(d.mean() / A.abs().mean()) if A.abs().mean() > 0 else None,
                        "max_rel_to_max_adv": float(d.max() / A.abs().max()) if A.abs().max() > 0 else None}
        return out
    res["A_vs_B_(both t=.75)"] = cmp("A_t.75 (1,-3,0)", "B_t.75 (1,-1,.5)")
    res["C2_vs_S2like_(t .75 vs .80)"] = cmp("C2_t.75 (1,-1,.5)", "S2like_t.80 (.5,-2,0)")
    # fraction of samples whose advantage sign differs
    for tag, (a, b) in {"A_vs_B": ("A_t.75 (1,-3,0)", "B_t.75 (1,-1,.5)"), "C2_vs_S2like": ("C2_t.75 (1,-1,.5)", "S2like_t.80 (.5,-2,0)")}.items():
        res[f"sign_flip_frac_{tag}"] = float((torch.sign(adv_[a]) != torch.sign(adv_[b])).float().mean())
    return res


out = {b: run(b) for b in ["math_test", "gsm8k_test"]}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1))
