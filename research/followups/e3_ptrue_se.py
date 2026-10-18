#!/usr/bin/env python3
"""E3: P(True) and semantic entropy baselines on the matched probe population.

Population (reproduces research/matched_population.py): problems with a committed correct/incorrect
probe record (S0 hidden-state meta) that also have a non-abstained K16 sample. Target = is the K16
mode answer correct. P(True) scores that same mode answer with the S0 merged model (the model that
produced the probe's hidden states). Semantic entropy and mode-vote agreement use the first K cached
samples. Stages: ptrue (GPU), se (CPU, math_verify), report. Usage: e3_ptrue_se.py <stage> <outdir>
"""
import json, os, sys, math, time
from collections import Counter, defaultdict
import numpy as np

REPO = os.environ.get("FEAR_REPO_DIR", ".")
OUTPUTS = os.environ.get("FEAR_OUTPUTS_DIR", "outputs")
S0 = f"{OUTPUTS}/grpo/qwen2_5_1_5b_S0_s42/_merged_for_vllm_float16"
BENCHES = ["math_test", "gsm8k_test"]
TRAIN = {"math_test": "math_train", "gsm8k_test": "gsm8k_train"}
KS = [2, 4, 8, 16]
ZS = ("Question: {q}\nProposed answer: {a}\nIs the proposed answer correct?\n"
      "(A) True\n(B) False\nAnswer:")


def load_pop(bench):
    meta = json.load(open(os.environ.get("FEAR_PROBING_DIR", "outputs/probing") + f"/qwen2_5_1_5b_S0_float32_{bench}_hidden.meta.json"))
    cp = {i["problem_id"] for i in meta["items"] if i["label"] in ("correct", "incorrect")}
    byp = defaultdict(list)
    for l in open(f"{REPO}/research/data/uc_strip/K16_S0_{bench}.jsonl"):
        if l.strip():
            d = json.loads(l); byp[d["pid"]].append(d)
    pop = {}
    for p in sorted(cp):
        ss = sorted(byp.get(p, []), key=lambda r: r["si"])[:16]
        na = [s for s in ss if not s["a"]]
        if not na:
            continue
        ans, _ = Counter(s["e"] for s in na).most_common(1)[0]
        pop[p] = {"samples": ss, "mode": ans, "label": int(any(s["c"] for s in na if s["e"] == ans))}
    return pop


def fewshot(bench):
    import pandas as pd
    df = pd.read_parquet(f"{REPO}/data/processed/{TRAIN[bench]}.parquet")
    df = df[df["answer"].astype(str).str.fullmatch(r"\d+") & (df["prompt"].str.len() < 400)]
    r1, r2 = df.iloc[0], df.iloc[1]
    wrong = str(int(r2["answer"]) + 2)
    blocks = [ZS.format(q=r1["prompt"], a=str(r1["answer"])) + " A\n\n",
              ZS.format(q=r2["prompt"], a=wrong) + " B\n\n"]
    return "".join(blocks)


def stage_ptrue(out):
    import torch, pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S0); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(S0, torch_dtype=torch.float16).cuda().eval()
    ids = {k: tok.encode(k, add_special_tokens=False) for k in [" A", " B", " True", " False"]}
    assert all(len(v) == 1 for v in ids.values()), ids
    iA, iB, iT, iF = (ids[k][0] for k in [" A", " B", " True", " False"])
    res = {}
    for bench in BENCHES:
        pop = load_pop(bench)
        df = pd.read_parquet(f"{REPO}/data/processed/{bench}.parquet").set_index("id")
        fs = fewshot(bench)
        print(bench, "n", len(pop), flush=True)
        pids = list(pop)
        for variant, prefix in [("zero", ""), ("two_shot", fs)]:
            texts = [prefix + ZS.format(q=df.loc[p, "prompt"], a=pop[p]["mode"]) for p in pids]
            order = np.argsort([len(t) for t in texts])
            sc = {}
            for s in range(0, len(order), 16):
                bi = order[s:s + 16]
                enc = tok([texts[i] for i in bi], return_tensors="pt", padding=True).to("cuda")
                with torch.no_grad():
                    h = model.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state[:, -1]
                    lg = model.lm_head(h).float()
                for j, i in enumerate(bi):
                    sc[pids[i]] = {"ab": float(lg[j, iA] - lg[j, iB]), "tf": float(lg[j, iT] - lg[j, iF])}
                if (s // 16) % 20 == 0:
                    print(f"  {bench} {variant} {s}/{len(order)}", flush=True)
            res.setdefault(bench, {})[variant] = sc
    json.dump(res, open(f"{out}/ptrue_scores.json", "w"))
    print("ptrue done", flush=True)


def stage_se(out):
    from math_verify import parse, verify
    def wrap(e):
        return e if "\\boxed" in e else f"\\boxed{{{e}}}"
    res = {}
    for bench in BENCHES:
        pop = load_pop(bench); res[bench] = {}
        for p, d in pop.items():
            parsed, reps, cid = {}, [], []   # reps: list of (string)
            for s in d["samples"]:
                if s["a"]:
                    cid.append("ABST"); continue
                e = s["e"]
                if e not in parsed:
                    try: parsed[e] = parse(wrap(e))
                    except Exception: parsed[e] = None
                hit = None
                for ri, r in enumerate(reps):
                    if r == e: hit = ri; break
                    if parsed[e] is not None and parsed[r] is not None:
                        try:
                            if verify(parsed[r], parsed[e]) and verify(parsed[e], parsed[r]): hit = ri; break
                        except Exception: pass
                if hit is None:
                    reps.append(e); hit = len(reps) - 1
                cid.append(hit)
            res[bench][p] = cid
        print(bench, "clustered", len(pop), flush=True)
    json.dump(res, open(f"{out}/se_clusters.json", "w"))


def ent(labels):
    if not labels: return 0.0
    c = Counter(labels); n = len(labels)
    return -sum(v / n * math.log(v / n) for v in c.values())


def auroc_boot(scores, labels, B=2000, seed=0):
    from scipy.stats import rankdata
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    def au(s, y):
        n1 = y.sum(); n0 = len(y) - n1
        if n1 == 0 or n0 == 0: return np.nan
        r = rankdata(s)
        return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    rng = np.random.default_rng(seed); n = len(scores)
    v = [au(scores[i], labels[i]) for i in (rng.integers(0, n, n) for _ in range(B))]
    return round(float(au(scores, labels)), 4), [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]


def stage_report(out, host_note):
    pt = json.load(open(f"{out}/ptrue_scores.json")); cl = json.load(open(f"{out}/se_clusters.json"))
    final = {"model": S0, "note": host_note, "benches": {}}
    for bench in BENCHES:
        pop = load_pop(bench); pids = list(pop); y = np.array([pop[p]["label"] for p in pids])
        r = {"n": len(pids), "mode_correct_rate": round(float(y.mean()), 4), "signals": {}}
        for v in ["zero", "two_shot"]:
            for k, nm in [("ab", "A/B"), ("tf", "True/False")]:
                a, ci = auroc_boot([pt[bench][v][p][k] for p in pids], y)
                r["signals"][f"ptrue_{v}_{nm}"] = {"compute": "1 forward pass", "auroc": a, "ci": ci}
        for K in KS:
            se_all, se_ex, vote = [], [], []
            for p in pids:
                c = cl[bench][p][:K]; nz = [x for x in c if x != "ABST"]
                se_all.append(-ent(c)); se_ex.append(-ent(nz))
                vote.append(Counter(nz).most_common(1)[0][1] / K if nz else 0.0)
            for nm, s in [("semantic_entropy_with_abstain", se_all), ("semantic_entropy_excl_abstain", se_ex), ("mode_vote_agreement", vote)]:
                a, ci = auroc_boot(s, y)
                r["signals"][f"{nm}_K{K}"] = {"compute": f"{K} generations", "auroc": a, "ci": ci}
        final["benches"][bench] = r
    json.dump(final, open(f"{out}/e3_results.json", "w"), indent=2)
    for b, r in final["benches"].items():
        print(b, "n", r["n"], "mode_correct", r["mode_correct_rate"])
        for k, v in r["signals"].items(): print(f"  {k:42s} {v['auroc']} {v['ci']}")


if __name__ == "__main__":
    st, out = sys.argv[1], sys.argv[2]
    {"ptrue": stage_ptrue, "se": stage_se}.get(st, lambda o: stage_report(o, "bootstrap 2000x over problems; label=K16 mode answer correct; SE/vote use first K cached samples (K=16 shares samples with the label)"))(out)
