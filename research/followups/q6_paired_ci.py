"""Q6: paired problem-level bootstrap CIs (same resampled problem indices for both conditions in a pair).
Run on the server: python q6_paired_ci.py out.json"""
import json, sys, collections
import numpy as np

EVAL = "/home/s2sadesh/master-thesis-asymmetric-training/outputs/eval/"
K, NB = 16, 2000
BENCH = ["math_test", "gsm8k_test"]


def path(c, b, greedy=False):
    n = {"P0": "P0_k32"}.get(c, c)
    return f"{EVAL}qwen2_5_1_5b_{n}_float16_{'greedy_' if greedy else ''}{b}_rollouts.jsonl"


def load(c, b, greedy=False):
    """per problem (sorted ids): n_abstained, n_committed, n_correct_committed, mode_count, mode_correct."""
    d = collections.defaultdict(list)
    for l in open(path(c, b, greedy)):
        r = json.loads(l)
        if r["sample_idx"] < K:
            d[r["problem_id"]].append((str(r["extracted"]), bool(r["correct"]), bool(r["abstained"])))
    ids = sorted(d)
    out = np.zeros((len(ids), 5), dtype=np.int64)
    for i, pid in enumerate(ids):
        s = d[pid]
        v = collections.Counter(e for e, c_, a in s if not a)
        out[i, 0] = sum(a for _, _, a in s)
        out[i, 1] = sum(not a for _, _, a in s)
        out[i, 2] = sum(c_ and not a for _, c_, a in s)
        if v:
            m, cnt = v.most_common(1)[0]
            out[i, 3] = cnt
            out[i, 4] = any(c_ for e, c_, a in s if not a and e == m)
    return ids, out


def aurc(cnt, ok, tb):
    """mean over coverage prefixes of (1 - cumulative accuracy of the mode answer); ranked by mode-vote count,
    ties broken by a fixed random per-problem key tb (shared across conditions)."""
    o = np.argsort(-(cnt * 10**7 + tb), kind="stable")
    c = ok[o]
    return float(np.mean(1 - np.cumsum(c) / np.arange(1, len(c) + 1)))


def acc_ans(x, i):
    return x[i, 2].sum() / max(x[i, 1].sum(), 1)


def abst(x, i, k=K):
    return x[i, 0].sum() / (len(i) * k)


def wise(x, ref, i):
    """e2 definition: abstained samples on problems where the base committed accuracy p<.5; base-ref, valid = base committed>0."""
    valid = ref[:, 1] > 0
    p = np.where(valid, ref[:, 2] / np.maximum(ref[:, 1], 1), 0.0)
    w = (p < .5) & valid
    j = i[valid[i]]
    tot = x[j, 0].sum()
    return (x[j, 0] * w[j]).sum() / tot if tot else np.nan


res = {}
for b in BENCH:
    D = {c: load(c, b) for c in ["S0", "S1", "S2", "S3", "S4", "C2", "P0", "base"]}
    ids = D["S0"][0]
    for c in D:
        assert D[c][0] == ids, (c, b)
    X = {c: D[c][1] for c in D}
    n = len(ids)
    rng = np.random.default_rng(0)
    tb = rng.permutation(n).astype(np.int64)
    G = {}
    if b == "math_test":
        G = {c: load(c, b, True)[1] for c in ["S0", "S3", "S4"]}
        assert load("S0", b, True)[0] == ids
    # sanity: point AURC with fixed tie-break
    print(b, "AURC point", {c: round(aurc(X[c][:, 3], X[c][:, 4], tb), 4) for c in X}, flush=True)
    metrics = {}
    for a_, b_ in [("S3", "S0"), ("S4", "S0"), ("C2", "S2"), ("S2", "S0")]:
        metrics[f"acc_answering {a_}-{b_}"] = (a_, b_, lambda x, i, tb_: acc_ans(x, i))
        metrics[f"abstention {a_}-{b_}"] = (a_, b_, lambda x, i, tb_: abst(x, i))
    for a_, b_ in [("S1", "S0"), ("S2", "S0"), ("S3", "S0"), ("S4", "S0"), ("C2", "S2"), ("P0", "S0")]:
        metrics[f"AURC {a_}-{b_}"] = (a_, b_, "aurc")
    for a_ in ("S3", "S4"):
        metrics[f"wise_frac {a_}-S0"] = (a_, "S0", "wise")
    if G:
        for a_ in ("S3", "S4"):
            metrics[f"greedy_acc_answering {a_}-S0"] = (a_, "S0", "greedy")

    def val(kind, c, i):
        if kind == "aurc":
            return aurc(X[c][i, 3], X[c][i, 4], tb[i])
        if kind == "wise":
            return wise(X[c], X["base"], i)
        if kind == "greedy":
            return acc_ans(G[c], i)
        return kind(X[c], i, None)

    full = np.arange(n)
    boots = {m: [] for m in metrics}
    pts = {}
    for m, (a_, b_, kind) in metrics.items():
        pts[m] = val(kind, a_, full) - val(kind, b_, full)
    for _ in range(NB):
        i = rng.integers(0, n, n)
        for m, (a_, b_, kind) in metrics.items():
            boots[m].append(val(kind, a_, i) - val(kind, b_, i))
    for m in metrics:
        v = np.array(boots[m])
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        res[f"{b}/{m}"] = dict(diff=float(pts[m]), ci=[float(lo), float(hi)], excludes_0=bool(lo > 0 or hi < 0))
    print(b, "done", flush=True)
json.dump(dict(K=K, bootstrap=NB, note="tie-break fixed random per problem; wise uses base reference", cells=res), open(sys.argv[1], "w"), indent=1)
