"""E2: is abstention wise (problem likely wrong) or wasteful? Difficulty from a reference model's committed samples."""
import json, os, sys
import numpy as np

EVAL = os.environ.get("FEAR_EVAL_DIR", "outputs/eval") + "/"
CONDS = ["S0", "C1", "S1", "C2", "S2", "S3", "S4", "P0"]
T = dict(S0=.50, C1=.60, S1=.67, C2=.75, S2=.80, S3=.91, S4=1.0, P0=0.0)
BENCHES = ["math_test", "gsm8k_test", "aime_2024", "aime_2025", "mmlu_stem", "truthful_qa_mc"]
K, NB = 16, 2000
rng = np.random.default_rng(0)


def path(c, b):
    name = {"P0": "P0_k32", "base": "base"}.get(c, c)
    return f"{EVAL}qwen2_5_1_5b_{name}_float16_{b}_rollouts.jsonl"


def load(c, b):
    """per problem: (n_abstained, n_committed, n_correct) over the first K samples."""
    d = {}
    for l in open(path(c, b)):
        r = json.loads(l)
        if r["sample_idx"] >= K:
            continue
        a = d.setdefault(r["problem_id"], [0, 0, 0])
        a[0] += r["abstained"]
        a[1] += not r["abstained"]
        a[2] += r["correct"] and not r["abstained"]
    return d


def auroc(score, pos):
    """P(score_pos > score_neg) + .5 ties, scores are ints 0..K."""
    hp = np.bincount(score[pos], minlength=K + 1).astype(float)
    hn = np.bincount(score[~pos], minlength=K + 1).astype(float)
    if hp.sum() == 0 or hn.sum() == 0:
        return np.nan
    cn = np.cumsum(hn) - hn
    return float(((hp * (cn + .5 * hn)).sum()) / (hp.sum() * hn.sum()))


def stats(na, wise, valid):
    """na: abstained count per problem (of K); wise: bool per problem; valid: bool mask."""
    na, wise = na[valid], wise[valid]
    tot = na.sum()
    return dict(abst_rate=float(na.mean() / K),
                wise_frac=float((na * wise).sum() / tot) if tot else np.nan,
                random_wise_frac=float(wise.mean()),
                auroc=auroc(na, wise))


def cell(na, p, valid):
    wise = p < .5
    pt = stats(na, wise, valid)
    n = len(na)
    bs = {k: [] for k in pt}
    for _ in range(NB):
        i = rng.integers(0, n, n)
        s = stats(na[i], wise[i], valid[i])
        for k in pt:
            bs[k].append(s[k])
    ci = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in bs.items()}
    return dict(point=pt, ci=ci, n_problems=int(valid.sum()))


res = {}
for b in BENCHES:
    ref = {r: load(r, b) for r in ("base", "S0")}
    for c in CONDS:
        own = load(c, b)
        ids = sorted(own)
        na = np.array([own[i][0] for i in ids])
        # primary variant uses base committed samples, then S0, then the condition's own
        for variant in ("base", "S0", "own"):
            if variant == "S0" and c == "S0":
                continue
            src = own if variant == "own" else ref[variant]
            nc = np.array([src[i][1] for i in ids])
            ncor = np.array([src[i][2] for i in ids])
            # problems with no committed reference samples have no p_hat: excluded
            valid = nc > 0
            p = np.where(valid, ncor / np.maximum(nc, 1), 0.0)
            res[f"{c}/{b}/{variant}"] = dict(cond=c, bench=b, ref=variant, t=T[c],
                                             n_excluded=int((~valid).sum()), **cell(na, p, valid))
        print(c, b, flush=True)
json.dump(dict(K=K, bootstrap=NB, cells=res), open(sys.argv[1], "w"), indent=1)
