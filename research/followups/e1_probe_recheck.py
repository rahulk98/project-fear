#!/usr/bin/env python3
"""E1: re-verify withdrawn probe numbers (nested layer selection, controls, transfer, K*).

Runs on CPU. Usage:
  e1_probe_recheck.py bench <bench>   # A, B, C for one benchmark -> bench_<bench>.json
  e1_probe_recheck.py combine         # D, E, merge -> e1_results.json
Positive class everywhere: label 1 = CORRECT (old convention). Committed rows only (correct/incorrect).
"""
import sys, json, os, re, time, collections
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

REPO = os.environ.get("FEAR_REPO_DIR", ".")
BASE = os.environ.get("FEAR_PROBING_DIR", "outputs/probing") + "/qwen2_5_1_5b_S0_float32_%s_hidden"
EVAL = os.environ.get("FEAR_EVAL_DIR", "outputs/eval") + "/qwen2_5_1_5b_S0_float16_%s_rollouts.jsonl"
K16 = REPO + "/research/data/uc_strip/K16_S0_%s.jsonl"
OUT = os.environ.get("FEAR_WORK_DIR", "outputs/followups") + "/e1"
BENCHES = ["math_test", "gsm8k_test", "mmlu_stem", "truthful_qa_mc"]
LAYERS = list(range(8, 25))
C = 0.5
B = 2000
NPERM = 5
OUTER, INNER = 5, 3


def load(bench):
    d = np.load(BASE % bench + ".npz")
    y = d["y"]
    m = json.load(open(BASE % bench + ".meta.json"))["items"]
    mask = (y == 0) | (y == 1)
    idx = np.where(mask)[0]
    H = d["H"][idx][:, LAYERS, :]                 # (n, 17, 1536); column j = layer LAYERS[j]
    items = [m[i] for i in idx]
    return H, (y[idx] == 0).astype(int), np.array([it["problem_id"] for it in items]), items


def fit_pred(Xa, ya, Xb):
    sc = StandardScaler().fit(Xa)
    clf = LogisticRegression(max_iter=2000, C=C).fit(sc.transform(Xa), ya)
    return clf.predict_proba(sc.transform(Xb))[:, 1]


def oof_layer(X, y, g, n_splits=5):
    oof = np.zeros(len(y))
    for a, b in GroupKFold(n_splits).split(X, y, g):
        oof[b] = fit_pred(X[a], y[a], X[b])
    return oof


def old_pipeline(H, y, g):
    """Non-nested: best of LAYERS by pooled 5-fold group AUROC (what the old scripts did)."""
    aucs, oofs = [], []
    for j in range(len(LAYERS)):
        o = oof_layer(H[:, j, :], y, g); oofs.append(o); aucs.append(roc_auc_score(y, o))
    j = int(np.argmax(aucs))
    return float(aucs[j]), LAYERS[j], oofs[j], [float(a) for a in aucs]


def nested_pipeline(H, y, g):
    oof = np.zeros(len(y)); chosen = []
    for a, b in GroupKFold(OUTER).split(H[:, 0, :], y, g):
        best = (-1, None)
        for j in range(len(LAYERS)):
            o = oof_layer(H[a][:, j, :], y[a], g[a], INNER)
            s = roc_auc_score(y[a], o)
            if s > best[0]:
                best = (s, j)
        j = best[1]; chosen.append(LAYERS[j])
        oof[b] = fit_pred(H[a][:, j, :], y[a], H[b][:, j, :])
    return float(roc_auc_score(y, oof)), chosen, oof


def cluster_ci(scores, y, g, B=B, seed=0):
    rng = np.random.default_rng(seed)
    qs, inv = np.unique(g, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(qs))]
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(qs), len(qs))
        idx = np.concatenate([members[k] for k in pick])
        if 0 < y[idx].sum() < len(idx):
            vals.append(roc_auc_score(y[idx], scores[idx]))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def summ(scores, y, g):
    return {"auroc": round(float(roc_auc_score(y, scores)), 4), "ci95": cluster_ci(scores, y, g)}


def perm_question_level(y, g, rng):
    """Permute per-question label vectors among questions with the same number of rows."""
    y2 = y.copy()
    qs = np.unique(g)
    bysize = collections.defaultdict(list)
    for q in qs:
        bysize[(g == q).sum()].append(q)
    for _, grp in bysize.items():
        vecs = [y[g == q] for q in grp]
        order = rng.permutation(len(grp))
        for q, k in zip(grp, order):
            y2[g == q] = vecs[k]
    return y2


def text_auroc(texts, y, g, folds, **kw):
    oof = np.zeros(len(y))
    for a, b in folds:
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, **kw)
        Xa = v.fit_transform([texts[i] for i in a]); Xb = v.transform([texts[i] for i in b])
        oof[b] = LogisticRegression(max_iter=2000, C=1.0).fit(Xa, y[a]).predict_proba(Xb)[:, 1]
    return summ(oof, y, g)


def letter_features(items, bench):
    rows = {}
    for line in open(EVAL % bench):
        d = json.loads(line); rows[(d["problem_id"], d["sample_idx"])] = d
    letters, nopt, gold0 = [], [], []
    for it in items:
        r = rows[(it["problem_id"], it["sample_idx"])]
        letters.append(r["extracted"] or "NONE")
        nopt.append(len(re.findall(r"(?m)^[A-Z]\. ", it["prompt_text"])))
        gold = json.loads(r["gold"]); gold0.append(int(np.argmax(gold)))
    return np.array(letters), np.array(nopt), np.array(gold0)


def run_bench(bench):
    t0 = time.time()
    H, y, g, items = load(bench)
    res = {"bench": bench, "positive_class": "correct", "n_rows": int(len(y)),
           "n_questions": int(len(np.unique(g))), "frac_correct": round(float(y.mean()), 4),
           "n_correct": int(y.sum()), "n_incorrect": int((1 - y).sum())}
    qlab = collections.defaultdict(set)
    for q, l in zip(g, y):
        qlab[q].add(int(l))
    res["n_questions_both_labels"] = int(sum(len(v) == 2 for v in qlab.values()))
    outer = list(GroupKFold(OUTER).split(H[:, 0, :], y, g))
    # C: leakage check
    res["leakage"] = {"outer_fold_group_overlap": [int(len(set(g[a]) & set(g[b]))) for a, b in outer]}
    # A: old pipeline and nested
    auc, L, oof_old, layer_aucs = old_pipeline(H, y, g)
    res["old_nonnested"] = {**summ(oof_old, y, g), "best_layer_idx": L, "layer_aucs": layer_aucs}
    print(bench, "old", res["old_nonnested"]["auroc"], L, round(time.time() - t0), flush=True)
    auc_n, chosen, oof_n = nested_pipeline(H, y, g)
    res["nested"] = {**summ(oof_n, y, g), "chosen_layer_per_outer_fold": chosen}
    print(bench, "nested", res["nested"]["auroc"], chosen, round(time.time() - t0), flush=True)
    res["oof_old"] = [round(float(v), 5) for v in oof_old]
    res["oof_nested"] = [round(float(v), 5) for v in oof_n]
    res["problem_id"] = g.tolist(); res["sample_idx"] = [it["sample_idx"] for it in items]
    res["logit_max_prob"] = [it.get("logit_max_prob") for it in items]
    # B1: shuffled labels through the old (max-bias, best-of-17-layers) pipeline
    rng = np.random.default_rng(123)
    ctl = {}
    for name in ["row_level", "question_level"]:
        vals = []
        for _ in range(NPERM):
            yp = rng.permutation(y) if name == "row_level" else perm_question_level(y, g, rng)
            vals.append(old_pipeline(H, yp, g)[0])
            print(bench, "perm", name, round(vals[-1], 4), round(time.time() - t0), flush=True)
        ctl[name] = {"aurocs": [round(v, 4) for v in vals], "mean": round(float(np.mean(vals)), 4),
                     "min": round(float(min(vals)), 4), "max": round(float(max(vals)), 4)}
    res["shuffled_label_controls_old_pipeline"] = ctl
    # B2: text-only baselines, same outer folds
    prompts = [it["prompt_text"] for it in items]; comps = [it["completion"] for it in items]
    res["text_only"] = {
        "prompt_plus_completion": text_auroc([p + "\n" + c for p, c in zip(prompts, comps)], y, g, outer),
        "completion_only": text_auroc(comps, y, g, outer),
        "prompt_only": text_auroc(prompts, y, g, outer)}
    print(bench, "text", {k: v["auroc"] for k, v in res["text_only"].items()}, flush=True)
    # B3: TruthfulQA chosen-letter baseline
    if bench == "truthful_qa_mc":
        let, nopt, gold0 = letter_features(items, bench)
        res["gold_index_distribution"] = {str(k): int(v) for k, v in collections.Counter(gold0.tolist()).items()}
        res["frac_rows_gold_is_A"] = round(float((gold0 == 0).mean()), 4)
        res["chosen_letter_distribution"] = dict(collections.Counter(let.tolist()))
        enc = OneHotEncoder(handle_unknown="ignore").fit(let.reshape(-1, 1))
        Xl = np.hstack([enc.transform(let.reshape(-1, 1)).toarray(), nopt.reshape(-1, 1)])
        oof = np.zeros(len(y))
        for a, b in outer:
            oof[b] = LogisticRegression(max_iter=2000).fit(Xl[a], y[a]).predict_proba(Xl[b])[:, 1]
        res["chosen_letter_baseline"] = {"features": "one-hot chosen letter + n_options", **summ(oof, y, g)}
        # question-prior baseline: n_options only
        oof2 = np.zeros(len(y))
        for a, b in outer:
            oof2[b] = LogisticRegression(max_iter=2000).fit(nopt[a].reshape(-1, 1), y[a]).predict_proba(nopt[b].reshape(-1, 1))[:, 1]
        res["n_options_only_baseline"] = summ(oof2, y, g)
    res["wall_s"] = round(time.time() - t0)
    json.dump(res, open(f"{OUT}/bench_{bench}.json", "w"))
    print(bench, "done", res["wall_s"], flush=True)


def fast_auroc(s, l):
    l = np.asarray(l)
    n1 = l.sum(); n0 = len(l) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[l == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def sc_per_pid(path, K):
    byp = collections.defaultdict(list)
    for line in open(path):
        if line.strip():
            d = json.loads(line); byp[d["pid"]].append(d)
    conf, corr = {}, {}
    for pid, ss in byp.items():
        ss = sorted(ss, key=lambda r: r["si"])[:K]
        na = [s for s in ss if not s["a"]]
        if not na:
            continue
        ans, cnt = collections.Counter(s["e"] for s in na).most_common(1)[0]
        conf[pid] = cnt / K
        corr[pid] = 1 if any(s["c"] for s in na if s["e"] == ans) else 0
    return conf, corr


def kstar(probe_a, scs, Ks=(2, 4, 8, 16)):
    if scs[0] >= probe_a:
        return 2.0
    for i in range(1, len(Ks)):
        if scs[i - 1] < probe_a <= scs[i]:
            return Ks[i - 1] + (Ks[i] - Ks[i - 1]) * (probe_a - scs[i - 1]) / (scs[i] - scs[i - 1])
    return None


def matched_population(bench, bj):
    pid = np.array(bj["problem_id"]); lmp = np.array(bj["logit_max_prob"], float)
    k16 = K16 % bench
    _, lbl = sc_per_pid(k16, 16)
    out = {}
    for tag, key in [("probe_old_nonnested_oof", "oof_old"), ("probe_nested_oof", "oof_nested")]:
        sc = np.array(bj[key])
        pb, lg = collections.defaultdict(list), collections.defaultdict(list)
        for p, s, l in zip(pid, sc, lmp):
            pb[p].append(s); lg[p].append(l)
        pids = [p for p in pb if p in lbl]
        label = np.array([lbl[p] for p in pids])
        sig = {"probe": np.array([np.mean(pb[p]) for p in pids]),
               "logit": np.array([np.mean(lg[p]) for p in pids])}
        for K in (2, 4, 8, 16):
            cK, _ = sc_per_pid(k16, K)
            sig[f"sc{K}"] = np.array([cK.get(p, 0.0) for p in pids])
        pt = {k: fast_auroc(v, label) for k, v in sig.items()}
        pt["kstar"] = kstar(pt["probe"], [pt[f"sc{K}"] for K in (2, 4, 8, 16)])
        rng = np.random.default_rng(0); n = len(pids)
        boots = {k: [] for k in pt}
        for _ in range(B):
            i = rng.integers(0, n, n)
            r = {k: fast_auroc(v[i], label[i]) for k, v in sig.items()}
            r["kstar"] = kstar(r["probe"], [r[f"sc{K}"] for K in (2, 4, 8, 16)])
            for k, v in r.items():
                boots[k].append(v)
        ci = {}
        for k, v in boots.items():
            vv = [x for x in v if x is not None and not np.isnan(x)]
            ci[k] = [round(float(np.percentile(vv, 2.5)), 3), round(float(np.percentile(vv, 97.5)), 3)]
            if k == "kstar":
                ci[k].append(f"{len(vv)}/{B} resamples with K* inside [2,16]")
        out[tag] = {"n_problems": n, "mode_correct_rate": round(float(label.mean()), 4),
                    "point": {k: (None if v is None else round(float(v), 4)) for k, v in pt.items()}, "ci95": ci}
    return out


def combine():
    t0 = time.time()
    bj = {b: json.load(open(f"{OUT}/bench_{b}.json")) for b in BENCHES}
    # D: train on all MATH, layer by group-CV on MATH only (= old pipeline's layer), frozen transfer
    Hm, ym, gm, _ = load("math_test")
    j = int(np.argmax(bj["math_test"]["old_nonnested"]["layer_aucs"]))
    sc = StandardScaler().fit(Hm[:, j, :])
    clf = LogisticRegression(max_iter=2000, C=C).fit(sc.transform(Hm[:, j, :]), ym)
    D = {"train": "math_test", "layer": LAYERS[j], "train_cv_auroc": bj["math_test"]["old_nonnested"]["auroc"],
         "positive_class": "correct", "transfer": {}}
    for b in BENCHES[1:]:
        H, y, g, _ = load(b)
        s = clf.predict_proba(sc.transform(H[:, j, :]))[:, 1]
        D["transfer"][b] = summ(s, y, g)
        # also report in-domain at the same layer is not recomputed here; see bench json
    # E
    E = {b: matched_population(b, bj[b]) for b in ["math_test", "gsm8k_test"]}
    slim = {}
    for b, r in bj.items():
        slim[b] = {k: v for k, v in r.items() if k not in ("oof_old", "oof_nested", "problem_id", "sample_idx", "logit_max_prob")}
    json.dump({"config": {"layers": "range(8,25) (column j = hidden_states index LAYERS[j] as in old scripts)", "C": C,
                          "outer_folds": OUTER, "inner_folds_nested": INNER, "bootstrap": B, "n_perm": NPERM,
                          "positive_class": "correct (label 1); committed rows only",
                          "ci": "question-level cluster bootstrap", "controls_pipeline": "old non-nested best-of-17-layers",
                          "tfidf": "word 1-2 grams, min_df=2, sublinear_tf, LR C=1.0"},
               "A_B_C_per_benchmark": slim, "D_transfer": D, "E_matched_population": E,
               "combine_wall_s": round(time.time() - t0)}, open(f"{OUT}/e1_results.json", "w"), indent=1)
    print("combined")


if __name__ == "__main__":
    combine() if sys.argv[1] == "combine" else run_bench(sys.argv[2])
