"""Q3: offline analysis of z-scored GRPO advantages for abstention. No training. S0 rollouts held fixed, only the reward varies."""
import json, sys
import numpy as np
from scipy.stats import rankdata

ROOT = "/home/s2sadesh/"
EVAL = ROOT + "master-thesis-asymmetric-training/outputs/eval/qwen2_5_1_5b_%s_float16_%s_rollouts.jsonl"
E1 = ROOT + "fear_followups_20261018/e1/bench_%s.json"
TS = dict(P0=0.0, S0=.50, C1=.60, S1=.67, C2=.75, S2=.80, S3=.91, S4=1.0)
BENCHES = ["math_test", "gsm8k_test"]
K, G, NB, FLOOR, MIN_SCORED = 16, 8, 1000, 1e-6, 1
rng = np.random.default_rng(0)


def adv(r):
    """z-score within group, sample std (ddof=1), std floor -> zero advantage."""
    s = r.std(ddof=1)
    return np.zeros_like(r) if s < FLOOR else (r - r.mean()) / s


def load(model, bench):
    """problem_id -> {sample_idx: (abstained, correct, extracted)} for the first K samples."""
    d = {}
    for l in open(EVAL % (model, bench)):
        r = json.loads(l)
        if r["sample_idx"] < K:
            d.setdefault(r["problem_id"], {})[r["sample_idx"]] = (bool(r["abstained"]), bool(r["correct"]), r["extracted"])
    return d


def auroc(pos, neg):
    """P(score_pos > score_neg) + .5 ties; pos = hard-problem abstain advantages."""
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    rk = rankdata(np.concatenate([pos, neg]))
    return float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def summ(pp, idx):
    e = np.concatenate([pp[i][0] for i in idx])
    h = np.concatenate([pp[i][1] for i in idx])
    m = lambda x: float(x.mean()) if len(x) else np.nan
    return dict(mean_easy=m(e), mean_hard=m(h), frac_pos_easy=m(e > 0), frac_pos_hard=m(h > 0),
                auroc=auroc(h, e), n_abst_easy=int(len(e)), n_abst_hard=int(len(h)))


def boot(pp, idx):
    """problem-level bootstrap. pp: per problem (easy_abstain_adv, hard_abstain_adv)."""
    pt = summ(pp, idx)
    bs = {k: [] for k in pt}
    for _ in range(NB):
        s = summ(pp, rng.choice(idx, len(idx)))
        for k in pt:
            bs[k].append(s[k])
    ci = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in bs.items() if not k.startswith("n_")}
    return dict(point=pt, ci=ci)


res = {}
for b in BENCHES:
    s0, base = load("S0", b), load("base", b)
    bj = json.load(open(E1 % b))
    probe = {(p, i): v for p, i, v in zip(bj["problem_id"], bj["sample_idx"], bj["oof_nested"])}
    # easy = base p_hat >= .5 over base committed samples (as in E2); problems without any committed sample are excluded
    keep, easy = [], {}
    for p in sorted(s0):
        bc = [v[1] for v in base.get(p, {}).values() if not v[0]]
        if bc:
            keep.append(p); easy[p] = np.mean(bc) >= .5
    pidx = {p: n for n, p in enumerate(keep)}
    # two groups of G per problem: sample_idx 0-7 and 8-15; sample = (idx, abstained, correct, extracted)
    groups = [(p, [(i,) + s0[p][i] for i in range(g * G, (g + 1) * G)]) for p in keep for g in range(K // G)]

    def run(tfun):
        """tfun(p, ans) -> abstain reward for the group, or None to skip the group."""
        pp = [([], []) for _ in keep]
        cw_n, cw_ok, mc, mw = 0, 0, [], []
        for p, smp in groups:
            ans = [s for s in smp if not s[1]]
            t = tfun(p, ans)
            if t is None:
                continue
            a = adv(np.array([t if s[1] else float(s[2]) for s in smp]))
            ab = a[[s[1] for s in smp]]
            if len(ab):
                pp[pidx[p]][0 if easy[p] else 1].append(ab)
            cor = np.array([(not s[1]) and s[2] for s in smp]); wr = np.array([(not s[1]) and not s[2] for s in smp])
            if cor.any() and wr.any():
                cw_n += 1; cw_ok += bool(a[cor].min() > a[wr].max()); mc.append(a[cor].mean()); mw.append(a[wr].mean())
        pp = [(np.concatenate(e) if e else np.zeros(0), np.concatenate(h) if h else np.zeros(0)) for e, h in pp]
        return pp, dict(groups_with_correct_and_wrong=cw_n, frac_correct_gt_wrong=cw_ok / max(cw_n, 1),
                        mean_adv_correct=float(np.mean(mc)), mean_adv_wrong=float(np.mean(mw)))

    allidx = np.arange(len(keep))
    out = dict(n_problems=len(keep), n_easy=int(sum(easy.values())), n_groups=len(groups),
               n_excluded_no_base_commit=len(s0) - len(keep),
               s0_abstain_rate=float(np.mean([s[1] for _, sm in groups for s in sm])))

    out["A_fixed_t"] = {}
    for c, t in TS.items():
        pp, chk = run(lambda p, ans, t=t: t)
        out["A_fixed_t"][c] = dict(t=t, **boot(pp, allidx), checks=chk)
        print(b, "A", c, flush=True)

    # B1: t_q = 1 - fraction of the group's answered samples that agree with the group's majority extracted answer
    def sc(p, ans):
        if not ans:
            return None
        e = [str(s[3]).strip() for s in ans]
        return max(e.count(x) for x in set(e)) / len(e)
    pp, chk = run(lambda p, ans: None if sc(p, ans) is None else 1 - sc(p, ans))
    cs = [sc(p, [s for s in smp if not s[1]]) for p, smp in groups]
    out["B1_self_consistency"] = dict(mean_t_q=float(1 - np.mean([c for c in cs if c is not None])), **boot(pp, allidx), checks=chk)
    print(b, "B1", flush=True)

    # B2: t_q = 1 - mean E1 out-of-fold probe score over the group's answered samples that have a score
    def pr(p, ans):
        v = [probe[(p, s[0])] for s in ans if (p, s[0]) in probe]
        return float(np.mean(v)) if len(v) >= MIN_SCORED else None
    nscored = [sum((p, s[0]) in probe for s in smp if not s[1]) for p, smp in groups]
    scored = [(p, smp) for p, smp in groups if pr(p, [s for s in smp if not s[1]]) is not None]
    n_ab = sum(s[1] for _, smp in scored for s in smp)
    out["B2_probe"] = dict(n_probe_rows=len(probe), groups_with_score=len(scored), groups_total=len(groups),
                           mean_scored_answers_per_group=float(np.mean(nscored)), abstained_samples_in_scored_groups=int(n_ab))
    if n_ab >= 50:
        pp, chk = run(lambda p, ans: None if pr(p, ans) is None else 1 - pr(p, ans))
        idx = np.array([i for i, (e, h) in enumerate(pp) if len(e) + len(h)])
        out["B2_probe"].update(boot(pp, idx), checks=chk)
        out["B2_probe"]["fixed_t_same_groups"] = {
            c: boot(run(lambda p, ans, t=t: None if pr(p, ans) is None else t)[0], idx)["point"] for c, t in TS.items()}
    print(b, "B2", len(scored), flush=True)

    # C: groups with exactly two outcome types: advantages must be identical for all interior t (0 < t < 1)
    interior = [t for t in TS.values() if 0 < t < 1]
    n2, same, t0d, t1d = 0, 0, 0, 0
    for p, smp in groups:
        typ = ["A" if s[1] else ("C" if s[2] else "W") for s in smp]
        if len(set(typ)) != 2:
            continue
        A = lambda t: adv(np.array([t if y == "A" else (1.0 if y == "C" else 0.0) for y in typ]))
        ref = A(interior[0]); n2 += 1
        same += all(np.allclose(A(t), ref) for t in interior)
        t0d += not np.allclose(A(0.0), ref); t1d += not np.allclose(A(1.0), ref)
    out["C_two_outcome_groups"] = dict(n_groups=n2, identical_for_interior_t=int(same), interior_t=interior,
                                       n_differing_at_t0=int(t0d), n_differing_at_t1=int(t1d))
    res[b] = out

json.dump(dict(K=K, G=G, bootstrap=NB, std_floor=FLOOR, note="offline; S0 rollouts fixed; no training", benches=res), open(sys.argv[1], "w"), indent=1)
