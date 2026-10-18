"""Q5: classify abstained completions (no boxed final answer) with deterministic rules.
Run on the server:
  python q5_abstention_audit.py classify out_dir      -> results.json, sample_blind.json, sample_rule.json
  python q5_abstention_audit.py agree out_dir labels.json   (labels.json: {id: category}, manual labels)
Generation cap: max_completion_length 1024 tokens (configs/training.yaml). Prompt: 'Solve the following problem step by step.
Put your final answer in \\boxed{}.' (scripts/eval_calibration.py PROMPT_TEMPLATE)."""
import json, sys, re, zlib, random, collections, os
import numpy as np

EVAL = os.environ.get("FEAR_EVAL_DIR", "outputs/eval") + "/"
TOK = os.environ.get("FEAR_MODELS_DIR", "models") + "/qwen2_5_1_5b"
CONDS = ["S0", "S3", "S4", "P0"]
BENCH = ["math_test", "gsm8k_test"]
K, CAP, MARGIN = 16, 1024, 8
CATS = ["truncated", "explicit_refusal", "answer_unboxed", "malformed_or_degenerate", "code_program", "unterminated_early_stop", "other"]

REFUSAL = [r"i don'?t know", r"i do not know", r"i'?m not sure", r"i am not sure", r"cannot determine", r"can'?not be determined",
           r"can'?t determine", r"not enough information", r"insufficient information", r"unable to (determine|solve|answer|find|provide|compute|calculate)",
           r"i (cannot|can'?t|am unable to|'?m unable to) (solve|answer|determine|provide|help|find)", r"no (way|means) to (determine|know)",
           r"cannot be (solved|answered|found)", r"\bIDK\b",
           r"double-check (the|your) (problem|information|question)", r"(please|kindly) (check|clarify|provide|verify)", r"i apologi[sz]e",
           r"(we|i) (can'?t|cannot) (find|solve|provide|calculate|proceed|run)", r"problem (statement )?(is|may be|might be) (incorrect|flawed|invalid)",
           r"no solution", r"not possible to (determine|find|solve)", r"need more information", r"without (further|more|additional) (information|context)"]
REF_RE = re.compile("|".join(REFUSAL), re.I)
NONLATIN = re.compile(r"[\u0400-\u04FF\u0600-\u06FF\u0E00-\u0E7F\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]")
CODE_RE = re.compile(r"(^|\n)\s*(import sympy|import math|from sympy|def \w+\(|print\()", re.M)


STRONG_RE = re.compile(r"(the )?(final |correct )?answer( to [^.\n]{0,60})? (is|=)|^\s*####|\banswer:", re.I | re.M)
ECHO_START = re.compile(r"\n?\s*You are an AI assistant[^\n]*$|\n?\s*(User will you give you|Your task is to answer as faithfully|Your goal is to complete the task|You will be given a task)[^\n]*$", re.I)
CODE_HDR = re.compile(r"^\s*Let'?s (write|program|use|create)[^.\n]{0,40}(Python|code|program)|```python|^(# Step \d+[:.].*\n?){3,}", re.I | re.M)


CONCL_RE = re.compile(r"\b(so|therefore|thus|hence|conclusion|in total|total|result|final|answer|means)\b", re.I)


def has_answer_text(t):
    """a solution that finished: a number or math in the last 300 chars, or an explicit answer/conclusion cue."""
    tl = t[-300:]
    return bool(re.search(r"\d|\\frac|=", tl) or STRONG_RE.search(tl) or CONCL_RE.search(tl))


def classify(text, ntok):
    """returns (category, flags)"""
    t = text.strip()
    if ntok >= CAP - MARGIN:
        tl = t[-600:]
        loop = len(tl) > 200 and (len(zlib.compress(tl.encode())) / len(tl.encode()) < 0.2)
        return "truncated", dict(loop=bool(loop))
    echo = bool(ECHO_START.search(t))
    if echo:
        t = ECHO_START.sub("", t).strip()
    fl = dict(echo=echo)
    if len(t) < 80 and not re.search(r"\d.*\d|=", t):
        return "malformed_or_degenerate", fl          # empty or a stray fragment, no solution content
    if STRONG_RE.search(t[-120:]):
        return "answer_unboxed", fl
    if REF_RE.search(t[-250:]):
        return "explicit_refusal", fl
    if CODE_HDR.search(t) or CODE_RE.search(t):
        return "code_program", fl
    if re.search(r"(?<!in )(?<!in \$)\\boxed\{\s*\}|\\boxed\{[^{}]*$", t):
        return "malformed_or_degenerate", fl            # empty or unclosed box (a prompt-restating 'in \\boxed{}' is not one)
    if len(t) > 300 and len(zlib.compress(t.encode())) / len(t.encode()) < 0.12:
        return "malformed_or_degenerate", fl
    nl = len(NONLATIN.findall(t))
    if nl > 0.2 * len(t) or (len(t) > 200 and nl > 0.05 * len(t)):
        return "malformed_or_degenerate", fl
    lines = [l for l in t.split("\n") if len(l.strip()) >= 8]
    if len(lines) >= 6 and collections.Counter(lines).most_common(1)[0][1] >= 5:
        return "malformed_or_degenerate", fl
    if "[asy]" in t[-400:] or (t.endswith("```") and "asy" in t[-600:]):
        return "other", fl                               # diagram source dumped instead of an answer
    if has_answer_text(t):
        return "answer_unboxed", fl
    if not re.search(r"[.!?$)\]}]\s*$|\\\]\s*$", t):
        return "unterminated_early_stop", fl
    return "other", fl


def path(c, b):
    return f"{EVAL}qwen2_5_1_5b_{ {'P0': 'P0_k32'}.get(c, c) }_float16_{b}_rollouts.jsonl"


def main_classify(out):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK)
    res, rows = {}, []
    for c in CONDS:
        for b in BENCH:
            ab, tot = [], 0
            for l in open(path(c, b)):
                r = json.loads(l)
                if r["sample_idx"] >= K:
                    continue
                tot += 1
                if r["abstained"]:
                    ab.append((r["problem_id"], r["sample_idx"], r["completion"]))
            nt = [len(x) for x in tok([a[2] for a in ab])["input_ids"]]
            cf = [classify(a[2], n) for a, n in zip(ab, nt)]
            cats = [x[0] for x in cf]
            cnt = collections.Counter(cats)
            nloop = sum(1 for x in cf if x[0] == "truncated" and x[1]["loop"])
            necho = sum(1 for x in cf if x[1].get("echo"))
            L = np.array(nt)
            near = {"==1024": int((L == CAP).sum()), "1016-1023": int(((L >= 1016) & (L < 1024)).sum()), ">1024": int((L > CAP).sum()),
                    "1000-1015": int(((L >= 1000) & (L < 1016)).sum()), "900-999": int(((L >= 900) & (L < 1000)).sum())}
            res[f"{c}/{b}"] = dict(n_samples=tot, n_abstained=len(ab), abst_rate=len(ab) / tot,
                                   counts={k: cnt.get(k, 0) for k in CATS}, shares={k: cnt.get(k, 0) / len(ab) for k in CATS},
                                   n_trunc_looping=nloop, n_with_echo_suffix=necho, len_pctl={p: float(np.percentile(L, p)) for p in (5, 25, 50, 75, 95)}, near_cap=near)
            for a, n, k in zip(ab, nt, cats):
                rows.append(dict(cond=c, bench=b, pid=a[0], idx=a[1], ntok=n, cat=k, text=a[2]))
            print(c, b, len(ab), dict(cnt), flush=True)
    ex = {}
    rng = random.Random(0)
    for k in CATS:
        pool = [r for r in rows if r["cat"] == k]
        ex[k] = [dict(cond=r["cond"], bench=r["bench"], pid=r["pid"], idx=r["idx"], ntok=r["ntok"], echo_suffix_stripped=bool(ECHO_START.search(r["text"].strip())), tail=ECHO_START.sub("", r["text"].strip()).strip()[-200:])
                 for r in rng.sample(pool, min(5, len(pool)))]
    # stratified validation sample: 9 per category (all if fewer), seed 0, pooled over conditions
    rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    samp = []
    for k in CATS:
        pool = [r for r in rows if r["cat"] == k]
        samp += rng.sample(pool, min(9, len(pool)))
    rng.shuffle(samp)
    blind = {str(i): dict(cond=r["cond"], bench=r["bench"], ntok=r["ntok"], head=r["text"].strip()[:150], tail=r["text"].strip()[-350:], nchar=len(r["text"])) for i, r in enumerate(samp)}
    rule = {str(i): r["cat"] for i, r in enumerate(samp)}
    json.dump(dict(cap=CAP, margin=MARGIN, refusal_patterns=REFUSAL, unboxed_cue_pattern=STRONG_RE.pattern, echo_suffix_pattern=ECHO_START.pattern, code_pattern=CODE_HDR.pattern, cells=res, examples=ex), open(f"{out}/results.json", "w"), indent=1)
    json.dump(blind, open(f"{out}/sample_blind.json", "w"), indent=1)
    json.dump(rule, open(f"{out}/sample_rule.json", "w"))


def main_agree(out, labels):
    lab = json.load(open(labels))
    rule = json.load(open(f"{out}/sample_rule.json"))
    r = json.load(open(f"{out}/results.json"))
    ids = sorted(lab, key=int)
    agree = sum(lab[i] == rule[i] for i in ids)
    per = {}
    for k in CATS:
        ii = [i for i in ids if rule[i] == k]
        per[k] = dict(n=len(ii), agree=sum(lab[i] == k for i in ii))
    conf = collections.Counter((rule[i], lab[i]) for i in ids if rule[i] != lab[i])
    pop = {k: sum(v["counts"][k] for v in r["cells"].values()) for k in CATS}
    wagree = sum(pop[k] * per[k]["agree"] / per[k]["n"] for k in CATS if per[k]["n"]) / sum(pop[k] for k in CATS if per[k]["n"])
    r["validation"] = dict(n=len(ids), agree=agree, population_weighted_agreement=wagree, population_counts=pop, per_rule_category=per, disagreements=[f"{a}->{b}: {n}" for (a, b), n in conf.items()],
                           disagree_ids=[i for i in ids if lab[i] != rule[i]])
    json.dump(r, open(f"{out}/results.json", "w"), indent=1)
    print(r["validation"])


if __name__ == "__main__":
    (main_classify if sys.argv[1] == "classify" else lambda o: main_agree(o, sys.argv[3]))(sys.argv[2])
