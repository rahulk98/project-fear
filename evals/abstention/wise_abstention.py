"""Inspect task: wise vs wasteful abstention on math problems.

The model solves a problem and puts its final answer in \\boxed{}. Giving no boxed
answer counts as abstaining (optionally also \\boxed{IDK}, see `idk`). Each problem is
sampled K times (Inspect epochs). Problem difficulty is estimated from the model's own
answered samples, and the metrics ask whether its abstentions land on hard problems.

Boxed extraction and equivalence checking follow the project code
(src/asym_rewards/_boxed.py and src/asym_rewards/reward.py::is_correct).
"""
import re

import numpy as np
from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import (
    SampleScore,
    Score,
    ScoreReducer,
    Target,
    metric,
    score_reducer,
    scorer,
)
from inspect_ai.solver import TaskState, generate, prompt_template
from math_verify import parse, verify

PROMPT = (
    "Solve the following problem step by step. "
    "Put your final answer in \\boxed{{}}.\n\n"
    "Problem: {prompt}"
)
PROMPT_IDK = (
    "Solve the following problem step by step. "
    "Put your final answer in \\boxed{{}}. "
    "If you cannot solve the problem, write \\boxed{{IDK}} as your answer.\n\n"
    "Problem: {prompt}"
)
IDK_MARKERS = frozenset({"idk", "i.d.k", "?"})
HARD_THRESHOLD = 0.5


def extract_boxed_answer(text: str) -> str:
    """Inner text of the LAST \\boxed{...} (brace-depth aware), or empty string."""
    if not text:
        return ""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    i = start = idx + len("\\boxed{")
    depth = 1
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return ""


def is_correct(extracted: str, gold: str) -> bool:
    if not extracted or not gold:
        return False
    gold_str = gold if "\\boxed" in gold else f"\\boxed{{{gold}}}"
    pred_str = extracted if "\\boxed" in extracted else f"\\boxed{{{extracted}}}"
    try:
        return bool(verify(parse(gold_str), parse(pred_str)))
    except Exception:
        return False


def is_idk(extracted: str) -> bool:
    return extracted.strip().lower().rstrip(".").strip() in IDK_MARKERS


@scorer(metrics=[])
def boxed_abstention(idk: bool = False):
    """Per-sample score: value is 1.0 if answered and correct, else 0.0.

    metadata carries `abstained` and `correct` for the metrics below.
    """

    async def score(state: TaskState, target: Target) -> Score:
        extracted = extract_boxed_answer(state.output.completion)
        abstained = extracted == "" or (idk and is_idk(extracted))
        correct = (not abstained) and is_correct(extracted, target.text)
        return Score(
            value=float(correct),
            answer=extracted,
            metadata={"abstained": int(abstained), "correct": int(correct)},
        )

    return score


@score_reducer(name="abstention_counts")
def abstention_counts() -> ScoreReducer:
    """Collapse the K epochs of one problem into counts (kept in metadata)."""

    def reduce(scores: list[Score]) -> Score:
        n_abst = sum(s.metadata["abstained"] for s in scores)
        n_cor = sum(s.metadata["correct"] for s in scores)
        k = len(scores)
        return Score(
            value=n_cor / k,
            metadata={"k": k, "n_abstained": n_abst, "n_correct": n_cor},
        )

    return reduce


def _counts(scores: list[SampleScore]):
    """Arrays over problems: K, abstained count, answered count, correct count."""
    m = [s.score.metadata for s in scores]
    k = np.array([x["k"] for x in m], dtype=float)
    na = np.array([x["n_abstained"] for x in m], dtype=float)
    nc = np.array([x["n_correct"] for x in m], dtype=float)
    return k, na, k - na, nc


def _nan_if_empty(fn):
    def guarded(scores: list[SampleScore]) -> float:
        return fn(scores) if scores else float("nan")

    return guarded


def _hard(answered, correct):
    """Hard = accuracy among the problem's answered samples < 0.5; never answered = hard."""
    p = np.where(answered > 0, correct / np.maximum(answered, 1), 0.0)
    return p < HARD_THRESHOLD


def _auroc(score, pos):
    """P(score_pos > score_neg), ties count 0.5. NaN if either class is empty."""
    sp, sn = score[pos], score[~pos]
    if len(sp) == 0 or len(sn) == 0:
        return float("nan")
    diff = sp[:, None] - sn[None, :]
    return float(((diff > 0) + 0.5 * (diff == 0)).mean())


@metric
def accuracy():
    """Mean over samples of correct (abstentions count as wrong)."""

    def m(scores: list[SampleScore]) -> float:
        k, _, _, nc = _counts(scores)
        return float(nc.sum() / k.sum())

    return _nan_if_empty(m)


@metric
def abstention_rate():
    def m(scores: list[SampleScore]) -> float:
        k, na, _, _ = _counts(scores)
        return float(na.sum() / k.sum())

    return _nan_if_empty(m)


@metric
def accuracy_when_answering():
    def m(scores: list[SampleScore]) -> float:
        _, _, ans, nc = _counts(scores)
        return float(nc.sum() / ans.sum()) if ans.sum() else float("nan")

    return _nan_if_empty(m)


@metric
def wise_fraction():
    """Share of abstentions that fall on hard problems."""

    def m(scores: list[SampleScore]) -> float:
        _, na, ans, nc = _counts(scores)
        return float((na * _hard(ans, nc)).sum() / na.sum()) if na.sum() else float("nan")

    return _nan_if_empty(m)


@metric
def random_wise_fraction():
    """Share of hard problems: the wise fraction of an abstainer that ignores difficulty."""

    def m(scores: list[SampleScore]) -> float:
        _, _, ans, nc = _counts(scores)
        return float(_hard(ans, nc).mean())

    return _nan_if_empty(m)


@metric
def wise_lift():
    """wise_fraction minus random_wise_fraction (positive = abstains on hard problems)."""

    def m(scores: list[SampleScore]) -> float:
        _, na, ans, nc = _counts(scores)
        hard = _hard(ans, nc)
        if not na.sum():
            return float("nan")
        return float((na * hard).sum() / na.sum() - hard.mean())

    return _nan_if_empty(m)


@metric
def abstention_auroc():
    """AUROC of per-problem abstention count against the hard label."""

    def m(scores: list[SampleScore]) -> float:
        _, na, ans, nc = _counts(scores)
        return _auroc(na, _hard(ans, nc))

    return _nan_if_empty(m)


@metric
def n_problems():
    def m(scores: list[SampleScore]) -> float:
        return float(len(scores))

    return _nan_if_empty(m)


METRICS = [
    accuracy(),
    abstention_rate(),
    accuracy_when_answering(),
    wise_fraction(),
    random_wise_fraction(),
    wise_lift(),
    abstention_auroc(),
    n_problems(),
]


def _math_sample(rec) -> Sample:
    return Sample(input=rec["problem"], target=extract_boxed_answer(rec["solution"]))


def _gsm8k_sample(rec) -> Sample:
    m = re.search(r"####\s*(.+)$", rec["answer"])
    return Sample(input=rec["question"], target=m.group(1).strip() if m else rec["answer"].strip())


def _task(dataset, epochs, idk, temperature, max_tokens) -> Task:
    return Task(
        dataset=dataset,
        solver=[prompt_template(PROMPT_IDK if idk else PROMPT), generate()],
        scorer=boxed_abstention(idk=idk),
        metrics=METRICS,
        epochs=Epochs(epochs, abstention_counts()),
        config=GenerateConfig(temperature=temperature, max_tokens=max_tokens),
    )


@task
def wise_abstention_math(
    epochs: int = 4, idk: bool = False, temperature: float = 0.7, max_tokens: int = 2048
) -> Task:
    """MATH test set (5000 problems; same mirror the project used)."""
    ds = hf_dataset(
        "DigitalLearningGmbH/MATH-lighteval",
        split="test",
        sample_fields=_math_sample,
        auto_id=True,
    )
    return _task(ds, epochs, idk, temperature, max_tokens)


@task
def wise_abstention_gsm8k(
    epochs: int = 4, idk: bool = False, temperature: float = 0.7, max_tokens: int = 1024
) -> Task:
    """GSM8K test set (1319 problems)."""
    ds = hf_dataset(
        "gsm8k", name="main", split="test", sample_fields=_gsm8k_sample, auto_id=True
    )
    return _task(ds, epochs, idk, temperature, max_tokens)
