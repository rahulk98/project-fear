"""Unit tests for the abstention metrics on synthetic data with known answers."""
import numpy as np
from inspect_ai.scorer import Score, SampleScore

import wise_abstention as w


def prob(k, n_abst, n_cor):
    return SampleScore(
        score=Score(value=n_cor / k, metadata={"k": k, "n_abstained": n_abst, "n_correct": n_cor})
    )


# 4 problems, K=4.
#  A: answered 4, correct 4        -> easy
#  B: answered 4, correct 0        -> hard
#  C: answered 2, correct 0, abst 2 -> hard (acc among answered 0)
#  D: answered 0, abst 4           -> hard (never answered)
SCORES = [prob(4, 0, 4), prob(4, 0, 0), prob(4, 2, 0), prob(4, 4, 0)]


def test_basic_rates():
    assert w.abstention_rate()(SCORES) == 6 / 16
    assert w.accuracy()(SCORES) == 4 / 16
    assert w.accuracy_when_answering()(SCORES) == 4 / 10
    assert w.n_problems()(SCORES) == 4


def test_wise_fraction_all_abstentions_on_hard():
    assert w.wise_fraction()(SCORES) == 1.0
    assert w.random_wise_fraction()(SCORES) == 3 / 4
    assert w.wise_lift()(SCORES) == 0.25


def test_wise_fraction_partial():
    s = [prob(4, 0, 0), prob(4, 1, 3)]
    # first problem hard (0/4), second easy (3/3): 1 abstention, on easy
    assert w.wise_fraction()(s) == 0.0
    assert w.random_wise_fraction()(s) == 0.5


def test_boundary_half_is_not_hard():
    # accuracy among answered exactly 0.5 -> not hard
    s = [prob(4, 0, 2), prob(4, 1, 0)]
    assert w.random_wise_fraction()(s) == 0.5
    assert w.wise_fraction()(s) == 1.0  # the abstention is on the second (hard) problem


def test_auroc_perfect_and_inverted():
    # abstain counts rise with hardness: perfect separation
    perfect = [prob(4, 0, 4), prob(4, 1, 0), prob(4, 4, 0)]
    # problem 2: answered 3, correct 0 -> hard; problem 3: never answered -> hard; problem 1 easy
    assert w.abstention_auroc()(perfect) == 1.0
    inverted = [prob(4, 4, 0), prob(4, 0, 4)]
    # first: never answered (hard, 4 abstentions); second easy, 0 abstentions -> still perfect
    assert w.abstention_auroc()(inverted) == 1.0
    wrong_way = [prob(4, 0, 0), prob(4, 3, 1)]
    # first: hard, 0 abst; second: answered 1, correct 1 -> easy, 3 abst -> AUROC 0
    assert w.abstention_auroc()(wrong_way) == 0.0


def test_auroc_ties_and_nan():
    tied = [prob(4, 1, 0), prob(4, 1, 3)]
    # hard (answered 3, correct 0) vs easy (answered 3, correct 3); equal counts -> 0.5
    assert w.abstention_auroc()(tied) == 0.5
    assert np.isnan(w.abstention_auroc()([prob(4, 0, 4), prob(4, 0, 4)]))


def test_no_abstention_gives_nan_wise():
    s = [prob(4, 0, 4), prob(4, 0, 0)]
    assert np.isnan(w.wise_fraction()(s))


def test_extraction_and_equivalence():
    assert w.extract_boxed_answer("so \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert w.extract_boxed_answer("no answer") == ""
    assert w.extract_boxed_answer("\\boxed{3} then \\boxed{4}") == "4"
    assert w.is_correct("0.5", "\\frac{1}{2}")
    assert not w.is_correct("3", "4")
    assert w.is_idk("IDK.") and not w.is_idk("12")


def test_reducer():
    def sc(a, c):
        return Score(value=float(c), metadata={"abstained": a, "correct": c})

    out = w.abstention_counts()([sc(0, 1), sc(1, 0), sc(0, 0), sc(0, 1)])
    assert out.metadata == {"k": 4, "n_abstained": 1, "n_correct": 2}
def test_empty():
    assert np.isnan(w.accuracy()([])) and np.isnan(w.wise_lift()([]))
