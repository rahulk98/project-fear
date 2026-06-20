"""Asymmetric reward functions for GRPO (Phase 2).

Defines reward families used across the FEAR / CWAR / D1 sweeps:

- ``family="asym"`` (default): the 7 FEAR conditions S0-S4 + C1-C2, plus the
  Zhu reproduction Z3 and the confidence-weighted CW. Abstention is implicit
  (no extractable ``\\boxed{...}``); reward depends only on correct / wrong /
  no-answer.
- ``family="d1"``: five-outcome dispatch with an explicit IDK marker inside
  the last ``\\boxed{...}``. Outcomes: correct, wrong, wise_idk (IDK with the
  forced-decode counterfactual NOT matching gold), wasteful_idk (IDK with the
  forced-decode counterfactual matching gold), silent (no extractable box).
  Pair with ``weighting="confidence"`` for the CW-D1 composite where conf
  scaling applies only to correct/wrong (attempt) channels.

The reward function exposes a per-batch telemetry dict ``_LAST_BATCH_TELEMETRY``
so the wandb trainer-callback can read outcome labels, per-channel rewards,
conf_i values, and forced-decode timings without trying to recover them from
post-normalized scalar advantages (codex-review-2026-05-27 P2 #2).
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

from math_verify import parse, verify

# Re-exported so existing `from asym_rewards.reward import extract_boxed_answer`
# call sites (generate_traces, smoke_judge) keep working. Single definition
# lives in _boxed (dependency-free) and is shared with judge.py.
from asym_rewards._boxed import extract_boxed_answer  # noqa: F401

_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

OUTCOME_CORRECT = "correct"
OUTCOME_WRONG = "wrong"
OUTCOME_WISE_IDK = "wise_idk"
OUTCOME_WASTEFUL_IDK = "wasteful_idk"
OUTCOME_SILENT = "silent"
# Silent-channel bifurcation (FD1xsil and later): when extend_silent_forced_decode
# is enabled on a d1 condition, silent samples ALSO get a forced-decode probe.
# If the forced decode would have produced the correct answer, the rollout
# is labeled silent_wasteful (model knew but did not commit). Otherwise it is
# silent_wise (legitimate uncertainty, still mildly penalized to encourage
# explicit IDK over implicit silence).
OUTCOME_SILENT_WISE = "silent_wise"
OUTCOME_SILENT_WASTEFUL = "silent_wasteful"

OUTCOME_CLASSES = (
    OUTCOME_CORRECT,
    OUTCOME_WRONG,
    OUTCOME_WISE_IDK,
    OUTCOME_WASTEFUL_IDK,
    OUTCOME_SILENT,
    OUTCOME_SILENT_WISE,
    OUTCOME_SILENT_WASTEFUL,
)

# Position-locked IDK markers; matched against the lowercase-stripped content
# of the LAST \\boxed{...} ONLY. Rejects CoT-leaked "I don't know" outside the
# answer box.
_IDK_MARKERS = frozenset({"idk", "i.d.k", "?"})

# Per-batch telemetry side channel. Populated by every D1 reward call and
# read by the trainer-side wandb callback after `_prepare_inputs`. Keys:
#   "outcomes"        list[str]  per-sample outcome label
#   "channel_rewards" list[float] per-sample reward BEFORE any trainer-side
#                                 conf rescale (so CW-D1 plots stay
#                                 decomposable)
#   "idk_flags"       list[bool]
#   "forced_decode_s" list[float] seconds, 0.0 when not invoked
#   "forced_correct"  list[Optional[bool]] None when forced_decode skipped
# The asym/CW reward path populates a reduced version with outcomes restricted
# to {correct, wrong, silent} so downstream aggregation is uniform.
_LAST_BATCH_TELEMETRY: dict[str, list] = {}


CONDITIONS: dict[str, dict[str, Any]] = {
    # ---------- asym family (existing 7 + Z3 + CW) ----------
    "S0": {"family": "asym", "alpha": 1.0, "beta": 1.0, "abstain": 0.0, "name": "symmetric"},
    "S1": {"family": "asym", "alpha": 1.0, "beta": 2.0, "abstain": 0.0, "name": "mild"},
    "S2": {"family": "asym", "alpha": 0.5, "beta": 2.0, "abstain": 0.0, "name": "moderate"},
    "S3": {"family": "asym", "alpha": 0.1, "beta": 1.0, "abstain": 0.0, "name": "zhu_lambda01"},
    "S4": {"family": "asym", "alpha": 0.0, "beta": 1.0, "abstain": 0.0, "name": "pure_nsr"},
    # P0 = positive-only RLVR baseline (alpha=1, beta=0, abstain=0). Standard
    # "company-style" reward used in most RLVR papers (OpenAI/DeepSeek/Qwen Math
    # style: +1 correct, 0 otherwise, no penalty, no abstention incentive).
    # Implicit threshold = beta/(alpha+beta) = 0 -> model commits on every prompt.
    "P0": {"family": "asym", "alpha": 1.0, "beta": 0.0, "abstain": 0.0, "name": "positive_only"},
    "C1": {"family": "asym", "alpha": 1.0, "beta": 1.0, "abstain": 0.2, "name": "explicit_abstain_mild"},
    "C2": {"family": "asym", "alpha": 1.0, "beta": 1.0, "abstain": 0.5, "name": "explicit_abstain_strong"},
    # Z3 = Zhu reproduction with forced commit (silent < -beta).
    "Z3": {"family": "asym", "alpha": 0.1, "beta": 1.0, "abstain": -1.5, "name": "zhu_repro_force_commit"},
    # CW = confidence-weighted reward on the asym family (CWAR).
    "CW": {"family": "asym", "alpha": 1.0, "beta": 1.0, "abstain": 0.0,
           "weighting": "confidence", "name": "confidence_weighted_symmetric"},
    # FCW = FEAR + CW: asymmetric attempt channel (beta=2) with CWAR conf
    # weighting. Direct A/B with CW isolates attempt-channel asymmetry under
    # matched confidence scaling.
    "FCW": {"family": "asym", "alpha": 1.0, "beta": 2.0, "abstain": 0.0,
            "weighting": "confidence", "name": "fear_cw_composite"},
    # OCP: Overconfidence Penalty. Pure-fear extension that adds an extra
    # penalty on confident-wrong samples only. Piecewise:
    #   R_wrong = -beta - lambda_oc * max(0, conf - tau_oc)
    # Uncertain-wrong (conf <= tau) is identical to S0. Confident-wrong
    # (conf > tau) gets up to (lambda_oc * (1 - tau_oc)) of extra fear.
    # Correct and silent channels are S0. Breaks Finding O affine-invariance
    # via per-sample conf; targets the confident-wrong subset specifically
    # (the per-attempt-accuracy killer at 0.5B).
    "OCP": {"family": "asym", "alpha": 1.0, "beta": 1.0, "abstain": 0.0,
            "weighting": "confidence", "wrong_rescale": "ocp",
            "lambda_oc": 2.0, "tau_oc": 0.5,
            "name": "overconfidence_penalty"},
    # CMF: Confidence-Modulated Fear. Smooth multiplicative scaling on the
    # wrong channel only (mirror-CW on the punishment side):
    #   R_wrong = -beta * (1 + gamma_cmf * conf)
    # Correct and silent channels are S0. Distinct from CW which scales BOTH
    # channels symmetrically; CMF only touches wrong. Distinct from FCW
    # because FCW raises alpha=beta asymmetry then scales BOTH channels.
    "CMF": {"family": "asym", "alpha": 1.0, "beta": 1.0, "abstain": 0.0,
            "weighting": "confidence", "wrong_rescale": "cmf",
            "gamma_cmf": 1.0,
            "name": "confidence_modulated_fear"},
    # GRF: Group-Relative Fear. Difficulty-calibrated fear that scales the
    # wrong penalty by how many peers in the SAME rollout group were correct:
    #   R_wrong = -beta * (1 + delta_grf * peer_correct_rate)
    # peer_correct_rate = (#correct_in_group - is_self_correct) / (G - 1).
    # In an all-wrong group peer_rate=0 -> R_wrong=-beta = S0. In a mostly-
    # right group with one outlier wrong sample peer_rate=1 -> R_wrong=-2*beta.
    # No conf signal used; the group composition is the difficulty proxy.
    "GRF": {"family": "asym", "alpha": 1.0, "beta": 1.0, "abstain": 0.0,
            "group_rescale": "grf", "delta_grf": 1.0,
            "name": "group_relative_fear"},
    # ---------- d1 family (explicit IDK + forced-decode counterfactual) ----------
    # Three magnitude variants; silent penalty fixed at 1.5 so silent < wrong
    # < wasteful_idk < 0 < wise_idk < correct under symmetric mu.
    "D1low":  {"family": "d1", "alpha": 1.0, "beta": 1.0,
               "mu_wise": 0.3, "mu_wasteful": 0.3, "silent_penalty": 1.5,
               "name": "d1_low_magnitude"},
    "D1mid":  {"family": "d1", "alpha": 1.0, "beta": 1.0,
               "mu_wise": 0.5, "mu_wasteful": 0.5, "silent_penalty": 1.5,
               "name": "d1_mid_magnitude"},
    "D1asym": {"family": "d1", "alpha": 1.0, "beta": 1.0,
               "mu_wise": 0.5, "mu_wasteful": 0.4, "silent_penalty": 1.5,
               "name": "d1_asymmetric_mu"},
    # FD1xsil = FEAR + D1 + eXtended silent forced decode. Same as FD1 except
    # the silent channel (no boxed answer at all) is bifurcated by a forced
    # decode probe: if forced decode yields the gold answer the rollout is
    # labeled silent_wasteful (model knew but did not commit, penalty 2.0 =
    # same as wrong); otherwise silent_wise (legitimate uncertainty, mild
    # penalty 0.5 to motivate using explicit IDK over implicit silence).
    # silent_penalty is unused under extend_silent_forced_decode=True.
    # Designed to address d1 IDK-channel unreachability on Qwen-1.5B
    # (zero IDK emissions observed in FD1+IDK training 2026-06-01).
    "FD1xsil": {"family": "d1", "alpha": 1.0, "beta": 2.0,
                "mu_wise": 0.3, "mu_wasteful": 0.3, "silent_penalty": 2.5,
                "extend_silent_forced_decode": True,
                "silent_wise_penalty": 0.5,
                "silent_wasteful_penalty": 2.0,
                "name": "fd1_extended_silent_forced_decode"},
    # SC = self-consistency-conditioned wrong-channel penalty. Built like GRF
    # but uses LOO peer-agreement on the EXTRACTED answer strings (not on
    # correct/wrong signs). Launched with --advantage-norm raw_rloo per spec
    # (2026-05-31 batch). Surface check on P0 K=32 rollouts showed 36-44% of
    # wrong rollouts cluster on the same wrong answer at count>=2/8 (math/gsm8k).
    "SC": {"family": "asym", "alpha": 1.0, "beta": 1.0, "abstain": 0.0,
           "group_rescale": "sc", "delta_sc": 1.0, "name": "self_consistency_loo"},
    # FD1 = FEAR + D1: asymmetric attempt channel (alpha=1, beta=2) layered on
    # D1low's IDK channel (mu=0.3) for a clean A/B with D1low isolating attempt
    # asymmetry. silent_penalty scaled to 2.5 to preserve invariant
    # silent < -beta < -mu_wasteful < 0 < +mu_wise < +alpha.
    "FD1": {"family": "d1", "alpha": 1.0, "beta": 2.0,
            "mu_wise": 0.3, "mu_wasteful": 0.3, "silent_penalty": 2.5,
            "name": "fear_d1_low_magnitude"},
    # FD1eq = FD1 with silent_penalty equal to wrong penalty (beta=2.0). Designed
    # to fix the global_mean basin observed in FD1+IDK 1.5B (wandb mb9ek0yk):
    # with silent_penalty=2.5 > wrong=2.0, global_mean preserves the asymmetry
    # and signals "silent worse than wrong" -> model collapses to wrong-commit
    # (silent 0.27 -> 0.09, wrong 0.33 -> 0.88 by step 400). Equalising silent
    # and wrong removes that gradient ordering under global_mean.
    "FD1eq": {"family": "d1", "alpha": 1.0, "beta": 2.0,
              "mu_wise": 0.3, "mu_wasteful": 0.3, "silent_penalty": 2.0,
              "name": "fear_d1_silent_eq_wrong"},
    # CW-D1: D1 channels + conf scaling on attempt channels only. The trainer
    # source-rewrite reads `weighting == "confidence"` AND `family == "d1"`
    # to scale ONLY samples whose telemetry outcome is in {correct, wrong}.
    "CWD1": {"family": "d1", "weighting": "confidence",
             "alpha": 1.0, "beta": 1.0,
             "mu_wise": 0.5, "mu_wasteful": 0.5, "silent_penalty": 1.5,
             "name": "cw_d1_composite"},
    # ---------- vc family (Verifier-Calibration: judge trained on PRM800K) ----------
    # The vc family trains a Qwen-1.5B adapter to rate solution steps (YES/NO)
    # using asymmetric RLVR. The "answer" column contains "YES" or "NO" (the
    # PRM800K gold step rating); the reward is determined by whether the model's
    # boxed output agrees with the gold (correct), disagrees (wrong), or abstains
    # (silent/no boxed output). Uses the same alpha/beta/abstain structure as the
    # asym family so the existing group_rescale and telemetry machinery work
    # unchanged. See judge.py classify_vc for the per-sample outcome classifier.
    #
    # VC   : asymmetric + self-consistency LOO (group_rescale="sc", delta_sc=1.0).
    #        The SC signal penalizes wrong answers that are also the consensus
    #        wrong answer, encouraging diverse exploration of the YES/NO space.
    "VC": {"family": "vc", "alpha": 1.0, "beta": 1.0, "abstain": 0.0,
           "group_rescale": "sc", "delta_sc": 1.0,
           "name": "vc_sc_loo"},
    # VC_flat: asymmetric, no conf or SC signal. Flat penalty. Direct A/B with
    #          VC isolates the SC signal contribution.
    "VC_flat": {"family": "vc", "alpha": 1.0, "beta": 1.0, "abstain": 0.0,
                "name": "vc_flat_penalty"},
    # VC0  : no wrong-answer penalty (beta=0). Positive-only RLVR baseline for
    #        vc family; calibrates how much of VC's gain comes from fear vs
    #        pure positive reinforcement.
    "VC0": {"family": "vc", "alpha": 1.0, "beta": 0.0, "abstain": 0.0,
            "name": "vc_no_penalty"},
}


def is_correct(extracted: str, gold: str) -> bool:
    """math_verify comparison; tolerant on parse errors -> False."""
    if not extracted or not gold:
        return False
    gold_str = gold if "\\boxed" in gold else f"\\boxed{{{gold}}}"
    pred_str = extracted if "\\boxed" in extracted else f"\\boxed{{{extracted}}}"
    try:
        return bool(verify(parse(gold_str), parse(pred_str)))
    except Exception:
        return False


def score_mc(extracted: str, gold_index: str) -> bool:
    """Multiple-choice scoring for boxed letter answers."""
    if not extracted:
        return False
    try:
        if gold_index.startswith("["):
            import ast
            labels = ast.literal_eval(gold_index)
            target_idx = next(i for i, v in enumerate(labels) if v)
        else:
            target_idx = int(gold_index)
        target_letter = _LETTERS[target_idx]
    except (ValueError, IndexError, SyntaxError, StopIteration):
        return False
    return extracted.strip().upper().strip(".)") == target_letter


def family_for_rollout_path(path) -> str | None:
    """Parse the condition token from a rollout filename and return its
    `CONDITIONS[cond]["family"]`, or None if no known condition is present.

    Convention: rollout filenames are `{model}_{condition}_{dtype}_{bench}_rollouts*.jsonl`.
    Matching is underscore-wrapped to avoid `CW` colliding with `CWD1`, and
    longest-first to prefer specific (`D1mid`) over generic (`D1`-like prefixes).
    """
    name = str(path).rsplit("/", 1)[-1]
    haystack = f"_{name}"
    for cond in sorted(CONDITIONS.keys(), key=len, reverse=True):
        if f"_{cond}_" in haystack:
            return CONDITIONS[cond].get("family")
    return None


def rescore_rollouts(rollouts: list[dict], family: str | None = None) -> list[dict]:
    """Recompute extracted/correct/abstained from cached rollout completions.

    For the d1 family, an extracted IDK marker counts as abstention, matching
    d1_reward semantics. Callers should pass the trained condition's family
    so cached D1/CWD1 rollouts re-score consistently with fresh ones.
    """
    is_d1 = (family == "d1")
    rescored = []
    for r in rollouts:
        completion = r.get("completion", "")
        extracted = extract_boxed_answer(completion)
        cached_extracted = r.get("extracted")
        if cached_extracted == extracted and "correct" in r:
            correct = bool(r["correct"])
        elif r.get("scorer") == "mc":
            correct = score_mc(extracted, str(r.get("gold", "")))
        else:
            correct = is_correct(extracted, str(r.get("gold", "")))
        abstained = (extracted == "") or (is_d1 and is_idk_marker(extracted))
        if abstained:
            correct = False
        out = dict(r)
        out.update({
            "completion": completion,
            "extracted": extracted,
            "correct": correct,
            "abstained": abstained,
        })
        rescored.append(out)
    return rescored


def rescore_change_counts(original: list[dict], rescored: list[dict]) -> dict[str, int]:
    """Count cached label fields that changed after re-scoring."""
    fields = ("extracted", "correct", "abstained")
    return {
        field: sum(a.get(field) != b.get(field) for a, b in zip(original, rescored))
        for field in fields
    }


def is_abstention(text: str) -> bool:
    """Implicit definition: no extractable boxed answer = abstention.

    For the D1 family this is the SILENT class only; explicit IDK markers in
    the box are detected separately by `is_idk_marker`.
    """
    return extract_boxed_answer(text) == ""


def is_idk_marker(extracted: str) -> bool:
    """Position-locked IDK detection on the LAST `\\boxed{...}` contents.

    Returns True iff the extracted box content (after strip / lower / trailing
    period removal) matches one of {"idk", "i.d.k", "?"}. Box-locked so a
    CoT-leaked "I don't know" outside the answer box is NOT treated as IDK.
    """
    if not extracted:
        return False
    s = extracted.strip().lower().rstrip(".").strip()
    return s in _IDK_MARKERS


def asymmetric_reward(
    completion: str,
    gold: str,
    alpha: float,
    beta: float,
    abstain: float = 0.0,
) -> float:
    """R = +alpha if correct, -beta if wrong-with-answer, +abstain if no answer.

    Implicit-abstention family (S0-S4 / C1-C2 / Z3 / CW). Any extractable
    box that is not gold-equivalent counts as wrong; IDK markers are TREATED
    AS WRONG here because this family has no IDK channel. Use the `d1` family
    for explicit IDK reward.
    """
    extracted = extract_boxed_answer(completion)
    if not extracted:
        return abstain
    return alpha if is_correct(extracted, gold) else -beta


def _classify_asym(completion: str, gold: str) -> str:
    """Return one of {correct, wrong, silent} for the asym family."""
    extracted = extract_boxed_answer(completion)
    if not extracted:
        return OUTCOME_SILENT
    return OUTCOME_CORRECT if is_correct(extracted, gold) else OUTCOME_WRONG


def d1_dispatch(
    completion: str,
    gold: str,
    forced_decode_fn: Optional[Callable[[str, str], str]] = None,
    prompt: Optional[str] = None,
    extend_silent_forced_decode: bool = False,
) -> tuple[str, dict]:
    """Five-outcome (or seven-outcome) D1 dispatch + per-sample telemetry.

    When ``extend_silent_forced_decode=True`` and a silent sample is detected
    (no boxed answer at all), the forced decode is ALSO invoked to bifurcate
    silent into silent_wise (forced decode would have been wrong) vs
    silent_wasteful (forced decode would have been correct). This mirrors
    the wise_idk vs wasteful_idk distinction for explicit IDK markers.

    Args:
        completion: full generated completion text.
        gold: ground-truth answer (raw string or `\\boxed{...}` form).
        forced_decode_fn: callable (prompt, completion) -> forced answer text.
            Invoked on IDK samples, and additionally on silent samples when
            extend_silent_forced_decode=True. When None, IDK and (extended)
            silent samples default to wise (safe fallback for offline unit
            tests where vLLM is not available).
        prompt: prompt text passed to the forced-decode fn; required if
            forced_decode_fn is not None.
        extend_silent_forced_decode: when True, fires forced decode on silent
            samples too and labels them silent_wise or silent_wasteful.

    Returns:
        (outcome_label, info_dict). info_dict keys:
            forced_decode_s    : float, 0.0 when not invoked
            idk_flag           : bool, True iff the box was an IDK marker
            forced_answer      : Optional[str], None when not invoked
            forced_correct     : Optional[bool], None when not invoked
    """
    info: dict[str, Any] = {
        "forced_decode_s": 0.0,
        "idk_flag": False,
        "forced_answer": None,
        "forced_correct": None,
    }
    extracted = extract_boxed_answer(completion)
    if not extracted:
        if not extend_silent_forced_decode:
            return OUTCOME_SILENT, info
        # Extended silent path: forced decode to bifurcate wise vs wasteful.
        if forced_decode_fn is None:
            return OUTCOME_SILENT_WISE, info
        t0 = time.perf_counter()
        forced_answer = forced_decode_fn(prompt or "", completion)
        info["forced_decode_s"] = time.perf_counter() - t0
        info["forced_answer"] = forced_answer
        info["forced_correct"] = is_correct(forced_answer, gold)
        return (
            OUTCOME_SILENT_WASTEFUL if info["forced_correct"] else OUTCOME_SILENT_WISE,
            info,
        )
    if is_idk_marker(extracted):
        info["idk_flag"] = True
        if forced_decode_fn is None:
            # Offline / unit-test fallback: treat IDK as wise (no penalty).
            return OUTCOME_WISE_IDK, info
        t0 = time.perf_counter()
        forced_answer = forced_decode_fn(prompt or "", completion)
        info["forced_decode_s"] = time.perf_counter() - t0
        info["forced_answer"] = forced_answer
        info["forced_correct"] = is_correct(forced_answer, gold)
        return (
            OUTCOME_WASTEFUL_IDK if info["forced_correct"] else OUTCOME_WISE_IDK,
            info,
        )
    return (
        OUTCOME_CORRECT if is_correct(extracted, gold) else OUTCOME_WRONG,
        info,
    )


def d1_reward(
    completion: str,
    gold: str,
    alpha: float,
    beta: float,
    mu_wise: float,
    mu_wasteful: float,
    silent_penalty: float,
    forced_decode_fn: Optional[Callable[[str, str], str]] = None,
    prompt: Optional[str] = None,
    extend_silent_forced_decode: bool = False,
    silent_wise_penalty: Optional[float] = None,
    silent_wasteful_penalty: Optional[float] = None,
) -> tuple[float, str, dict]:
    """D1 reward: 5-outcome (or 7-outcome) dispatch.

    Returns (reward, outcome_label, info_dict). The trainer-side wandb callback
    consumes outcome_label + info via the `_LAST_BATCH_TELEMETRY` side channel.

    Reward hierarchy under symmetric mu (5-outcome):
        -silent_penalty < -beta < -mu_wasteful < 0 < +mu_wise < +alpha

    When extend_silent_forced_decode=True the silent channel is replaced by
    two bifurcated outcomes:
        silent_wise:     reward = -silent_wise_penalty
        silent_wasteful: reward = -silent_wasteful_penalty
    The original silent_penalty is unused in this mode. Recommended FD1xsil
    magnitudes (silent_wise_penalty=0.5, silent_wasteful_penalty=3.0) yield:
        -silent_wasteful < -beta < -mu_wasteful < -silent_wise < 0 < +mu_wise < +alpha
    """
    outcome, info = d1_dispatch(
        completion, gold, forced_decode_fn, prompt,
        extend_silent_forced_decode=extend_silent_forced_decode,
    )
    if outcome == OUTCOME_CORRECT:
        r = alpha
    elif outcome == OUTCOME_WRONG:
        r = -beta
    elif outcome == OUTCOME_WISE_IDK:
        r = mu_wise
    elif outcome == OUTCOME_WASTEFUL_IDK:
        r = -mu_wasteful
    elif outcome == OUTCOME_SILENT:
        r = -silent_penalty
    elif outcome == OUTCOME_SILENT_WISE:
        if silent_wise_penalty is None:
            raise ValueError("silent_wise_penalty required when extend_silent_forced_decode=True")
        r = -silent_wise_penalty
    elif outcome == OUTCOME_SILENT_WASTEFUL:
        if silent_wasteful_penalty is None:
            raise ValueError("silent_wasteful_penalty required when extend_silent_forced_decode=True")
        r = -silent_wasteful_penalty
    else:  # pragma: no cover - exhaustive over OUTCOME_CLASSES
        raise ValueError(f"Unknown outcome '{outcome}'")
    return r, outcome, info


def _reset_telemetry(n_samples: int) -> None:
    _LAST_BATCH_TELEMETRY.clear()
    _LAST_BATCH_TELEMETRY["outcomes"] = []
    _LAST_BATCH_TELEMETRY["channel_rewards"] = []
    _LAST_BATCH_TELEMETRY["idk_flags"] = []
    _LAST_BATCH_TELEMETRY["forced_decode_s"] = []
    _LAST_BATCH_TELEMETRY["forced_correct"] = []
    _LAST_BATCH_TELEMETRY["extracted"] = []
    _LAST_BATCH_TELEMETRY["n_samples"] = n_samples


def _record_sample(
    outcome: str,
    reward: float,
    *,
    idk_flag: bool = False,
    forced_decode_s: float = 0.0,
    forced_correct: Optional[bool] = None,
    extracted: str = "",
) -> None:
    _LAST_BATCH_TELEMETRY["outcomes"].append(outcome)
    _LAST_BATCH_TELEMETRY["channel_rewards"].append(reward)
    _LAST_BATCH_TELEMETRY["idk_flags"].append(idk_flag)
    _LAST_BATCH_TELEMETRY["forced_decode_s"].append(forced_decode_s)
    _LAST_BATCH_TELEMETRY["forced_correct"].append(forced_correct)
    _LAST_BATCH_TELEMETRY["extracted"].append(extracted)


def _make_scalar_reward_fn(
    condition_id: str,
    alpha: float,
    beta: float,
    abstain: float,
    classify: Callable[[str, Any], str],
    gold_adapter: Callable[[Any], Any],
    telemetry_extract: Callable[[str], str],
) -> Callable:
    """Factory for the asym and vc scalar-reward closures.

    Both families share the same per-batch loop skeleton:
        require kwargs["answer"]
        _reset_telemetry(len)
        loop completions -> classify -> map outcome to reward -> _record_sample -> append

    The three differing pieces are injected as callables:

        classify(completion, adapted_gold) -> outcome str
            For asym: _classify_asym(c, g)
            For vc:   classify_vc(c, gold_rating_int)

        gold_adapter(raw_gold_str) -> adapted_gold
            For asym: identity (returns raw_gold_str unchanged)
            For vc:   maps "YES"->"1" / "NO"->"-1" as int, raises ValueError on anything else

        telemetry_extract(completion) -> str
            For asym: extract_boxed_answer(c)
            For vc:   normalized vote string "YES" / "NO" / ""

    Preserves:
        - missing-"answer" RuntimeError
        - exact reward values (+alpha / -beta / +abstain)
        - recorded extracted strings (so SC group_rescale LOO is unchanged)
        - __name__ = f"reward_{condition_id}"
    """
    a = alpha
    b = beta
    s = abstain

    def _fn(prompts, completions, **kwargs):
        golds = kwargs.get("answer")
        if golds is None:
            raise RuntimeError(
                f"reward_func for {condition_id} missing 'answer' kwarg; "
                "check the dataset has an 'answer' column"
            )
        _reset_telemetry(len(completions))
        rewards: list[float] = []
        for c, g_raw in zip(completions, golds):
            adapted_gold = gold_adapter(g_raw)
            outcome = classify(c, adapted_gold)
            extracted = telemetry_extract(c)
            if outcome == OUTCOME_CORRECT:
                r = a
            elif outcome == OUTCOME_WRONG:
                r = -b
            else:  # silent
                r = s
            _record_sample(outcome, r, extracted=extracted)
            rewards.append(r)
        return rewards

    _fn.__name__ = f"reward_{condition_id}"
    return _fn


def make_reward_func(
    condition_id: str,
    forced_decode_fn: Optional[Callable[[str, str], str]] = None,
) -> Callable:
    """Return a TRL-compatible reward callable for the given condition.

    TRL GRPOTrainer calls reward_funcs as `fn(prompts, completions, **kwargs)`
    where extra dataset columns are forwarded through kwargs. We expect
    `answer` (the gold) to be present.

    Dispatches on `CONDITIONS[condition_id]["family"]`:
        - "asym" (default): the existing 7-cond + Z3 + CW path.
        - "d1": explicit IDK + forced-decode counterfactual + silent channel.

    The optional `forced_decode_fn(prompt, completion) -> forced_answer_text`
    is invoked once per IDK sample; pass it in at trainer init wired to the
    in-process vLLM engine. Pass None for offline reward unit tests; in that
    mode every IDK sample is classified as wise_idk.

    Telemetry for every sample lands in `_LAST_BATCH_TELEMETRY` (cleared on
    each invocation). Trainer-side wandb callback reads it after the reward
    return.
    """
    if condition_id not in CONDITIONS:
        raise KeyError(f"Unknown condition '{condition_id}'. Known: {list(CONDITIONS)}")
    cfg = CONDITIONS[condition_id]
    family = cfg.get("family", "asym")

    if family == "asym":
        return _make_scalar_reward_fn(
            condition_id=condition_id,
            alpha=float(cfg["alpha"]),
            beta=float(cfg["beta"]),
            abstain=float(cfg["abstain"]),
            classify=_classify_asym,
            gold_adapter=lambda g: g,
            telemetry_extract=extract_boxed_answer,
        )

    if family == "d1":
        a = float(cfg["alpha"])
        b = float(cfg["beta"])
        mu_wise = float(cfg["mu_wise"])
        mu_wasteful = float(cfg["mu_wasteful"])
        silent_pen = float(cfg["silent_penalty"])
        extend_silent = bool(cfg.get("extend_silent_forced_decode", False))
        silent_wise_pen = cfg.get("silent_wise_penalty")
        silent_wasteful_pen = cfg.get("silent_wasteful_penalty")
        if extend_silent and (silent_wise_pen is None or silent_wasteful_pen is None):
            raise ValueError(
                f"condition {condition_id}: extend_silent_forced_decode=True requires "
                "silent_wise_penalty and silent_wasteful_penalty in CONDITIONS entry"
            )
        silent_wise_pen_f = float(silent_wise_pen) if silent_wise_pen is not None else None
        silent_wasteful_pen_f = float(silent_wasteful_pen) if silent_wasteful_pen is not None else None

        def _fn_d1(prompts, completions, **kwargs):
            golds = kwargs.get("answer")
            if golds is None:
                raise RuntimeError(
                    "reward_func missing 'answer' kwarg; check dataset has 'answer'"
                )
            _reset_telemetry(len(completions))
            rewards: list[float] = []
            for p, c, g in zip(prompts, completions, golds):
                r, outcome, info = d1_reward(
                    c, g, a, b, mu_wise, mu_wasteful, silent_pen,
                    forced_decode_fn=forced_decode_fn, prompt=p,
                    extend_silent_forced_decode=extend_silent,
                    silent_wise_penalty=silent_wise_pen_f,
                    silent_wasteful_penalty=silent_wasteful_pen_f,
                )
                _record_sample(
                    outcome,
                    r,
                    idk_flag=info["idk_flag"],
                    forced_decode_s=info["forced_decode_s"],
                    forced_correct=info["forced_correct"],
                    extracted=extract_boxed_answer(c),
                )
                rewards.append(r)
            return rewards

        _fn_d1.__name__ = f"reward_{condition_id}"
        return _fn_d1

    if family == "vc":
        # Deferred imports (avoid a module-level judge<->reward cycle).
        # Both callables are imported once at closure-build time so the
        # per-sample loop stays allocation-free.
        from asym_rewards.judge import classify_vc as _classify_vc_fn
        from asym_rewards.judge import judge_label as _judge_label_fn

        def _vc_gold_adapter(g_str: Any) -> int:
            """Map gold string "YES"/"NO" to numeric rating +1/-1.

            Raises ValueError on anything else, matching the original behaviour
            and surfacing misconfigured build_judge_dataset output at step 0.
            """
            g_str_norm = str(g_str).strip().upper()
            if g_str_norm == "YES":
                return 1
            if g_str_norm == "NO":
                return -1
            raise ValueError(
                f"vc reward: expected gold 'YES' or 'NO', got {g_str!r}. "
                "Check build_judge_dataset produced correct answer strings."
            )

        def _vc_telemetry_extract(completion: str) -> str:
            """Extract the normalized YES/NO vote for SC group_rescale (LOO).

            judge_label returns +1/0/-1; map back to "YES"/"NO"/"" so
            sc_delta_local in the trainer sees the same extracted-string format
            as the asym family.
            """
            vote = _judge_label_fn(completion)
            return "YES" if vote == 1 else ("NO" if vote == -1 else "")

        return _make_scalar_reward_fn(
            condition_id=condition_id,
            alpha=float(cfg["alpha"]),
            beta=float(cfg["beta"]),
            abstain=float(cfg.get("abstain", 0.0)),
            classify=_classify_vc_fn,
            gold_adapter=_vc_gold_adapter,
            telemetry_extract=_vc_telemetry_extract,
        )

    raise ValueError(f"Unknown reward family '{family}' for condition {condition_id}")
