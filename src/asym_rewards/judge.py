"""Step-correctness judging: shared helpers for the zero-shot judge harness
(scripts/smoke_judge.py) and the trained Verifier-Calibration (`vc`) reward
family (reward.py).

Lives in its own module (not reward.py / grpo.py, both already past the
300-line smell threshold) so the label parser, vc classifier, and judge-dataset
builder have one split-by-responsibility home and neither big file grows.

PRM800K human ratings are in {-1, 0, +1}. The judge emits a 3-way decision via
a boxed token:

    \\boxed{YES}  -> +1  (step is correct)
    \\boxed{NO}   -> -1  (step is incorrect)
    no/other box ->  0  (abstain)

Circular-import contract
------------------------
judge.py imports from reward.py ONLY inside function bodies (deferred imports),
never at module level. reward.py must do the same for judge.py. Module-level
cross-imports between the two would create a judge -> reward -> judge cycle that
breaks both modules on first import. The outcome-label constants below
(_VC_OUTCOME_*) mirror reward.py's OUTCOME_* values exactly so classify_vc can
use string literals without importing reward.py at module level. If reward.py
renames those constants, update the mirrors here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

# Single source of truth for the boxed parser, dependency-free (no math_verify),
# shared with reward.py. Keeps judge.py importable on a host without math_verify.
from asym_rewards._boxed import extract_boxed_answer

# Mirror of reward.py's OUTCOME_* constants. Kept local to avoid a module-level
# circular import (judge -> reward -> judge). Values MUST match reward.py.
_VC_OUTCOME_CORRECT = "correct"
_VC_OUTCOME_WRONG = "wrong"
_VC_OUTCOME_SILENT = "silent"


def _problem_id(problem_text: str) -> str:
    """Stable content hash of a problem, used as the leak-detection identity.

    Leakage is a PROBLEM appearing in both train and test (PRM800K phase2
    splits are problem-disjoint by construction; this is the safety assertion).
    Keying on a hash of the problem text -- not a positional row index -- means
    the same problem produces the same id in either file, so a genuine overlap
    actually collides and trips the D2 guard, while distinct problems never do.
    """
    return hashlib.sha1(problem_text.strip().encode("utf-8")).hexdigest()[:16]


JUDGE_PROMPT_TEMPLATE = (
    "You are checking a single step of a math solution.\n\n"
    "Problem:\n{problem}\n\n"
    "Proposed reasoning step:\n{step}\n\n"
    "Is this step mathematically correct and useful for solving the problem? "
    "Reply with exactly one of \\boxed{{YES}} or \\boxed{{NO}}.\n"
    "Answer: "
)


def judge_label(text: str) -> int:
    """Parse a judge completion into {-1, 0, +1}: NO / abstain / YES.

    Returns 0 (abstain) for missing or unparseable boxed content.

    Uses extract_boxed_answer (pure stdlib, no math_verify) rather than a
    deferred import from reward.py. This makes judge.py fully importable
    without math_verify on the laptop, enabling the test suite to run locally.
    The circular-import contract (no module-level cross-imports between judge.py
    and reward.py) is preserved; reward.py still uses deferred in-function
    imports when it needs judge.py symbols.
    """
    boxed = extract_boxed_answer(text)
    if not boxed:
        return 0
    norm = boxed.strip().upper().rstrip(".").rstrip(")")
    if norm in ("YES", "Y", "TRUE", "CORRECT"):
        return 1
    if norm in ("NO", "N", "FALSE", "INCORRECT"):
        return -1
    return 0  # unparseable boxed content - treat as abstain


def classify_vc(completion: str, gold_rating: int) -> str:
    """Map a judge completion + gold rating to one of {correct, wrong, silent}.

    This is the per-sample outcome classifier for the vc reward family. The
    judge vote is obtained from judge_label(completion), then compared to the
    gold PRM800K rating (+1 = correct step, -1 = incorrect step).

    Outcome mapping:
        vote == gold_rating           -> "correct"  (judge agrees with human)
        vote != gold_rating AND vote != 0 -> "wrong" (judge disagrees)
        vote == 0 (no parseable box)  -> "silent"   (judge abstained)

    gold_rating must be +1 or -1. Passing 0 is undefined; the function does
    not validate it because the caller (reward.py _fn_vc) converts "YES"/"NO"
    strings from kwargs["answer"] before calling here.

    The outcome constants are defined locally in this module (_VC_OUTCOME_*)
    to avoid a circular import with reward.py. The reward.py vc family branch
    uses a deferred `from asym_rewards.judge import classify_vc` inside the
    closure body, not at module level, for the same reason.
    """
    vote = judge_label(completion)
    if vote == 0:
        return _VC_OUTCOME_SILENT
    if vote == gold_rating:
        return _VC_OUTCOME_CORRECT
    return _VC_OUTCOME_WRONG


def judge_problem_ids(split_path: "str | Path", limit: Optional[int] = None) -> set:
    """Set of problem-identity hashes present in a phase-2 split file.

    Used to build the `forbid_problem_ids` set from the held-out TEST split so
    build_judge_dataset can assert no test problem leaks into training.
    """
    path = Path(split_path)
    ids: set = set()
    with path.open("r", encoding="utf-8") as fh:
        for row_idx, line in enumerate(fh):
            if limit is not None and row_idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            problem = row.get("question", {}).get("problem")
            if problem:
                ids.add(_problem_id(problem))
    return ids


def build_judge_dataset(
    split_path: "str | Path",
    max_pos_per_row: int = 1,
    max_neg_per_row: int = 1,
    limit: Optional[int] = None,
    forbid_problem_ids: Optional[set] = None,
) -> list[dict]:
    """Build a flat judge training dataset from a phase-2 JSONL rollout file.

    Reads a JSONL file where each row has the schema used by smoke_judge.py:
        row["question"]["problem"]       : str  problem text
        row["label"]["steps"]            : list of step dicts
            step["completions"]          : list of completion dicts
                completion["rating"]     : int  +1 or -1 (PRM800K human label)
                completion["text"]       : str  step text

    For each row, samples up to max_pos_per_row positive (rating==+1) and
    max_neg_per_row negative (rating==-1) completions from each step to build
    balanced training examples. Each example becomes a dict:
        {
          "prompt": JUDGE_PROMPT_TEMPLATE.format(problem=..., step=...),
          "answer": "YES" if rating==+1 else "NO",
          "problem_id": "<sha1(problem)[:16]>",   # leak-detection identity
          "id": "<problem_id>_<step_index>_<completion_index>",  # row-unique
        }

    D2 HARD GUARD - train/test leak prevention (keyed on PROBLEM identity):
        If forbid_problem_ids is not None (build it from the TEST split via
        judge_problem_ids), asserts that NO produced example shares a problem_id
        with the test split. Keying on a content hash of the problem -- not a
        positional row index -- means the guard actually fires on a real
        train/test overlap and never false-positives on distinct problems that
        merely share a row position across the two files.

    Args:
        split_path: path to the phase-2 (TRAIN) JSONL rollout file.
        max_pos_per_row: max positive-rated completions to sample per step.
        max_neg_per_row: max negative-rated completions to sample per step.
        limit: if not None, stop after this many rows (for unit tests / smoke).
        forbid_problem_ids: set of problem_id hashes (from the test split) that
            must NOT appear in the output.

    Returns:
        List of dicts with keys {prompt, answer, problem_id, id}.

    Raises:
        AssertionError: if any produced problem_id is in forbid_problem_ids.
        FileNotFoundError: if split_path does not exist.
    """
    path = Path(split_path)
    examples: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for row_idx, line in enumerate(fh):
            if limit is not None and row_idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            problem = row["question"]["problem"]
            pid = _problem_id(problem)
            steps = row["label"]["steps"]
            for step_idx, step in enumerate(steps):
                completions = step.get("completions", [])
                pos_seen = 0
                neg_seen = 0
                for comp_idx, comp in enumerate(completions):
                    rating = comp.get("rating")
                    if rating not in (1, -1):
                        continue
                    if rating == 1:
                        if pos_seen >= max_pos_per_row:
                            continue
                        pos_seen += 1
                    else:
                        if neg_seen >= max_neg_per_row:
                            continue
                        neg_seen += 1
                    step_text = comp.get("text", "")
                    examples.append({
                        "prompt": JUDGE_PROMPT_TEMPLATE.format(
                            problem=problem,
                            step=step_text,
                        ),
                        "answer": "YES" if rating == 1 else "NO",
                        "problem_id": pid,
                        "id": f"{pid}_{step_idx}_{comp_idx}",
                    })

    if forbid_problem_ids is not None:
        leaked = sorted({ex["problem_id"] for ex in examples
                         if ex["problem_id"] in forbid_problem_ids})
        # Explicit raise (NOT assert): a `python -O` run strips asserts, which
        # would silently disable this hard train/test leak guard.
        if leaked:
            raise RuntimeError(
                f"D2 GUARD VIOLATION: {len(leaked)} test-split problem(s) leaked into "
                f"the training dataset produced from {path}. Leaking problem_ids: "
                f"{leaked!r}. The train and test phase-2 splits must be problem-disjoint."
            )

    return examples


# ---------------------------------------------------------------------------
# Vote aggregation + matched-coverage helpers
# ---------------------------------------------------------------------------

def aggregate_votes(votes: list[int]) -> dict:
    """Aggregate a list of per-sample judge votes in {-1, 0, +1} into a
    committed prediction and confidence.

    Returns a dict with keys: pred, votes_yes, votes_no, votes_abstain,
    conf_yes.

    pred is +1 when yes_votes > no_votes, -1 when no_votes > yes_votes, 0
    otherwise (tie or all abstain).

    Originally defined in scripts/smoke_judge.py; moved here so all judge
    primitives live in one canonical place. smoke_judge re-imports from here.
    """
    k = len(votes)
    if k == 0:
        return {"pred": 0, "votes_yes": 0, "votes_no": 0, "votes_abstain": 0,
                "conf_yes": 0.0}
    votes_yes = votes.count(1)
    votes_no = votes.count(-1)
    votes_abstain = votes.count(0)
    conf_yes = votes_yes / k
    if votes_yes > votes_no:
        pred = 1
    elif votes_no > votes_yes:
        pred = -1
    else:
        pred = 0
    return {
        "pred": pred,
        "votes_yes": votes_yes,
        "votes_no": votes_no,
        "votes_abstain": votes_abstain,
        "conf_yes": conf_yes,
    }


def matched_subset(per_condition_committed: "dict[str, set[int]]",
                   attempt_floor: float = 0.2,
                   n_total: int = 0) -> dict:
    """Compute the intersection of example indices that every non-degenerate
    condition committed on.

    Args:
        per_condition_committed: mapping condition -> set of example indices
            where pred != 0.
        attempt_floor: conditions with coverage (committed / n_total) below
            this are excluded as degenerate and listed separately.
        n_total: total number of judged steps (used to compute coverage for
            the degeneracy check); if 0, all conditions are treated as
            non-degenerate.

    Returns a dict with keys:
        matched_indices: sorted list of indices in the intersection
        n_matched: len of matched_indices
        conditions_included: list of condition codes included in intersection
        conditions_excluded: list of condition codes excluded as degenerate

    Originally defined in scripts/smoke_judge.py; moved here so all judge
    primitives live in one canonical place. smoke_judge re-imports from here.
    """
    included = []
    excluded = []
    for cond, committed in per_condition_committed.items():
        if n_total > 0:
            coverage = len(committed) / n_total
            if coverage < attempt_floor:
                excluded.append(cond)
                continue
        included.append(cond)

    if not included:
        return {
            "matched_indices": [],
            "n_matched": 0,
            "conditions_included": [],
            "conditions_excluded": excluded,
        }

    # Start from the first condition's committed set, intersect the rest.
    intersection: set[int] = set(per_condition_committed[included[0]])
    for cond in included[1:]:
        intersection &= per_condition_committed[cond]

    matched = sorted(intersection)
    return {
        "matched_indices": matched,
        "n_matched": len(matched),
        "conditions_included": sorted(included),
        "conditions_excluded": sorted(excluded),
    }


# ---------------------------------------------------------------------------
# Experiment-3 scoring helpers (numpy-only; metrics.auroc + bootstrap_ci)
# ---------------------------------------------------------------------------

def score_condition(rows: "list[dict]") -> dict:
    """Score a single condition's per-prompt rows for AUROC over committed steps.

    Each row must have at minimum:
        gold      : int  gold rating (+1 or -1)
        pred      : int  committed prediction (+1, -1, or 0)
        conf_yes  : float  fraction of YES votes (0 <= conf_yes <= 1)

    Committed steps are those where pred != 0. Positive class = INCORRECT step
    (gold == -1, matching Exp-3 orientation in smoke_judge._run_condition).
    Score = 1 - conf_yes (higher score = judge more uncertain about YES = more
    likely to flag an incorrect step as incorrect).

    Uses asym_rewards.metrics.auroc + bootstrap_ci (numpy-only, no sklearn).

    Returns a dict:
        coverage         : float  committed / n
        auroc_incorrect  : float  AUROC over committed rows (nan if degenerate)
        auroc_ci_lo      : float
        auroc_ci_hi      : float
        f1_incorrect     : float  F1 score on committed rows (positive = gold==-1,
                                  prediction-positive = pred==-1); 0.0 when
                                  denominators are zero (degenerate)
        n_pos            : int  incorrect (gold==-1) committed steps
        n_neg            : int  correct   (gold==+1) committed steps
        n_attempted      : int  total committed steps
        committed_indices: list[int]  sorted indices of committed rows
    """
    import numpy as np
    from asym_rewards.metrics import auroc as _auroc, bootstrap_ci as _bootstrap_ci

    n = len(rows)
    if n == 0:
        return {
            "coverage": 0.0,
            "auroc_incorrect": float("nan"),
            "auroc_ci_lo": float("nan"),
            "auroc_ci_hi": float("nan"),
            "f1_incorrect": 0.0,
            "n_pos": 0,
            "n_neg": 0,
            "n_attempted": 0,
            "committed_indices": [],
        }

    committed_indices = [i for i, r in enumerate(rows) if r["pred"] != 0]
    n_attempted = len(committed_indices)
    coverage = n_attempted / n

    if n_attempted > 0:
        committed_rows = [rows[i] for i in committed_indices]
        scores_arr = np.array([1.0 - r["conf_yes"] for r in committed_rows])
        labels_arr = np.array([int(r["gold"] == -1) for r in committed_rows])
        pred_arr = np.array([int(r["pred"]) for r in committed_rows])
        auroc_val = _auroc(scores_arr, labels_arr)
        if not (auroc_val != auroc_val):  # not NaN
            ci_lo, ci_hi = _bootstrap_ci(_auroc, scores_arr, labels_arr)
        else:
            ci_lo = ci_hi = float("nan")
        n_pos = int((labels_arr == 1).sum())
        n_neg = int((labels_arr == 0).sum())
        # F1 on committed rows: positive class = incorrect (gold==-1),
        # prediction-positive = pred==-1.
        tp = int(((pred_arr == -1) & (labels_arr == 1)).sum())
        fp = int(((pred_arr == -1) & (labels_arr == 0)).sum())
        fn = int(((pred_arr != -1) & (labels_arr == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_val = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
    else:
        auroc_val = float("nan")
        ci_lo = ci_hi = float("nan")
        n_pos = n_neg = 0
        f1_val = 0.0

    return {
        "coverage": coverage,
        "auroc_incorrect": auroc_val,
        "auroc_ci_lo": ci_lo,
        "auroc_ci_hi": ci_hi,
        "f1_incorrect": f1_val,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_attempted": n_attempted,
        "committed_indices": committed_indices,
    }


def matched_coverage_report(
    per_condition_committed: "dict[str, set[int]]",
    per_condition_rows: "dict[str, list[dict]]",
    attempt_floor: float,
    tiny_n: int = 50,
) -> dict:
    """Compute matched-coverage AUROC across conditions (N3 logic).

    Finds the intersection of committed-step indices across all non-degenerate
    conditions (via matched_subset), then re-scores each condition's rows
    restricted to that intersection. Falls back to a warning when the matched
    set is below tiny_n.

    Args:
        per_condition_committed: mapping condition -> set of committed indices.
        per_condition_rows: mapping condition -> full rows list (same alignment
            as per_condition_committed indices).
        attempt_floor: floor passed to matched_subset for degeneracy exclusion.
        tiny_n: minimum matched-set size to report matched AUROC; below this
            the function sets fallback=True and skips per-condition reporting.

    Returns a dict with keys:
        n_matched          : int
        conditions_included: list[str]
        conditions_excluded: list[str]
        fallback           : bool  True when matched set is below tiny_n
        warning            : str or None
        per_condition      : dict  mapping condition -> {auroc_incorrect,
                             auroc_ci_lo, auroc_ci_hi, n}; empty when fallback
    """
    import numpy as np
    from asym_rewards.metrics import auroc as _auroc, bootstrap_ci as _bootstrap_ci

    n_total = max((max(s) + 1 for s in per_condition_committed.values() if s), default=0)
    # Use the length of the first rows list as n_total when available (more
    # accurate than max-index-based estimate).
    for rows in per_condition_rows.values():
        if rows:
            n_total = len(rows)
            break

    mc_info = matched_subset(per_condition_committed, attempt_floor=attempt_floor,
                             n_total=n_total)
    n_matched = mc_info["n_matched"]
    matched_indices_set = set(mc_info["matched_indices"])

    result: dict = {
        "n_matched": n_matched,
        "conditions_included": mc_info["conditions_included"],
        "conditions_excluded": mc_info["conditions_excluded"],
        "fallback": False,
        "warning": None,
        "per_condition": {},
    }

    if n_matched < tiny_n:
        msg = (
            f"[warn] matched-coverage subset has only {n_matched} steps "
            f"(threshold {tiny_n}); falling back to per-condition "
            f"AUROC only - matched AUROC not reported."
        )
        result["fallback"] = True
        result["warning"] = msg
        return result

    for cond in mc_info["conditions_included"]:
        cond_rows = per_condition_rows.get(cond, [])
        mc_rows = [cond_rows[i] for i in sorted(matched_indices_set)
                   if i < len(cond_rows)]
        if not mc_rows:
            continue
        mc_scores = np.array([1.0 - r["conf_yes"] for r in mc_rows])
        mc_labels = np.array([int(r["gold"] == -1) for r in mc_rows])
        mc_auroc_val = _auroc(mc_scores, mc_labels)
        if not (mc_auroc_val != mc_auroc_val):  # not NaN
            mc_lo, mc_hi = _bootstrap_ci(_auroc, mc_scores, mc_labels)
        else:
            mc_lo = mc_hi = float("nan")
        result["per_condition"][cond] = {
            "auroc_incorrect": mc_auroc_val,
            "auroc_ci_lo": mc_lo,
            "auroc_ci_hi": mc_hi,
            "n": len(mc_rows),
        }

    return result
