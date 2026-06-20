# Three pivots, built and tested end-to-end (2026-06-04)

All three project-pivot finalists were implemented and run end-to-end in research/
(isolated from the live pipeline), each with an adversarial verification pass.
This is the consolidated verdict. Per-pivot detail: PIVOT1_RESULTS.md,
SOCIAL_CHOICE_THEORY.md + outputs/social_choice_voting.json, PIVOT3_RESULTS.md.

## Scorecard

| Pivot | What was tested | Verdict | Ceiling | Risk now |
|---|---|---|---|---|
| 1. Abstaining Verifier | zero-training MC de-risk: step-level (PRM800K, GPU) + answer-level (cached) | **GO** | high (EMNLP/ACL main) | low |
| 2. Social-Choice of self-consistency | clone-robust voting (empirical) + impossibility theorem | **NO (as-is)** | high if it worked | realized: theorem broken, empirics marginal |
| 3. Compute-aware selective prediction | reproduce K*(c) crossover + cross-domain transfer | **REPRODUCED (safe)** | modest (measurement paper) | low |

## Pivot 1 -- Abstaining Verifier: GO

Train a 3-way process verifier {correct, incorrect, UNVERIFIABLE}, where the
unverifiable class is self-supervised by Monte-Carlo rollout variance (Wilson CI
of success rate straddles 0.5). The zero-training de-risk passed all three checks:

- Unverifiable bucket is a real, non-degenerate slice: 16-19% (answer-level, K=32),
  34% (step-level, K=12; the higher number is the K=12 wide-CI effect).
- It holds genuine gold-correct mass: mean MC success 0.90 / 0.48 / 0.04 across
  ver-correct / unverifiable / ver-wrong. The unverifiable class is NOT "wrong"
  renamed.
- MC value traces a strong coverage-risk frontier (AUROC 0.98-0.99) and, the
  headline, a best-of-K re-sample on the unverifiable band lifts accuracy from
  ~0.48 to 0.99-1.00. Every uncertain step was recoverable by re-sampling.
- Honest caveat: CI WIDTH alone is not a correctness signal; the method must key
  on the MC point estimate.

Why it wins: highest ceiling of the three, now empirically de-risked, reuses the
student's FEAR reward code + PRM800K + judges, and has a clean training next step
(3-way generative PRM with proper scoring + asymmetric cost; K>=16). It is the
natural evolution of the abstention work from the solver to the verifier.

## Pivot 2 -- Social-Choice of self-consistency: NO (as-is)

Two independent failures, both surfaced by adversarial verification:

- Empirical (clone-robust voting): clone-merged plurality uses real math_verify
  equivalence clustering, but it does NOT reliably beat plain plurality. On MATH
  the accuracy gain is 0.2-0.34pp at small K (10-17 problems out of 5000) and
  decays to ~0 or negative at large K; on GSM8K at K>=16 it is at parity or worse
  (-0.22pp at K=32). The calibration (AUROC) gain is more consistent (25/28 rows,
  mean +0.38pp) but tiny and not significance-tested. The clone-attack stress
  test was invalid as implemented (injecting distinct strings does not split an
  existing vote cluster). Natural clone mass exists (11-23% of MATH problems) but
  merging it barely helps.
- Theory (impossibility theorem): the headline Theorem 3.1 is FALSE as stated. A
  constructive counterexample satisfies all five axioms; the proof silently
  assumes a sixth, unstated condition (clone-invariant confidence A5'). The real,
  narrower result is that frequency-faithful confidence and clone-invariant
  confidence are jointly unsatisfiable when a split clone set wins -- a genuine
  but much smaller claim that needs A5' added as an explicit axiom. The
  normalization escape (Theorem 3.3) is correct.

Net: the practical payoff is marginal and the headline theorem does not hold.
Salvageable only as a narrower note (the A5-vs-A5' incompatibility plus the
normalization result), which is well below the original "Arrow-for-LLMs" ambition
and not a strong standalone paper. This matches its pre-build risk rating
(highest risk, biggest swing) -- the risk materialized.

## Pivot 3 -- Compute-aware selective prediction: REPRODUCED (safe)

The full pipeline reproduces exactly (max delta 0.0006 across ~40 values). The
compute-crossover law holds: a single probe forward-pass equals self-consistency
at K* ~ 3.1 on MATH (non-overlapping CIs, well-constrained). On GSM8K the probe
is statistically equivalent to K=4 (CIs overlap), so the honest claim is K* in
[3, 4] rather than a precise 3.68. Cross-domain transfer holds (math->gsm8k probe
AUROC 0.744 ~ in-domain 0.751; TruthfulQA in-domain 0.99). This is a real,
reproducible, paper-ready empirical contribution, but lower ceiling: it is a
measurement / practitioner-guide result, not a new method or theorem.

## Recommendation

Pick **Pivot 1 (Abstaining Verifier)**. It is the only finalist that is both
high-ceiling and now de-risked with a striking, clean result (the 0.48 -> 1.0
deferral gain), and it builds directly on assets you already have. **Pivot 3** is
the safe fallback if you want the fastest possible path to a submitted paper.
**Pivot 2** should be dropped as a headline (it can survive only as a short note
inside another paper, if at all).

Next step on a Pivot-1 GO: train the 3-way PRM (reward.py change -> grpo-reward-
reviewer + codex gate), with K>=16 MC labeling and a verification coverage-risk
curve + Best-of-N deferral gain as the headline figures.
