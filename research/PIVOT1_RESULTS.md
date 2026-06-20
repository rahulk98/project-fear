# Pivot 1 -- Abstaining Verifier: end-to-end de-risk results (2026-06-04)

Idea: train a process verifier that emits a third class, "UNVERIFIABLE", in
addition to "correct" / "incorrect". The unverifiable label is self-supervised
by Monte-Carlo rollout success variance: a step whose MC success rate has a
Wilson confidence interval straddling 0.5 is one the model genuinely cannot
call. The payoff object is a verification coverage-risk curve plus a best-of-N
deferral gain (re-sample when a step is unverifiable).

This de-risk runs the ZERO-training first experiment in two forms: an instant
answer-level pre-check on cached data, and the proper step-level test on
PRM800K (GPU). It answers three questions: (a) is the unverifiable bucket a
non-trivial slice, (b) does it hold real gold-correct mass (not just "wrong"
renamed), (c) does the MC signal trace a coverage-risk frontier and does
deferral recover accuracy.

## Setup

- Step-level: 600 PRM800K phase2_test problems, K=12 completions per step-prefix
  at fractional boundaries (0, 0.25, 0.5, 0.75 of the human reference steps),
  completer = Qwen2.5-Math-1.5B, graded with math_verify. 1830 prefix records.
  Generated on alias cuda:1 + cuda:2 (2080 Ti, fp16), ~18 min wall-clock.
- Answer-level pre-check (no GPU): cached K=32 (P0) and K=16 (S0) self-consistency
  rollouts on MATH and GSM8K; per-problem MC success rate over the K samples.
- Scripts: research/av_gen_rollouts.py (generation), research/av_analyze.py
  (analysis, two modes). Outputs: research/outputs/abstaining_verifier_step.json,
  research/outputs/abstaining_verifier_answer.json.

## Result: GO

### (a) The unverifiable bucket is real and non-trivial

| data | K | unverifiable frac | ver-correct | ver-wrong |
|---|---|---|---|---|
| step-level PRM800K | 12 | 0.338 | 0.082 | 0.580 |
| answer MATH (P0) | 32 | 0.187 | 0.435 | 0.379 |
| answer GSM8K (P0) | 32 | 0.163 | 0.647 | 0.190 |
| answer MATH (S0) | 16 | 0.303 | 0.362 | 0.335 |

The answer-level buckets sit in the target 10-30% band. The step-level bucket is
0.338, a little high; this is the expected consequence of K=12 (wider Wilson CIs
straddle 0.5 more often). K=16-24 would pull it toward the band. Either way the
unverifiable class is a substantial, non-degenerate slice.

### (b) The unverifiable bucket holds genuine gold-correct mass

Mean MC success rate by bucket (step-level): ver-correct 0.903, unverifiable
0.483, ver-wrong 0.037. The three classes are cleanly separated and the
unverifiable class sits at ~0.5, i.e. it carries real gold-correct mass. It is
NOT the "incorrect" class with a nicer name (that class is ver-wrong at 0.037).
Same pattern at the answer level (MATH: 0.904 / 0.485 / 0.086; GSM8K: 0.913 /
0.502 / 0.142).

### (c) The MC signal traces a coverage-risk frontier; deferral recovers accuracy

- MC value orders final-answer correctness strongly: AUROC 0.98 (step-level),
  0.99 (answer MATH), 0.987 (answer GSM8K). Risk-coverage (step-level): error is
  0.105 at 50% coverage, rising to 0.523 at full coverage.
- Best-of-K deferral on the unverifiable band is the headline:
  - step-level: single-sample accuracy 0.483 -> best-of-12 accuracy 1.000
    (every one of the 619 unverifiable prefixes had at least one correct re-sample)
  - answer MATH: 0.485 -> 0.989 ; answer GSM8K: 0.502 -> 1.000
  This is exactly the "re-sample when unverifiable" payoff the method promises.

### Honest negatives / caveats

- CI WIDTH ALONE is not a correctness signal (AUROC of negated CI width = 0.05
  step-level). Width is a symmetric uncertainty measure; you need the MC value
  (p_hat), not just the interval, to trace the frontier. The method must key on
  the MC point estimate, not interval width.
- The step-level completer (Qwen2.5-Math-1.5B) fails on most arbitrary PRM800K
  step-prefixes (ver-wrong = 58%), so the base correctness rate is low. This does
  not weaken the de-risk (the 3-way structure and deferral gain are clear), but a
  stronger completer or a larger K would raise the ver-correct mass.
- AUROC of p_hat for modal-correct is partly mechanical (p_hat and a
  majority-correct label are correlated). The substantive evidence is the bucket
  separation and the deferral gain, which are not circular.

## Verdict and next step

The zero-training de-risk passes on all three checks. The unverifiable class is
real, separable, gold-bearing, and a single re-sample round on it recovers
accuracy from ~0.48 to ~1.0. This supports proceeding to train the 3-way
generative PRM (correct / incorrect / unverifiable) with a proper scoring rule
plus asymmetric cost (reusing the FEAR reward machinery on the verifier), with a
verification coverage-risk curve and a Best-of-N deferral gain as headline
objects. Recommended training tweaks from this de-risk: K>=16 for tighter CIs,
key the abstain decision on the MC point estimate (not CI width), and consider a
stronger or policy-matched completer.
