# Goal 1: single-shot utility under asymmetric cost (RETRACTED)

Date: 2026-06-04. Analysis: `cost_sensitive_utility.py` (no GPU, offline on cached rollouts).

**Status: retracted (2026-06-04, see FINAL_VERDICT.md, Part 1).** The apparent win below relied
on a cost metric constructed for this test, and S0 was compared at its own operating point with
no single-pass confidence signal of its own. Under the standard matched-compute metrics
(risk-coverage, selective accuracy, AUROC) no punishment condition shows an advantage. The
analysis is kept as a record of what was tried; none of its numbers support a claim.

## The question

Every FEAR eval metric so far is symmetric (accuracy, ECE, acc_on_attempted, AURC) and
all point the same negative way (Finding O: the reward sweep is a 1-D abstention dial; on
the risk-coverage frontier "no fear condition beats its non-fear counterpart", results.md
L930). But the asymmetric reward was designed for an ASYMMETRIC deployment cost (a wrong
answer hurts more than an abstention). The fair test is the matching deployment metric:

    U(c) = P(correct) - c * P(wrong)        (abstention scores 0)

evaluated at the operating point each policy actually deploys at. results.md L932 already
noted fear "wins in absolute utility at high cost" but dismissed it because thresholding
S0/P0 confidence dominates. That rebuttal hides a compute cost: the threshold is built from
K=16 mode-vote confidence (16x inference). This analysis isolates the regime where that
rebuttal does not apply: SINGLE-SHOT (K=1) deployment.

## Result 1 (retracted): single-shot FEAR appeared utility-optimal under asymmetric cost

Using the greedy (T=0, K=1) rollouts -- one forward pass per problem, the cost-efficient
deployment regime -- the maximum-punishment policy S4 overtakes the symmetric baseline S0
above a benchmark-specific crossover cost c*:

| Benchmark      | c* (S0 -> best FEAR) | at c=3, S0 U (95% CI)   | at c=3, S4 U (95% CI)   |
|----------------|----------------------|-------------------------|-------------------------|
| TruthfulQA-MC  | **0.92**             | [-0.454, -0.289]        | [-0.318, -0.177]        |
| MMLU-STEM      | 1.70                 | [-0.527, -0.419]        | [-0.453, -0.354]        |
| MATH           | 2.14                 | [-0.514, -0.409]        | [-0.344, -0.275]        |
| GSM8K          | 2.63                 | [-0.201, -0.009]        | [-0.025,  0.003]        |
| AIME-24/25     | ~0 (tiny n=30)       | --                      | --                      |

- On MATH the 95% bootstrap CIs are DISJOINT at c=3: single-shot S4 has significantly higher
  expected utility than single-shot S0 when a wrong answer costs >= ~3x a correct one.
- On **TruthfulQA the crossover is 0.92 < 1**: the punishment-trained model is the better
  single-shot policy even when wrong and right answers are weighted EQUALLY. Because base
  accuracy there is ~12%, attempting is barely +EV, so trained-in caution pays off almost
  immediately. This is the cleanest "fear is better" statement: in honesty-critical,
  low-accuracy domains, single-shot punishment training beats the symmetric reward with no
  test-time machinery.
- The intermediate conditions tile the cost axis (S0 optimal at low c, then S3/S4 as c rises);
  each FEAR level is the EV-optimal single-shot policy in its own cost band.

Figure: `figures/fig1_single_shot_utility.png`, `figures/fig3_crossover_by_benchmark.png`.

## Result 2 (retracted with Result 1): the apparent win was specific to K=1

`figures/fig2_utility_per_sample.png`. The "just threshold confidence" baseline is genuinely
strong, but it REQUIRES K>=2 samples (self-consistency); at K=1 it is undefined and reduces
to the policy's own operating point. Accounting for compute (utility per generated sample),
post-hoc thresholding of S0/P0 stays POSITIVE at c=3 math (U/sample 0.024-0.12 across K=2..16)
while single-shot fear is negative (S4 -0.31). So once you can afford >=2 samples, post-hoc
thresholding dominates even per-sample. FEAR's deployable niche is therefore precisely:
single-sample inference under asymmetric / honesty-critical cost.

Among true single-shot policies, though, the ordering is unambiguous: at c=3 math, S4 (-0.31)
> S0 (-0.46); fear is the only way to reach a high-cost operating point from one sample.

## Open confirming experiment (the one remaining baseline)

A single-shot confidence signal for S0 -- mean per-token sequence logprob / answer-token
entropy at K=1 -- is the one untested competitor. The cached rollouts store no logprobs, so
this needs one forward pass per cached completion (teacher-forced; ~5000 on MATH, ~30 min on
a single small GPU; no re-generation). If S0+logprob-threshold at K=1 still trails S4 at high
cost, the single-shot FEAR win is airtight. Would be a standalone `research/` script (no edits
to the main eval pipeline); not yet written -- awaiting go-ahead.

## One-line takeaway

Retracted. The single-shot utility advantage of punishment training came from the constructed
cost metric and the missing single-pass confidence baseline, not from better abstention.
