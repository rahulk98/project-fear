# Results

Curated headline numbers for Project FEAR. These are drawn from the project's working
results log and from an adversarially-refuted final verdict in which four follow-up
experiments were written, run on the cached rollouts, each refuted, and synthesized, with
all numbers recomputed from raw files and no constructed metrics. The end-to-end narrative
with full methodology is in [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md).

All training used Qwen 2.5 base models with GRPO and LoRA. All seven single-seed conditions
were trained at 1.5B (1500 GRPO steps each) and evaluated with K=16 and K=32 vLLM rollouts
across six benchmarks. Frontier conditions were scaled to 3B.

---

## 1. The sweep is a one-dimensional abstention dial (Finding O)

Single-shot (greedy, K=1) frontier on MATH. As the implied threshold rises from S0 to S4,
the attempt rate falls monotonically while accuracy-on-attempted stays essentially flat.
Punishment does not preferentially remove wrong answers, it just makes the model answer less
often.

| Condition | MATH attempt rate | MATH acc-on-attempted |
|---|---|---|
| S0 | 0.867 | 0.618 |
| S1 | 0.857 | 0.617 |
| S2 | 0.849 | 0.612 |
| S3 | 0.836 | 0.602 |
| S4 | 0.325 | 0.513 |

On GSM8K the same pattern holds (S0 attempt 0.951 acc 0.722; S3 0.717 / 0.699; S4 collapses
to 0.017 / 0.591). S4, the pure-penalty degenerate corner, abstains on almost everything.

---

## 2. No fear condition beats its non-fear counterpart on the frontier

Matched-compute risk-coverage frontier (AURC, lower is a better confidence ranking),
reported as MATH and then GSM8K.

| Condition | MATH AURC | GSM8K AURC |
|---|---|---|
| P0 (positive-only) | 0.107 | 0.041 |
| CW (confidence-weighted) | 0.108 | 0.046 |
| D1low | 0.109 | 0.046 |
| C1 | 0.111 | 0.051 |
| S0 | 0.112 | 0.050 |
| S1 | 0.114 | 0.053 |
| C2 | 0.116 | 0.060 |
| S2 | 0.116 | 0.063 |
| base | 0.120 | 0.066 |
| S3 | 0.122 | 0.064 |
| S4 | 0.254 | 0.286 |

The ordering is monotonic in inverse-fear. The non-fear escapes (P0, CW, D1low) have the
best frontier, the symmetric and mild-fear conditions sit in one tight band (the
operating-point dial), and fear degrades back to base level and then collapses at S4. C2 and
S2 land on top of each other at 0.116 on MATH, a fifth orthogonal confirmation of the
explicit-versus-implicit equivalence.

---

## 3. The 3B check

K=32 calibration eval on Qwen 2.5 3B adapters (one seed each). S0 and P0 differ only in
where abstention sits. Accuracy is over all problems.

| Condition (3B) | MATH acc | MATH abstention | GSM8K abstention | MMLU-STEM acc |
|---|---|---|---|---|
| P0 | 0.419 | 0.032 | 0.010 | 0.184 |
| S0 | 0.422 | 0.069 | 0.067 | 0.185 |

S0 abstains about twice as often as P0 on MATH and about seven times as often on GSM8K, at
the same MATH and MMLU-STEM accuracy (MMLU-STEM abstention is the same, 0.653 versus 0.654).
On GSM8K S0 accuracy is lower (0.689 versus 0.729). An FD1 row (0.418 / 0.183) in earlier
versions of this file is removed: its 3B evaluation files could not be located.

---

## 4. The normalization is what stabilizes the asymmetry

Removing GRPO's within-group z-score normalization (the `zhu_signed` advantage) lets the raw
reward magnitude reach behavior again. The high-penalty S2 then abstains about three times as
often as S0.

| Benchmark | S0 abstention | S2 abstention |
|---|---|---|
| MATH | 15% | 46% |
| GSM8K | 19% | 80% |
| MMLU-STEM | 48% | 93% |

1.5B, one seed, K=32. This reframes Finding O: z-score
normalization is not merely what hides the asymmetry, it is what keeps high-penalty training
stable. Unnormalized, the magnitude reaches behavior, but toward over-abstention and
degradation. Partial normalization (a global-mean baseline) sits monotonically between the
two, confirming the mechanism end to end.

---

## 5. Hidden-state probes

**Corrected (2026-10-18).** An earlier version of this section described a cross-condition
"correctness" probe transfer (mean off-diagonal AUROC 0.989 MATH, 0.992 GSM8K). That matrix
comes from an abstained-versus-attempted probe (scripts/probe_analysis.py,
cross_condition_transfer), not a correctness probe. Abstained completions lack a boxed answer,
which is visible at the last token, so near-perfect separation is expected and says nothing
about a shared correctness direction. The claim is withdrawn.

**Re-check (2026-10-18, follow-up E1).** The withdrawn probe numbers were re-run from the
cached hidden states with question-grouped 5-fold cross-validation, nested layer selection,
shuffled-label controls, a text-only baseline (TF-IDF on prompt plus completion) and 95%
question-level cluster bootstrap intervals. Outputs: results/followups/e1_results.json.

| Quantity | Earlier value | Re-check (nested layer selection) | Text-only baseline |
|---|---|---|---|
| MATH in-domain probe AUROC | 0.828 | 0.824 [0.806, 0.843] | 0.776 |
| GSM8K in-domain probe AUROC | 0.751 | 0.779 [0.756, 0.802] | 0.690 |
| MMLU-STEM in-domain probe AUROC | 0.673 | 0.689 [0.660, 0.716] | 0.641 |
| MATH probe transferred to GSM8K | 0.744 | 0.744 [0.720, 0.767] | |
| MATH probe transferred to MMLU-STEM | 0.574 | 0.574 [0.543, 0.604] | |

Shuffled-label controls stay near chance (0.51 to 0.55). On MATH the text-only baseline comes
within 0.05 of the probe, so part of the probe's signal is likely visible in the text itself.

Probe versus self-consistency on the matched population (per problem, target = K=16 majority
answer correct):

| | MATH (n = 1663) | GSM8K (n = 1033) |
|---|---|---|
| probe, 1 forward pass (nested) | 0.776 | 0.721 |
| free logit confidence | 0.563 | 0.595 |
| self-consistency K=2 / 4 / 8 / 16 | 0.725 / 0.831 / 0.878 / 0.904 | 0.637 / 0.736 / 0.816 / 0.850 |
| K* where self-consistency matches the probe | 2.96 (non-nested 3.13 [2.70, 3.59]) | 3.69 (non-nested 3.68 [2.82, 5.32]) |

The self-consistency target shares samples with the K=16 majority label, so self-consistency
is optimistic here and K* is, if anything, an underestimate.

**Retracted: TruthfulQA-MC probe AUROC 0.990.** In this project's TruthfulQA-MC data the
correct option is always listed first (all 817 questions), and options were never shuffled.
"Correct" is therefore identical to "the model chose option A": a baseline that reads only the
chosen letter reaches AUROC 1.0. The probe (0.995 on re-check) reads the model's own answer
choice, not a correctness signal. TruthfulQA-MC was re-evaluated with shuffled options on
2026-10-18 (section 9, Q1); accuracy and calibration numbers from the original runs are
superseded by that re-run. The MATH-to-TruthfulQA transfer value
(0.663) is withdrawn with it. MMLU-STEM is not affected (correct options are spread over all
four positions).

---

## 6. Retracted: single-shot utility under asymmetric cost

An earlier analysis reported that the maximum-punishment policy beats the symmetric baseline
in single-shot expected utility above a cost crossover. That result relied on a cost metric
constructed for the test, and S0 was compared with no single-pass confidence signal of its own.
Under the standard matched-compute metrics no punishment condition shows an advantage. It is
retracted (see research/FINAL_VERDICT.md and research/FINDINGS.md).

---

## 7. Cross-task transfer of the abstention threshold

Using the trained 1.5B adapters as zero-shot PRM800K step-correctness judges: judge
abstention rises from S0 to S4 at judging temperature 0 (0.342, 0.367, 0.386, 0.426, 0.549)
and keeps the same order at 0.7. At 1.0 the order is not monotone (S0 0.717, S1 0.699,
S2 0.743, S3 0.718, S4 0.747; n = 2000). Judging accuracy on attempted items is flat across
S0 to S4 (0.635 to 0.648 at temperature 0). The abstention tendency learned on math
generation carries over to step judging; the judging competence does not change.

---

## 9. Follow-ups (2026-10-18)

Run on the cached K=16 rollouts (1.5B, one training seed per condition). Scripts in
research/followups/, outputs in results/followups/ and results/abstention_ci.json.

**Abstention error bars.** 95% bootstrap intervals over problems for every condition and
benchmark (fig01). Half-widths are at most about 0.01 on MATH, GSM8K, MMLU-STEM and
TruthfulQA-MC and up to 0.095 on AIME 2024 and 2025 (30 problems each). These cover
evaluation noise only, not training-seed variation.

**Wise versus wasteful abstention.** Problem difficulty is estimated from the base model's
K=16 samples (accuracy among its answered samples). An abstention is wise if that estimate is
below 0.5. A random abstainer would score the share of hard problems (0.501 on MATH, 0.290 on
GSM8K).

| Condition | MATH abstention | MATH wise fraction [95% CI] | MATH AUROC | GSM8K wise fraction |
|---|---|---|---|---|
| P0 | 0.117 | 0.872 | 0.771 | 0.495 |
| S0 | 0.142 | 0.797 [0.780, 0.813] | 0.758 | 0.288 |
| C1 | 0.151 | 0.779 | 0.756 | |
| S1 | 0.163 | 0.755 | 0.748 | |
| C2 | 0.178 | 0.716 | 0.727 | |
| S2 | 0.183 | 0.716 | 0.732 | |
| S3 | 0.204 | 0.682 [0.665, 0.699] | 0.714 | 0.286 |
| S4 | 0.613 | 0.526 [0.511, 0.540] | 0.570 | 0.291 |

AUROC is problem-level abstention count against the hard-problem label. On MATH the wise
fraction falls as the implied threshold rises; on GSM8K abstention is unrelated to difficulty
for S0 to S4 (AUROC 0.48 to 0.51). On MMLU-STEM and TruthfulQA-MC the wise fraction is at or
below the random baseline. Variants using S0 or the condition itself as the difficulty
reference give the same pattern. P0 on GSM8K abstains on under 1% of samples.

**P(True) and semantic entropy (E3).** Same problems as the probe comparison (n = 1663 MATH,
1033 GSM8K), S0 model, target = whether the K=16 majority answer is correct (as in
research/matched_population.py). AUROC with 95% bootstrap intervals over problems.

| Signal | Compute | MATH | GSM8K |
|---|---|---|---|
| P(True), zero-shot, (A)/(B) | 1 forward pass | 0.482 [0.452, 0.513] | 0.571 [0.527, 0.618] |
| P(True), zero-shot, True/False | 1 forward pass | 0.565 [0.537, 0.596] | 0.593 [0.547, 0.639] |
| P(True), 2-shot, (A)/(B) | 1 forward pass | 0.668 [0.641, 0.695] | 0.599 [0.557, 0.638] |
| Semantic entropy, K=2 | 2 generations | 0.709 [0.687, 0.730] | 0.626 [0.592, 0.659] |
| Semantic entropy, K=4 | 4 generations | 0.808 [0.786, 0.828] | 0.756 [0.720, 0.789] |
| Semantic entropy, K=8 | 8 generations | 0.851 [0.832, 0.869] | 0.829 [0.797, 0.858] |
| Semantic entropy, K=16 | 16 generations | 0.873 [0.856, 0.889] | 0.855 [0.826, 0.882] |

Semantic entropy clusters final answers by math_verify equivalence, with abstentions as their
own cluster. P(True) asks the same model whether the majority answer is correct. Caveat: the
target is computed from the same 16 samples that semantic entropy draws from, so the larger-K
rows share samples with the label and are optimistic; the same applies to self-consistency.
The 2-shot negative example is the gold answer plus 2, which is crude.

**TruthfulQA-MC with shuffled options (Q1).** All 817 questions re-evaluated with a fixed
random permutation of the answer options (gold position consistent with uniform: chi-square
8.53, 12 df), K=16, same prompt, sampling and scoring as the original runs. 95% bootstrap
intervals over questions.

| Condition | Accuracy, original -> shuffled | Abstention, original -> shuffled | Accuracy when answering, original -> shuffled |
|---|---|---|---|
| base | 0.078 -> 0.078 [0.070, 0.087] | 0.790 -> 0.779 | 0.372 -> 0.354 [0.325, 0.384] |
| S0 | 0.087 -> 0.085 | 0.771 -> 0.757 | 0.379 -> 0.351 |
| S1 | 0.084 -> 0.082 | 0.778 -> 0.766 | 0.377 -> 0.351 |
| S2 | 0.082 -> 0.079 | 0.783 -> 0.772 | 0.376 -> 0.346 |
| S3 | 0.075 -> 0.073 | 0.799 -> 0.786 | 0.373 -> 0.344 |
| S4 | 0.051 -> 0.048 | 0.867 -> 0.863 | 0.381 -> 0.346 |
| C1 | 0.086 -> 0.085 | 0.772 -> 0.760 | 0.377 -> 0.352 |
| C2 | 0.080 -> 0.077 | 0.787 -> 0.778 | 0.375 -> 0.347 |
| P0 | 0.140 -> 0.142 [0.131, 0.154] | 0.637 -> 0.612 | 0.385 -> 0.366 |

Among answered samples the rate of choosing the first-listed option falls from 0.37 to 0.38
to 0.21 to 0.22 (chance 0.226), so the original accuracies were inflated by a first-option
preference only slightly; the ordering of conditions is unchanged. Wise fraction of
abstentions: 0.618 to 0.621 against a random baseline of 0.625, AUROC 0.46 to 0.48 (all
intervals include 0.5): abstention on TruthfulQA-MC is not targeted at hard questions.
S0 correctness probe on the shuffled data (1600 answered rows, 582 questions, question-grouped
5-fold CV): nested 0.792 [0.762, 0.821]; chosen-letter baseline 0.589; TF-IDF text baseline
0.669; shuffled-label controls 0.53. The probe signal survives the shuffle.
Outputs: results/followups/q1_results.json.

**Offline advantage analysis (Q3).** S0's cached K=16 MATH and GSM8K samples, split into
groups of 8, re-scored with reward (correct 1, wrong 0, abstain t) for each condition's t, no
training. Easy = base-model accuracy on answered samples >= 0.5.

| t | MATH: easy abstentions with positive advantage | MATH AUROC (hard vs easy) | GSM8K: easy abstentions with positive advantage |
|---|---|---|---|
| 0 (P0) | 0.00 | 0.927 | 0.00 |
| 0.50 (S0) | 0.19 | 0.936 | 0.05 |
| 0.67 (S1) | 0.41 | 0.915 | 0.21 |
| 0.80 (S2) | 0.49 | 0.893 | 0.28 |
| 0.91 (S3) | 0.69 | 0.859 | 0.49 |
| 1.00 (S4) | 0.69 | 0.814 | 0.49 |

Higher t makes abstaining on easy problems advantageous, which explains the falling wise
fraction above. Two per-question abstain rewards (t_q = 1 - within-group answer agreement; t_q =
1 - mean probe score, thin coverage) did not beat fixed t = 0.5 on this AUROC. In every group of
exactly two outcome types, advantages are identical for all interior t (0.50 to 0.91). Rollouts
are held fixed, so this says nothing about how a policy trained on these rewards would behave.
Outputs: results/followups/q3_results.json.

**Stale summary file.** results/analysis_summary.json (dated 2026-05-25) predates the final
S1 and S3 rollouts (2026-05-26): 11 S1/S3 abstention cells differ from the raw rollouts by up
to 0.004. Fixed on 2026-10-18: the summary files were regenerated from the final rollouts
and fig02 to fig08 redrawn (see CHANGELOG.md).

---

## 8. Reference trace corpus (Phase 1)

6000 reasoning traces generated with qwen3:30b (MoE A3B) via Ollama: 2000 MATH Level 4 to 5
traces (36.1% correct) and 4000 GSM8K traces (64.0% correct). The mixed-class pool usable as
judge-evaluation items is 299 problems (112 MATH, 187 GSM8K). GSM8K is the primary
error-detection source because MATH traces were partly truncated at the 4096-token cap.

---

## The 23-finding catalogue

The offline analysis produced 23 publication-grade findings. The peer-review pass retained three
as sound, one as understated, and tightened five that were overstated. Headlines:

- **A.** Corrected ECE (mode-vote-confidence versus mode-correct binary).
- **B.** Expected-value threshold recovery test.
- **C.** Discrimination index (L5 abstention over L1 abstention on MATH).
- **D.** C2-versus-S2 equivalence on every corrected metric.
- **E.** Capability preservation gap (Pass@16 minus Pass@1).
- **F.** Token efficiency: asymmetric reward saves compute.
- **G.** Answer entropy: S4 near-collapse on GSM8K.
- **H.** Reliability bins show systematic under-confidence.
- **I.** Oracle-adjusted selectivity.
- **J.** Cross-condition Jaccard on per-problem abstention masks.
- **K.** Per-problem stacking: 92.4% MATH and 93.9% GSM8K, 25 to 98 sigma above the
  empirical null.
- **L.** The recovered difficulty axis is human-meaningful.
- **M.** Asymmetric pressure hurts per-attempt accuracy even on consistently-attempted
  problems.
- **N.** Always-abstained problems are not capability-bound.
- **O.** GRPO normalization reduction: the seven-condition sweep is a one-dimensional
  abstention-bias dial. (The core contribution.)

See [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) and the supporting documents under
[research/](research/) for the evidence trail behind each finding.
