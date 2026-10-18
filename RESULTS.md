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

## 5. Hidden-state probes: one shared correctness direction

A linear correctness probe trained on one condition's last-token hidden states decodes
correctness on every other condition almost as well as that condition's own probe. Across 14
trained conditions (1.5B), the mean off-diagonal AUROC is 0.989 on MATH and 0.992 on GSM8K,
against 0.998 and 1.000 on the diagonal (worst single pair: 0.787 on MATH, 0.832 on GSM8K).
All conditions appear to share one correctness direction and differ in abstention threshold.

Earlier versions of this file also reported a probe-versus-self-consistency compute crossover
(K*), cross-domain probe transfer, and a TruthfulQA-MC in-domain probe AUROC of 0.990. The
output files behind those numbers were not preserved, so they are withdrawn until re-run
with question-grouped cross-validation, a shuffled-label control and bootstrap intervals.

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

**Stale summary file.** results/analysis_summary.json (dated 2026-05-25) predates the final
S1 and S3 rollouts (2026-05-26): 11 S1/S3 abstention cells differ from the raw rollouts by up
to 0.004. fig01 now uses the raw values; fig02 to fig08 still read the summary file.

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
- **K.** Per-problem stacking: 92.4% MATH and 93.9% GSM8K, 25 to 99 sigma above the
  empirical null.
- **L.** The recovered difficulty axis is human-meaningful.
- **M.** Asymmetric pressure hurts per-attempt accuracy even on consistently-attempted
  problems.
- **N.** Always-abstained problems are not capability-bound.
- **O.** GRPO normalization reduction: the seven-condition sweep is a one-dimensional
  abstention-bias dial. (The core contribution.)

See [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) and the supporting documents under
[research/](research/) for the evidence trail behind each finding.
