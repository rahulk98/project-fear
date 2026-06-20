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

## 3. The 3B scale-up reproduces Finding O

K=32 calibration eval on Qwen 2.5 3B adapters. Per-attempt accuracy is matched across
conditions to within about 0.004 on every benchmark. Only the abstention rate moves.

| Condition (3B) | MATH acc | MMLU-STEM acc |
|---|---|---|
| FD1 | 0.418 | 0.183 |
| P0 | 0.419 | 0.184 |
| S0 | 0.422 | 0.185 |

The symmetric-penalty (S0) condition abstains roughly twice as often as the positive-only
conditions on math and gsm8k, while buying no accuracy. The 1.5B mechanism reappears
unchanged at 3B: an operating-point dial, not a capability dial.

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

The 31-point MATH gap is confidence-interval-separated. This reframes Finding O: z-score
normalization is not merely what hides the asymmetry, it is what keeps high-penalty training
stable. Unnormalized, the magnitude reaches behavior, but toward over-abstention and
degradation. Partial normalization (a global-mean baseline) sits monotonically between the
two, confirming the mechanism end to end.

---

## 5. The cheap win is a single forward-pass probe (compute-crossover law)

A linear probe on last-token hidden states, one forward pass, is a strong
selective-prediction signal. Recomputed on a matched population, per problem, with 2000x
bootstrap confidence intervals (AUROC):

| Signal | MATH (n=1663) | GSM8K (n=1033) |
|---|---|---|
| 1-pass probe | 0.785 [0.763, 0.807] | 0.720 [0.682, 0.756] |
| free logit confidence | 0.563 | 0.595 |
| self-consistency K=2 | 0.725 | 0.637 |
| self-consistency K=4 | 0.831 | 0.736 |
| self-consistency K=8 | 0.878 | 0.816 |
| self-consistency K=16 | 0.904 | 0.850 |

The probe significantly beats free logit confidence and two-sample self-consistency, and
equals roughly three self-consistency samples. The compute-crossover is K* = 3.13 on MATH
and K* = 3.68 on GSM8K. One cheap probe pass buys about three self-consistency samples of
selective-prediction quality.

**Cross-domain transfer** (math-trained probe, best layer 22, group-CV AUROC 0.828):

| Tested on | Transfer AUROC | In-domain ceiling |
|---|---|---|
| GSM8K | 0.744 | 0.751 |
| MMLU-STEM | 0.574 | 0.673 |
| TruthfulQA-MC | 0.663 | 0.990 |

The probe transfers near-losslessly within the reasoning family (math to gsm8k) but needs
per-family refitting for multiple-choice and factual tasks. On TruthfulQA the in-domain probe
reaches AUROC 0.990: the model internally separates right from wrong almost perfectly yet
does not express it, a striking internal-knowledge-versus-output gap on a hallucination
benchmark.

---

## 6. Where punishment training does pay off: single-shot under asymmetric cost

The honest boundary on the negative result. Under a deployment utility
`U(c) = P(correct) - c * P(wrong)` (abstention scores 0), evaluated at each policy's own
single-shot operating point, the maximum-punishment policy overtakes the symmetric baseline
above a benchmark-specific cost crossover c*.

| Benchmark | c* (S0 to best fear) |
|---|---|
| TruthfulQA-MC | 0.92 |
| MMLU-STEM | 1.70 |
| MATH | 2.14 |
| GSM8K | 2.63 |

On MATH at c=3 the bootstrap intervals are disjoint (S4 utility [-0.344, -0.275] versus S0
[-0.514, -0.409]). On TruthfulQA the crossover is below 1, so punishment training is the
better single-shot policy even when right and wrong answers are weighted equally. The win is
specific to single-sample inference: once you can afford two or more samples, post-hoc
confidence thresholding dominates again. The deployable niche for fear training is therefore
single-shot inference under asymmetric or honesty-critical cost.

---

## 7. Cross-task transfer of the abstention threshold

Using the trained adapters as zero-shot PRM800K step-correctness judges, across a judging
temperature sweep: the condition ordering on abstention is temperature-invariant (the
D-family commits most, S4 abstains most, the S-sweep is monotone S0 > S1 > S2 > S3 > S4 at
every temperature), while quality-when-attempted is reward-invariant. The abstention
threshold learned on math generation transfers to a different task (step judging), but the
competence does not. This is a cross-task confirmation that the reward family acts on the
abstain axis.

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
