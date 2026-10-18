# Under GRPO, asymmetric "don't guess" rewards reduce to a single abstention dial

*Rahul Krishnan. Work done March to June 2026, in collaboration with Sanjana Deshpande. Code: https://github.com/rahulk98/project-fear*

**TL;DR.** We trained Qwen 2.5 1.5B and 3B base models with GRPO under seven reward schemes that punish wrong answers more than they reward right ones, hoping the models would learn to abstain when they are likely to be wrong. The rewards changed how often the models abstained, but not accuracy on the questions they still answered, not how well their confidence ranked right versus wrong answers, and the extra abstention landed less on hard problems. The reason is elementary: after GRPO's within-group z-scoring, every (correct, wrong, abstain) reward triple is equivalent to (1, 0, t) for a single number t, so the seven-condition sweep is one dial. Concurrent work (Che et al. 2026) derives a closely related result formally; ours is the empirical side, with explicit abstention rewards, a 3B check, a no-normalization ablation and a cross-task transfer test.

## Why this matters for honesty

A model that answers every question, including the ones it does not know, will hallucinate. One proposed fix is to change the incentives: score a wrong answer worse than an abstention, so that guessing stops paying. Kalai et al. (2025) argue that binary right/wrong grading is a root cause of hallucination for this reason, and several recent RL papers add an abstention option with a wrong-answer penalty (TruthRL, Reinforced Hesitation, Abstain-R1, KARL).

The decision theory is clean. With reward $+\alpha$ for a correct answer, $-\beta$ for a wrong one and $r_a$ for abstaining, an expected-value maximizer should answer only when its probability of being right exceeds

$$\tau = \frac{\beta + r_a}{\alpha + \beta}.$$

The hope is that RL then teaches the model to abstain on the specific questions it would get wrong, a per-question calibration skill. Under GRPO as usually implemented, the reward cannot teach that skill; it can only move a global threshold.

## Setup

**Training.** Qwen 2.5 1.5B base (main) and Qwen 2.5 3B base. GRPO via TRL 0.15.2 with LoRA, group size $G = 8$, 1500 steps, learning rate $10^{-6}$, KL coefficient 0.04, sampling temperature 0.7. Training problems: MATH train levels 4 and 5 plus the longest quarter of GSM8K train (5883 problems). Answers are graded symbolically with math_verify.

**Abstention.** There is no "I don't know" token. A response with no boxed final answer counts as an abstention and receives $r_a$, so the model has to discover silence as a strategy.

**Conditions.**

| Condition | $\alpha$ (correct) | $\beta$ (wrong) | $r_a$ (abstain) | $\tau$ |
|---|---|---|---|---|
| S0 | 1.0 | 1.0 | 0.0 | 0.50 |
| S1 | 1.0 | 2.0 | 0.0 | 0.67 |
| S2 | 0.5 | 2.0 | 0.0 | 0.80 |
| S3 | 0.1 | 1.0 | 0.0 | 0.91 |
| S4 | 0.0 | 1.0 | 0.0 | 1.00 |
| C1 | 1.0 | 1.0 | 0.2 | 0.60 |
| C2 | 1.0 | 1.0 | 0.5 | 0.75 |

S0 to S4 raise the wrong-answer penalty (implicit abstention). C1 and C2 pay for abstaining (explicit abstention). C2 and S2 reach nearly the same threshold through different mechanisms. A positive-only control P0 uses +1 / 0 / 0.

**Evaluation.** K=16 sampled rollouts per problem (K=32 for some runs) and greedy decoding, on MATH, GSM8K, AIME 2024 and 2025, MMLU-STEM and TruthfulQA-MC. We report abstention rate, accuracy on attempted questions ("accuracy when answering"), and selective-prediction curves.

## The mechanism: z-scoring leaves one degree of freedom

GRPO samples $G$ responses per prompt and uses the within-group z-score as the advantage:

$$A_i = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r)}.$$

This is invariant to any transform $r \mapsto a r + b$ with $a > 0$. Map the wrong-answer reward $-\beta$ to 0 and the correct-answer reward $\alpha$ to 1. The abstain reward then lands at

$$t = \frac{r_a + \beta}{\alpha + \beta},$$

which is exactly the threshold $\tau$. Every condition in the table is therefore the triple (correct = 1, wrong = 0, abstain = t), and GRPO can only tell conditions apart through $t$. Two consequences follow.

1. **Two-outcome groups ignore the reward values.** If a group contains only two distinct outcomes, for example only correct and wrong answers, every condition gives the same advantages (for groups mixing abstention with one other outcome this holds for $0 < t < 1$; at $t = 0$ or $t = 1$ abstention ties with a wrong or a correct answer). With $k$ of $G$ responses at the higher reward they are $+\sqrt{(G-k)(G-1)/(Gk)}$ and $-\sqrt{k(G-1)/(G(G-k))}$, functions of the counts alone.
2. **Three-outcome groups see only $t$.** When correct, wrong and abstained responses all occur, the design enters only through where abstention sits between wrong and correct.

A worked example: a group of 4 with one correct, two wrong and one abstained response, z-scored with the sample standard deviation as in the trainer.

| Condition | $t$ | correct | wrong | abstain |
|---|---|---|---|---|
| S0 | 0.50 | +1.306 | -0.783 | +0.261 |
| C1 | 0.60 | +1.225 | -0.816 | +0.408 |
| C2 | 0.75 | +1.091 | -0.849 | +0.606 |
| S2 | 0.80 | +1.046 | -0.856 | +0.665 |
| S3 | 0.91 | +0.946 | -0.864 | +0.782 |
| S4 | 1.00 | +0.866 | -0.866 | +0.866 |

By contrast, a group of 8 with two correct and six wrong answers gets advantages +1.620 and -0.540 under all seven conditions.

Raising $t$ raises the abstain advantage in mixed groups, which pushes the model to abstain more. But the reward values carry no information about which question the model is likely to get wrong; any per-question signal comes from the group composition, which is the same for every condition. The sweep should move a global abstention rate and leave the per-question ranking alone. S4 is a degenerate corner: at $t = 1$, abstaining is worth as much as being right.

None of this is deep: affine invariance of the z-score is standard, and `tests/test_advantage_invariance.py` in the repository checks it with the trainer's z-score formula. The point is the consequence for reward design: tuning the wrong-answer penalty under GRPO is tuning one scalar.

## Results

All numbers are recomputed from the raw evaluation files (per-run metrics and rollouts). Unless stated otherwise, each condition is one training seed; see Limitations.

### 1. The sweep moves abstention, not accuracy when answering

Greedy decoding on MATH test (5000 problems), 1.5B:

| Condition | $t$ | Attempt rate | Accuracy when answering |
|---|---|---|---|
| S0 | 0.50 | 0.867 | 0.618 |
| S1 | 0.67 | 0.857 | 0.617 |
| S2 | 0.80 | 0.849 | 0.612 |
| S3 | 0.91 | 0.836 | 0.602 |
| S4 | 1.00 | 0.325 | 0.513 |

As $t$ rises the model answers less often, and its remaining answers are no more accurate: if punishment taught the model to drop the questions it would get wrong, accuracy when answering would rise, but it falls slightly through S3 and sharply at S4. GSM8K is the same (attempt rate and accuracy when answering: S0 0.951 and 0.722, S3 0.717 and 0.699, S4 0.017 and 0.591).

With K=16 sampling, accuracy when answering on MATH goes from 0.535 (S0) to 0.510 (S3) and 0.410 (S4). This is not only a selection effect: on the 1324 MATH problems that all seven conditions attempt by majority vote, per-attempt accuracy still drops from 0.626 (S0) to 0.494 (S4).

![Abstention rate by condition and benchmark](../docs/figures/fig01_abstention_sweep.png)

*Figure 1. Abstention rate (mean over K=16 samples) per condition on six benchmarks, Qwen 2.5 1.5B. Error bars are 95% bootstrap intervals over problems; they show evaluation noise only, since each condition is one training seed.*

Abstention rises with $t$ on every benchmark except AIME 2024 and 2025, where S1 abstains slightly less than S0; with 30 problems each, those intervals are wide and overlap.

### 2. The confidence ranking does not change

Better per-question calibration would show up as the model's own confidence (agreement of the majority answer across K=16 samples) separating right from wrong answers better. It does not: on MATH the selective-accuracy curves of S0 to S3, C1 and C2 lie on top of each other, and only S4 is worse (Figure 2).

![Selective accuracy curves on MATH](../docs/figures/fig04_auacc_curves.png)

*Figure 2. Accuracy on the covered subset when MATH test problems are ranked by mode-vote confidence, 1.5B.*

Area under the risk-coverage curve (AURC, lower is better), recomputed from the rollouts with the same ranking:

| Condition | MATH AURC | GSM8K AURC |
|---|---|---|
| P0 (positive-only) | 0.1067 | 0.0409 |
| C1 | 0.1114 | 0.0511 |
| S0 | 0.1125 | 0.0497 |
| S1 | 0.1141 | 0.0530 |
| C2 | 0.1158 | 0.0604 |
| S2 | 0.1165 | 0.0631 |
| base model | 0.1196 | 0.0663 |
| S3 | 0.1219 | 0.0644 |
| S4 | 0.2537 | 0.2859 |

No punishment condition beats the symmetric baseline, and the positive-only control P0 ($t = 0$) is best.

### 3. Explicit and implicit abstention are interchangeable

C2 pays for abstaining; S2 punishes wrong answers; their thresholds are 0.75 and 0.80. With K=16 they have MATH abstention 0.178 and 0.183 and accuracy when answering 0.518 and 0.519 (GSM8K: 0.339 and 0.347, 0.653 and 0.652). Two different-looking designs give the same policy.

### 4. The 3B check

At 3B we trained S0 and P0 (one seed each, K=32 evaluation), which differ only in $t$. MATH accuracy over all problems is 0.422 for S0 and 0.419 for P0, while S0 abstains more: 0.069 versus 0.032 on MATH, 0.067 versus 0.010 on GSM8K. On MMLU-STEM accuracy (0.185 versus 0.184) and abstention (0.653 versus 0.654) match. S0's MATH accuracy when answering is therefore about 2 points higher (0.453 versus 0.432); with one seed per condition we cannot tell whether that gap is real. On GSM8K, S0 has lower accuracy (0.689 versus 0.729).

### 5. Removing the normalization lets the magnitude through, badly

We trained S0 and S2 with a signed, unnormalized advantage (1.5B, one seed, K=32):

| Benchmark | S0 abstention | S2 abstention |
|---|---|---|
| MATH | 0.152 | 0.461 |
| GSM8K | 0.186 | 0.797 |
| MMLU-STEM | 0.484 | 0.934 |

Now the penalty reaches behavior: S2 abstains three times as often as S0 on MATH. But it moves toward over-abstention, not selective abstention. Subtracting only a global mean gives intermediate values (S0 versus S2: 0.162 versus 0.226 on MATH, 0.280 versus 0.502 on GSM8K). The z-score hides the asymmetry, and it also keeps high-penalty training from sliding into refusal.

### 6. The threshold transfers to another task, competence does not

We used the 1.5B adapters zero-shot as step-correctness judges on PRM800K. Judge abstention rises with $t$: 0.342, 0.367, 0.386, 0.426 and 0.549 for S0 to S4 at temperature 0, with the same order at 0.7. At 1.0 the order breaks (S0 0.717, S1 0.699, S2 0.743, S3 0.718, S4 0.747; n = 2000). Judging accuracy on attempted items stays flat (0.635 to 0.648 at temperature 0): the willingness to abstain carries over, the competence does not.

### 7. When the model abstains, was it likely to be wrong?

We estimate each problem's difficulty from the base model's K=16 samples (accuracy among its answered samples), independent of the condition being judged, and call an abstention "wise" if that accuracy is below 0.5. A random abstainer's wise fraction equals the share of hard problems.

| Condition | $t$ | MATH abstention | MATH wise fraction [95% CI] | GSM8K wise fraction |
|---|---|---|---|---|
| P0 | 0 | 0.117 | 0.872 | 0.495 |
| S0 | 0.50 | 0.142 | 0.797 [0.780, 0.813] | 0.288 |
| C1 | 0.60 | 0.151 | 0.779 | |
| S1 | 0.67 | 0.163 | 0.755 | |
| C2 | 0.75 | 0.178 | 0.716 | |
| S2 | 0.80 | 0.183 | 0.716 | |
| S3 | 0.91 | 0.204 | 0.682 [0.665, 0.699] | 0.286 |
| S4 | 1.00 | 0.613 | 0.526 [0.511, 0.540] | 0.291 |
| random abstainer | | | 0.501 | 0.290 |

On MATH, abstentions land on hard problems more often than chance, but the wise fraction falls as $t$ rises, and at S4 it is close to random. On GSM8K abstention is unrelated to difficulty in every penalty condition, and on MMLU-STEM it is at or slightly below the random baseline. Raising the penalty buys more abstention, not better-targeted abstention. The advantages show why. Holding S0's samples fixed and changing only $t$, the share of abstentions on easy problems that receive a positive advantage rises from 0.19 at $t = 0.5$ to 0.69 at $t = 0.91$ on MATH (0.05 to 0.49 on GSM8K): a higher $t$ rewards abstaining on problems the model would have solved. Caveat: the difficulty estimate uses only 16 samples per problem.

### 8. The information is in the model, but the reward cannot use it

A linear probe on the S0 model's last-token hidden state predicts whether its majority answer is correct. On the same problems we compare it with other confidence signals (AUROC, question-grouped cross-validation, nested layer selection):

| Signal | Cost | MATH (n = 1663) | GSM8K (n = 1033) |
|---|---|---|---|
| linear probe | 1 forward pass | 0.776 | 0.721 |
| P(True), zero-shot | 1 forward pass | 0.48 to 0.57 | 0.57 to 0.59 |
| semantic entropy, K=4 | 4 generations | 0.808 | 0.756 |
| self-consistency, K=2 / 4 / 8 | 2 / 4 / 8 generations | 0.725 / 0.831 / 0.878 | 0.637 / 0.736 / 0.816 |

One probe pass is worth about three self-consistency samples (2.96 on MATH, 3.69 on GSM8K), while asking the model whether its answer is true is near chance at this scale. A text-only baseline (TF-IDF on question and answer) reaches 0.776 against the probe's 0.824 in a per-sample version of the task on MATH, so part of the signal is visible in the text itself. The sampled rows share samples with the label and are optimistic. The point: a per-question confidence signal exists in the model's representations, and the z-scored reward has no channel to use it.

## What we can and cannot conclude

**Observed.** Across seven conditions at 1.5B, abstention moved with $t$ while accuracy when answering stayed flat or fell, the confidence ranking did not change, and extra abstention became less targeted at hard problems. The two matched-threshold conditions behaved alike. At 3B, the two conditions differed in abstention, not MATH accuracy. Without normalization, the reward magnitude mattered, in the over-abstaining direction.

**Inferred.** Under z-scored GRPO with a scalar correct/wrong/abstain reward, the penalty acts as a global abstention bias, not a per-question calibration signal.

**Not shown.** That asymmetric rewards cannot teach calibration with other estimators, reward shapes, larger models or longer training.

## Limitations

- **Scale.** 1.5B and 3B only, trained on one shared A100 40GB. 7B was out of reach.
- **Seeds.** Most runs have one seed. S0, S3 and C2 have a second seed at K=16, and the seeds agree closely on MATH (accuracy 0.459 and 0.458, 0.406 and 0.401, 0.426 and 0.426). All other conditions, the 3B runs and the ablations have one seed and no error bars.
- **3B coverage.** Only S0 and P0 were trained at 3B: two points on the dial, not the sweep.
- **Abstention format.** "No boxed answer" mixes deliberate refusal with failures to finish. Mean completion length falls with $t$ (1519 characters for S0, 1466 for S3, 976 for S4 on MATH), which argues against truncation as the cause.
- **TruthfulQA-MC.** In our copy of the data the correct option is always listed first, so accuracy there measures how often the model picks option A. We report only abstention on it. (This also explains a TruthfulQA probe AUROC of 0.99 in our earlier notes, now retracted.)
- **Estimators.** We tested z-score, global-mean and unnormalized advantages, not RLOO, Dr. GRPO or decoupled abstention advantages.

## Related work and what is new

- **Che, Yuan, Zhao and Yu (2026), arXiv 2608.00301**, is the closest work. For the +1 / $-\lambda$ / 0 rule they show that in the sparse-answer regime GRPO's group normalization replaces every designed penalty with an effective penalty of one, moving the learned threshold to 1/2, with experiments at 1.5B and 7B. Their treatment is formal and covers penalty-only rules; ours was done independently and is empirical.
- **Liu et al. (2025), arXiv 2503.20783 (Dr. GRPO)**, show that dividing by the group standard deviation biases updates by question difficulty, and remove it. They do not discuss reward-scale invariance.
- **Zhu et al. (2025), arXiv 2506.01347**, study negative-sample reinforcement; our S3 matches their weighting.
- **Abstention and calibration rewards.** TruthRL (arXiv 2509.25760) uses a ternary reward with z-scored GRPO. Reinforced Hesitation (arXiv 2511.11500) uses +1 / 0 / $-\lambda$ without dividing by the standard deviation. TIAR (arXiv 2605.25850) notes that an abstention reward shifts the group statistics and decouples the abstention advantage. RLCR (arXiv 2507.16806) rewards verbalized confidence with a Brier score.

What we think is new: the reduction stated for full three-level designs including explicit abstention rewards (every scheme is (1, 0, t)), and evidence that a full sweep behaves as that one dial: the C2 and S2 equivalence, the unchanged confidence ranking, the less-targeted abstention, the ablation and the transfer to step judging.

## Next steps

1. **Rewards that vary within an outcome class.** Any reward under which two correct answers can score differently escapes the reduction. A verbalized confidence scored with a proper scoring rule (as in RLCR) is one option. A probe score is another: the probe in section 8 carries per-question correctness information, so a probe-shaped reward would add a per-sample signal that group composition at $G = 8$ cannot.
   Caution: in an offline check on cached samples (no training), abstention rewards set by within-group answer agreement or by the probe score did not separate hard from easy problems better than a fixed $t = 0.5$.
2. **Estimators that keep magnitude without collapsing.** RLOO or a mean-only baseline keep the reward scale, but our ablation suggests they need a guard against over-abstention, such as a decoupled abstention advantage (TIAR) or Che et al.'s report-level repair.
3. **Measure per-question calibration directly.** Report the wise fraction of abstentions (section 7) next to the abstention rate, so that more abstention is not mistaken for better abstention. The repository packages this as an Inspect eval that runs on any model (`evals/abstention/`).
4. **More seeds** for the main 1.5B conditions, and the no-normalization ablation at 3B.

## Acknowledgements

Sanjana Deshpande collaborated on the project.
