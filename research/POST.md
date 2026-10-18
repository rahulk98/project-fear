# The abstention dial: what asymmetric rewards actually change under GRPO

*Rahul Krishnan. An academic project, March to June 2026, in collaboration with Sanjana Deshpande. Code, data summaries and an Inspect eval: https://github.com/rahulk98/project-fear*

**TL;DR.** I trained Qwen 2.5 1.5B with GRPO under seven reward schemes that punish wrong answers more than they reward right ones, hoping the model would learn to abstain when it is likely to be wrong. Three findings:

1. **One parameter.** After GRPO's within-group z-scoring, every (correct, wrong, abstain) reward triple is equivalent to (1, 0, t) for a single number t. Seven reward designs are seven settings of one dial. The math is elementary, and concurrent work (Che et al. 2026) treats a closely related effect formally.
2. **More abstention, not better abstention.** Turning the dial up made the model abstain more often, but accuracy on the questions it still answered went down slightly, its confidence ranking did not improve, and the extra abstentions landed less on hard problems. A closer look shows the "abstentions" were almost never refusals: the model ran out of tokens, or, under heavy penalties, answered in Python code or plain text so that the grader could not score it.
3. **The signal was there.** A linear probe on the hidden state of the symmetric-reward model predicts whether its answer is correct. Abstaining wherever that cheap signal is lowest beats the penalty-trained models at the same answer rate on GSM8K, and matches them on MATH.

This is a small-scale academic project (1.5B, mostly one training seed), not a paper. Corrections made while preparing this write-up are listed at the end.

## Why this matters

A model that answers every question, including the ones it does not know, will hallucinate. One proposed fix is to change the incentives: score a wrong answer worse than an abstention, so that guessing stops paying. Kalai et al. (2025) argue that binary right/wrong grading is a root cause of hallucination for this reason, and several recent RL papers add an abstention option with a wrong-answer penalty (TruthRL, Reinforced Hesitation, Abstain-R1, KARL).

With reward $+\alpha$ for a correct answer, $-\beta$ for a wrong one and $r_a$ for abstaining, an expected-value maximizer should answer only when its probability of being right exceeds

$$\tau = \frac{\beta + r_a}{\alpha + \beta}.$$

The hope is that RL teaches the model to abstain on the specific questions it would get wrong. This project asks what the reward actually changes.

## Setup

**Training.** Qwen 2.5 1.5B base. GRPO via TRL 0.15.2 with LoRA, group size $G = 8$, 1500 steps, learning rate $10^{-6}$, KL coefficient 0.04, sampling temperature 0.7, maximum 1024 generated tokens. Training problems: MATH train levels 4 and 5 plus the longest quarter of GSM8K train (5883 problems). Answers are graded with math_verify on the boxed final answer.

**Abstention.** There is no "I don't know" token. A response with no boxed final answer counts as an abstention and receives $r_a$. Finding 2 shows what this definition ends up measuring.

**Conditions.**

| Condition | $\alpha$ (correct) | $\beta$ (wrong) | $r_a$ (abstain) | $t = \tau$ |
|---|---|---|---|---|
| P0 | 1.0 | 0.0 | 0.0 | 0 |
| S0 | 1.0 | 1.0 | 0.0 | 0.50 |
| C1 | 1.0 | 1.0 | 0.2 | 0.60 |
| S1 | 1.0 | 2.0 | 0.0 | 0.67 |
| C2 | 1.0 | 1.0 | 0.5 | 0.75 |
| S2 | 0.5 | 2.0 | 0.0 | 0.80 |
| S3 | 0.1 | 1.0 | 0.0 | 0.91 |
| S4 | 0.0 | 1.0 | 0.0 | 1.00 |

S1 to S4 raise the wrong-answer penalty; C1 and C2 instead pay for abstaining; P0 is a positive-only control.

**Evaluation.** K=16 sampled answers per problem and greedy decoding on MATH (5000 problems), GSM8K (1319), AIME 2024 and 2025, MMLU-STEM and TruthfulQA-MC. Intervals are 95% bootstrap intervals over problems; differences between conditions use paired resampling.

## Finding 1: z-scoring leaves one degree of freedom

GRPO samples $G$ responses per prompt and uses the within-group z-score as the advantage:

$$A_i = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r)}.$$

This is invariant to any transform $r \mapsto a r + b$ with $a > 0$. Map the wrong-answer reward to 0 and the correct-answer reward to 1; the abstain reward lands at

$$t = \frac{r_a + \beta}{\alpha + \beta},$$

which is exactly $\tau$. Every condition is therefore the triple (correct = 1, wrong = 0, abstain = t). Two consequences:

1. **Two-outcome groups ignore the reward values.** A group with only correct and wrong answers gets the same advantages under every condition: with $k$ of $G$ correct, $+\sqrt{(G-k)(G-1)/(Gk)}$ and $-\sqrt{k(G-1)/(G(G-k))}$. (For groups mixing abstention with one other outcome this holds for $0 < t < 1$.)
2. **Three-outcome groups see only $t$.** The design enters only through where abstention sits between wrong and correct.

A worked example, for a group of 4 with one correct, two wrong and one abstained response (sample standard deviation, as in the trainer):

| Condition | $t$ | correct | wrong | abstain |
|---|---|---|---|---|
| S0 | 0.50 | +1.306 | -0.783 | +0.261 |
| C1 | 0.60 | +1.225 | -0.816 | +0.408 |
| C2 | 0.75 | +1.091 | -0.849 | +0.606 |
| S2 | 0.80 | +1.046 | -0.856 | +0.665 |
| S3 | 0.91 | +0.946 | -0.864 | +0.782 |
| S4 | 1.00 | +0.866 | -0.866 | +0.866 |

A group of 8 with two correct and six wrong answers gets +1.620 and -0.540 under all conditions.

As a numerical check with the trainer's exact advantage code on 12,638 real groups of 8 from the S0 model's samples, two different triples with the same $t = 0.75$, (1, -3, 0) and (1, -1, 0.5), give advantages that differ by at most 0.0007 (typical advantage size 0.6); the residue comes from the trainer's $10^{-4}$ stabilizer. Moving $t$ from 0.75 to 0.80 changes three-outcome advantages by about 4%. `tests/test_advantage_invariance.py` checks the identity.

Raising $t$ raises the abstain advantage in mixed groups, so the model abstains more. The reward values carry no information about which question the model is likely to get wrong; any per-question signal comes from the group's composition, which is the same for every condition. None of this is deep. The consequence for reward design is the point: tuning the wrong-answer penalty under GRPO is tuning one scalar.

## Finding 2: more abstention, not better abstention

**Accuracy when answering goes down, not up.** Greedy decoding on MATH:

| Condition | $t$ | Attempt rate | Accuracy when answering |
|---|---|---|---|
| S0 | 0.50 | 0.867 | 0.618 |
| S1 | 0.67 | 0.857 | 0.617 |
| S2 | 0.80 | 0.849 | 0.612 |
| S3 | 0.91 | 0.836 | 0.602 |
| S4 | 1.00 | 0.325 | 0.513 |

If the penalty taught the model to drop the questions it would get wrong, accuracy when answering would rise. It falls: S3 minus S0 is -0.016 [-0.027, -0.006] greedy, and -0.025 [-0.029, -0.022] on MATH and -0.027 [-0.035, -0.019] on GSM8K with K=16 sampling. On the 1324 MATH problems that every condition attempts by majority vote, per-attempt accuracy drops from 0.626 (S0) to 0.494 (S4).

![Abstention rate by condition and benchmark](../docs/figures/fig01_abstention_sweep.png)

*Figure 1. Abstention rate (mean over K=16 samples) per condition on six benchmarks. Error bars are 95% bootstrap intervals over problems; they show evaluation noise only, since each condition is one training seed.*

Abstention rises with $t$ on every benchmark except AIME 2024 and 2025 (30 problems each), where S1 abstains slightly less than S0 within wide intervals.

**The confidence ranking does not improve.** If the model became better calibrated, its own confidence (agreement of the majority answer across K=16 samples) would separate right from wrong answers better. On MATH the selective-accuracy curves of S0 to S3, C1 and C2 lie on top of each other, and only S4 is worse:

![Selective accuracy curves on MATH](../docs/figures/fig04_auacc_curves.png)

*Figure 2. Accuracy on the covered subset when MATH problems are ranked by majority-vote confidence.*

The area under the risk-coverage curve (AURC, lower is better) gets slightly worse as $t$ rises (S3 minus S0: +0.009 [0.006, 0.013] on MATH, +0.015 [0.006, 0.024] on GSM8K), and the positive-only control P0 is the best condition (P0 minus S0: -0.006 [-0.009, -0.002] on MATH).

| Condition | MATH AURC | GSM8K AURC |
|---|---|---|
| P0 | 0.107 | 0.041 |
| C1 | 0.111 | 0.051 |
| S0 | 0.113 | 0.050 |
| S1 | 0.114 | 0.053 |
| C2 | 0.116 | 0.060 |
| S2 | 0.117 | 0.063 |
| base model | 0.120 | 0.066 |
| S3 | 0.122 | 0.064 |
| S4 | 0.25 | 0.29 |

(S4's value depends on how ties in the confidence ranking are broken, between 0.23 and 0.29 on MATH.)

**Explicit and implicit abstention are interchangeable.** C2 pays for abstaining and S2 punishes wrong answers; their thresholds are close (0.75 and 0.80). Their accuracy when answering and AURC are indistinguishable (C2 minus S2 accuracy when answering: -0.001 [-0.003, 0.001] on MATH), and C2 abstains slightly less, as its lower $t$ predicts.

**The extra abstentions are less targeted.** We estimate each problem's difficulty from the base model's 16 samples and call an abstention "wise" if the base model gets that problem right less than half the time. A random abstainer would score the share of hard problems.

| Condition | $t$ | MATH abstention | MATH wise fraction [95% CI] | GSM8K wise fraction |
|---|---|---|---|---|
| P0 | 0 | 0.117 | 0.872 | 0.495 |
| S0 | 0.50 | 0.142 | 0.797 [0.780, 0.813] | 0.288 |
| S1 | 0.67 | 0.163 | 0.755 | |
| C2 | 0.75 | 0.178 | 0.716 | |
| S3 | 0.91 | 0.204 | 0.682 [0.665, 0.699] | 0.286 |
| S4 | 1.00 | 0.613 | 0.526 [0.511, 0.540] | 0.291 |
| random | | | 0.501 | 0.290 |

On MATH the wise fraction falls as $t$ rises (S3 minus S0: -0.115 [-0.125, -0.106]) and S4 is close to random. On GSM8K, MMLU-STEM and TruthfulQA-MC (with shuffled answer options), abstention is no better than random in any penalty condition. The advantages explain the MATH trend: holding the S0 model's samples fixed and changing only $t$, the share of abstentions on easy problems that receive a positive advantage rises from 0.19 at $t = 0.5$ to 0.69 at $t = 0.91$. A higher $t$ rewards abstaining on problems the model would have solved.

**What the "abstentions" actually are.** Reading the abstained completions changes the picture further. A rule-based classification of every abstained sample (checked by hand on a stratified sample; about 89% agreement weighted by category size):

| Condition, benchmark | Abstained samples | Hit the 1024-token limit | Answer stated but not boxed | Python code instead of an answer | Explicit refusal |
|---|---|---|---|---|---|
| P0, MATH | 9,340 | 99% | 1% | 0% | 0.0% |
| S0, MATH | 11,399 | 85% | 12% | 3% | 0.1% |
| S3, MATH | 16,308 | 58% | 19% | 22% | 0.3% |
| S4, MATH | 49,068 | 8% | 24% | 68% | 0.4% |
| S0, GSM8K | 3,650 | 3% | 92% | 4% | 0.5% |
| S4, GSM8K | 18,964 | 0.4% | 58% | 41% | 0.2% |

Explicit refusals are at most 0.5% in every cell. Under the symmetric reward, a MATH "abstention" is almost always a solution that ran out of tokens, which is why it correlates with difficulty. As the penalty grows, the model increasingly answers in a form the grader cannot score, Python code or an unboxed answer, which earns the abstain reward instead of the wrong-answer penalty. The penalty did not teach the model to say "I don't know"; it taught it to avoid being graded. That is a small, concrete instance of a model optimizing the measurement rather than the intended behavior, and a caution for any abstention reward defined by output format.

## Finding 3: the confidence signal exists, and using it works better

A linear probe on the S0 model's last-token hidden state predicts whether its majority answer is correct (question-grouped cross-validation, nested layer selection). Compared with other confidence signals on the same problems:

| Signal | Cost | MATH (n = 1663) | GSM8K (n = 1033) |
|---|---|---|---|
| linear probe | 1 forward pass | 0.776 | 0.721 |
| P(True), zero-shot | 1 forward pass | 0.48 to 0.57 | 0.57 to 0.59 |
| semantic entropy, K=4 | 4 generations | 0.808 | 0.756 |
| self-consistency, K=2 / 4 / 8 | 2 / 4 / 8 generations | 0.725 / 0.831 / 0.878 | 0.637 / 0.736 / 0.816 |

(AUROC. The sampled signals share samples with the label and are optimistic.) One probe pass is worth about three self-consistency samples, while asking the 1.5B model whether its answer is true is near chance. A text-only baseline (TF-IDF on question and answer) gets within about 0.05 of the probe on MATH, so part of the signal is visible in the text. On TruthfulQA-MC with shuffled options the probe reaches 0.79 [0.76, 0.82], against 0.67 text-only and 0.59 for the chosen option letter.

**Using the signal at the same answer rate.** The direct test: take the symmetric S0 model, answer greedily, and abstain on the problems where the probe (or the agreement of 4 samples) is lowest, choosing the cut-off so that S0 answers exactly as often as a penalty-trained model. Then compare accuracy when answering:

| Penalty-trained model | GSM8K answer rate | Its accuracy when answering | S0 + probe at the same rate | Difference [95% CI] |
|---|---|---|---|---|
| S1 | 0.911 | 0.678 | 0.707 | +0.029 [0.012, 0.045] |
| S2 | 0.841 | 0.678 | 0.725 | +0.047 [0.024, 0.071] |
| C2 | 0.839 | 0.672 | 0.726 | +0.053 [0.029, 0.078] |
| S3 | 0.722 | 0.662 | 0.772 | +0.110 [0.075, 0.141] |

On MATH the penalty-trained models answer about 90% of problems and the probe selector matches them within noise (S3: +0.016 [-0.001, 0.033]; with 4-sample agreement +0.020 [0.003, 0.037]). Against S4, which answers only 35% of MATH problems, the probe selector is 0.33 more accurate. Caveats: this uses the problems covered by the probe analysis, and the probe score comes from the model's sampled answers to the same question, not from the greedy answer itself.

So the information needed for well-targeted abstention is present in the symmetric model, cheaply, and the scalar penalty does not get the model to use it.

## Further checks

- **3B.** At 3B we trained only S0 and P0 (one seed each). MATH accuracy is the same (0.422 and 0.419) while S0 abstains more (0.069 versus 0.032 on MATH, 0.067 versus 0.010 on GSM8K). This is a consistency check at two points on the dial, not a scale-up of the sweep.
- **Without normalization.** Training S0 and S2 with an unnormalized signed advantage (1.5B, one seed) lets the penalty through: S2 abstains on 0.461 of MATH problems versus 0.152 for S0 (GSM8K 0.797 versus 0.186), toward over-abstention rather than selective abstention. Subtracting only a global mean gives intermediate values.
- **Another task.** Used zero-shot as step-correctness judges on PRM800K, the adapters' judge abstention rises with $t$ (0.342 for S0 to 0.549 for S4 at temperature 0, same order at 0.7, order broken at 1.0), while judging accuracy on attempted items stays flat (0.635 to 0.648).
- **Per-question rewards, offline.** On cached samples (no training), abstain rewards set per group by answer agreement or by the probe score did not separate hard from easy problems better than a fixed $t = 0.5$.

## What I can and cannot conclude

**Observed.** In this reward family at 1.5B, raising $t$ increased abstention while accuracy when answering fell slightly, the confidence ranking did not improve, and abstention became less targeted. The abstentions were almost never refusals. A cheap confidence signal from the symmetric model did as well or better at the same answer rate.

**Inferred.** Under z-scored GRPO with a scalar correct/wrong/abstain reward and format-based abstention, the penalty acts as a global push away from gradeable answers, not as a per-question calibration signal.

**Not shown.** That other estimators, reward shapes, an explicit "I don't know" option, larger models or longer training would behave the same way.

## Limitations

- **Scale and seeds.** 1.5B for the sweep, two conditions at 3B, one shared A100 40GB. Most conditions have one training seed; S0, S3 and C2 have a second seed that agrees closely on MATH (accuracy 0.459 and 0.458, 0.406 and 0.401, 0.426 and 0.426). The intervals above cover evaluation noise, not training-seed variation.
- **Abstention definition.** "No boxed answer" conflates running out of tokens, formatting and refusal (Finding 2). A study of deliberate abstention needs an explicit abstain option and a larger token budget.
- **Probe comparison.** The matched-rate comparison covers 1663 MATH and 1033 GSM8K problems and uses probe scores from sampled answers.
- **TruthfulQA-MC.** Our original copy listed the correct option first; we re-evaluated with shuffled options and report only that re-run.

## Related work

- **Che, Yuan, Zhao and Yu (2026), arXiv 2608.00301**, the closest work: for the +1 / $-\lambda$ / 0 rule, GRPO's group normalization replaces the designed penalty with an effective penalty of one in the sparse-answer regime, moving the learned threshold to 1/2, with a formal analysis, experiments at 1.5B and 7B, and a repair. This project was done independently, before that paper appeared, and is empirical.
- **Liu et al. (2025), arXiv 2503.20783 (Dr. GRPO)**: dividing by the group standard deviation biases updates by question difficulty.
- **Zhu et al. (2025), arXiv 2506.01347**: negative-sample reinforcement; S3 matches their weighting.
- **Abstention and calibration rewards**: TruthRL (arXiv 2509.25760, ternary reward with z-scored GRPO), Reinforced Hesitation (arXiv 2511.11500, no division by the standard deviation), TIAR (arXiv 2605.25850, decoupled abstention advantage), RLCR (arXiv 2507.16806, Brier-scored verbalized confidence).
- **Confidence signals**: P(True) (Kadavath et al. 2022), semantic entropy (Kuhn et al. 2023; Farquhar et al. 2024), and hidden-state probes of correctness.

## Next steps

1. **An explicit abstain option and enough tokens.** Rerun a few points on the dial with an "I don't know" answer and a 2048-token budget, to separate deliberate abstention from formatting and truncation.
2. **Rewards that vary within an outcome class.** A verbalized confidence scored with a proper scoring rule (as in RLCR), or a probe-shaped abstain reward, can carry per-question information that a scalar triple cannot.
3. **Report targeting, not just rate.** Abstention rate alone hides whether abstention is useful. The repository includes an Inspect eval of the wise fraction (`evals/abstention/`) that runs on any model.

## Corrections

While preparing this write-up I audited every earlier number against the raw evaluation files (2026-10-18). Three earlier claims were withdrawn: a TruthfulQA-MC probe AUROC of 0.99 (the correct option was always listed first, so the probe read the chosen letter), a "shared correctness direction" across conditions (the underlying probe separated abstained from attempted answers, not right from wrong), and a single-shot utility advantage for the penalty conditions (a constructed metric without a single-pass confidence baseline). Details are in the repository's CHANGELOG.

## Acknowledgements

Sanjana Deshpande collaborated on the project.
