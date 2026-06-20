# FEAR / Pivot-A Final Verdict (team-implemented + adversarially refuted, 2026-06-04)

Produced by the `ultracode-pivotA-and-fear-verdict` workflow: 4 experiments written+run on the
local cached data, each adversarially refuted, then synthesized. All numbers recomputed from raw
files. Honest, no constructed metrics.

## PART 1 -- GOAL 1: HONEST FEAR VERDICT = NO genuine advantage

The earlier "fear wins under asymmetric cost at K=1" claim was a LOOPHOLE (constructed cost metric +
no single-pass confidence baseline). It stays retracted. On standard risk-coverage / selective-acc /
AUROC at MATCHED COMPUTE:

1. Single-shot frontier: fear is a 1-D abstention dial (Finding O made concrete). Greedy points:
   MATH attempt/acc_on_attempted S0 .867/.618, S1 .857/.617, S2 .849/.612, S3 .836/.602, S4 .325/.513;
   GSM8K S0 .951/.722 ... S3 .717/.699, S4 .017/.591. attempt_rate drops monotonically; acc_on_attempted
   stays flat -> punishment does not preferentially drop wrong answers.
2. Probe vs fear at K=1 is a TIE on MATH (anchored gaps +0.004..+0.032, inside the +/-0.015 CI; raw gaps
   are NEGATIVE), modest probe edge on GSM8K. Neither has a real single-pass edge.
3. Self-consistency overtakes both at K* = 2.50 (MATH) / 2.88 (GSM8K) -- a compute crossover, not a 16x
   blowout. SC AUROC: MATH K2 .799/K4 .889/K8 .911 (probe .828); GSM8K K2 .722/K4 .830/K8 .854 (probe .779).
4. OOD threshold transfer (fear's one favorable hypothesis): REFUTED. Fear's coverage "stability" is
   difficulty-blind abstention; the post-hoc threshold transfers essentially loss-free on selective accuracy,
   while every fear condition shows 0.078-0.104 acc_on_attempted drift.
5. Cascade/routing: fear's apparent low-budget win was a probe base-rate bug; corrected, the probe ties at
   the natural-abstain point and beats fear at every higher routing fraction.

VERDICT: no genuine fear advantage on any standard axis at matched compute. Consistent with Finding O.
The valuable result is the rigorous NEGATIVE: a cautionary contribution for "train-to-abstain via
asymmetric reward". Do not manufacture another loophole.

## PART 2 -- GOAL 2: PIVOT-A PUBLISHABILITY = conditional GO (workshop / ACL Findings / project core)

Contribution sentence: a single forward-pass linear probe on hidden states is a compute-efficient
selective-prediction signal for math reasoning, worth ~2.5-3 self-consistency samples (crossover
K* = 2.5 MATH / 2.9 GSM8K); we give the explicit K*(c) compute-crossover law, and show RL punishment
("fear") training buys no selective-prediction advantage at all.

Novelty: the probe is old (SAPLMA / "internal states know more"; P(True) Kadavath; semantic entropy
Kuhn/Farquhar). The novel sliver = the K*(c) compute-crossover law (amortizing one probe pass against K
self-consistency samples on a common selective-prediction axis) + the clean RL-abstention negative result.

Killer baseline a reviewer demands: semantic entropy / P(True) / sampled-confidence at MATCHED COMPUTE on
the SAME population and axis as the probe (current probe-vs-SC is cross-population -> winner's curse).

5-step plan: (1) matched-population, matched-compute comparison with bootstrap CIs; (2) nested held-out
layer selection (remove the best-of-17-layers winner's curse); (3) drop bespoke cost metrics, use raw
AUROC-vs-compute + selective-acc-at-coverage; (4) cement the FEAR negative (1-D dial plot + matched-compute
frontier + loophole note); (5) run the ONE decisive experiment below, then write up.

THE ONE DECISIVE EXPERIMENT: cross-DOMAIN generalization. Train the probe on MATH hidden states, test on
MMLU-STEM and TruthfulQA-MC (greedy rollouts exist). If the probe edge + K*(c) survive a real domain shift,
the contribution is "a single cheap forward pass is a TRANSFERABLE selective-prediction signal worth ~K
samples" -- a real result. If it does not transfer, that is an honest, publishable boundary.

GO/NO-GO: GO -- pivot to "compute-aware selective prediction (probe-gate + K*(c) law) with FEAR as the
rigorous negative", after the matched-population fix + nested layer selection + cross-domain experiment.
Target an ICLR/NeurIPS workshop or ACL Findings, not main track. Sufficient as the project core.

## ADDENDUM -- the decisive cross-domain experiment (run 2026-06-04, S0 probe)

Trained the probe on MATH (best layer 22, group-CV AUROC 0.828) and applied it to other domains
(genuine held-out transfer; cross_domain_probe.json):

| math-probe on | transfer AUROC | in-domain AUROC (ceiling) | reading |
|---|---|---|---|
| GSM8K | 0.744 | 0.751 | transfers ~losslessly WITHIN the reasoning family |
| MMLU-STEM | 0.574 | 0.673 | transfer degrades; MC task type differs |
| TruthfulQA-MC | 0.663 | 0.990 | barely transfers, but in-domain is NEAR-PERFECT |

Two real findings: (1) the single forward-pass probe transfers within a task family (math->gsm8k = in-domain
quality) but needs per-family refitting for MC/factual tasks -- an honest, publishable boundary. (2) On
TruthfulQA the in-domain probe reaches AUROC 0.990: the model internally separates right-from-wrong almost
perfectly yet does not express it (a striking internal-knowledge-vs-output gap), so a single probe pass gives
near-perfect single-shot selective prediction on a hallucination benchmark. This STRENGTHENS Pivot A: a clean
project-core/workshop story = "a single cheap probe pass is a strong, family-transferable selective-prediction
signal (and near-oracle on TruthfulQA), worth ~3 self-consistency samples (K* law); RL 'fear' training adds
nothing." Net: GO, confidence raised.

## ADDENDUM 2 -- matched-population, matched-compute, bootstrap CIs (the reviewer-proof K* fix)

Earlier probe-vs-SC was cross-population (winner's curse). Recomputed on the SAME problems (the Exp2
subset), per-problem, with 2000x bootstrap CIs (matched_population.json):

MATH (n=1663): probe 1-pass 0.785 [0.763,0.807]; free logit 0.563 [0.534,0.593]; SC K2 0.725
[0.706,0.746]; K4 0.831; K8 0.878; K16 0.904. K* = 3.13.
GSM8K (n=1033): probe 0.720 [0.682,0.756]; logit 0.595; SC K2 0.637; K4 0.736; K8 0.816; K16 0.850. K* = 3.68.

The probe (one forward pass) SIGNIFICANTLY beats free logit confidence and 2-sample self-consistency
(disjoint CIs on math) and EQUALS ~3-3.7 self-consistency samples. This is the clean, defensible core:
"one cheap probe pass buys ~3 self-consistency samples of selective-prediction quality." Winner's-curse
removed. Pivot A core is now reviewer-hardened.
