# Control panel: where both goals stand + the decisions you need to make (2026-06-04)

Everything below is in research/ (isolated; live pipeline untouched). Two decisions gate further work.

## GOAL 1 -- "is FEAR/punishment better?" -> HONEST ANSWER: no genuine win (from all available data)

Experiments run (all adversarially checked), fear verdict:
| experiment | file | fear wins? |
|---|---|---|
| single-shot cost-utility (the original) | FINDINGS.md | LOOPHOLE (retracted) |
| single-shot risk-coverage frontier (matched compute) | FINAL_VERDICT.md | no |
| probe vs fear at K=1 | FINAL_VERDICT.md | no (tie; probe not fear) |
| K*(c) compute law, matched-population + CIs | matched_population.json | no |
| OOD threshold transfer | FINAL_VERDICT.md | no (refuted) |
| cascade / routing | FINAL_VERDICT.md | no |
| cross-domain probe transfer | cross_domain_probe.json | n/a (Pivot-A evidence) |
| live evals: S0_zhu_signed, SC | GOAL1_LIVE_TRACKING.md | no (P0/no-punishment best) |
| VOI-GRPO offline pre-check | voi_precheck.json | no escape (collapses in unanimous groups) |

Live watch (ongoing): Orion queue still producing VC/VC_flat/VC0/CWD1; their evals are PENDING (post-train
monitor died on reboot). I am polling each loop tick.

The ONLY remaining avenue to a possible fear-win is to TRAIN a smarter-reward condition (not derivable from
cached data): probe-in-the-loop RL (shape GRPO reward with the probe's per-sample correctness signal) or
calibration-reward RL (proper-scoring-rule on a verbalized confidence). Each needs ~9h on one GPU + a reward.py
change (grpo-reward-reviewer + codex gate). Orion is busy ~4 more days; could run the no-vLLM path on a 2080 Ti.

>>> DECISION 1: (a) accept the Goal-1 negative as the deliverable, or (b) greenlight me to train the
    probe-in-the-loop RL condition as the last shot at a fear-win. <<<

## GOAL 2 -- pivot: 30 vetted ideas, 3 finalists

Idea sets: PIVOT_OPTIONS.md (10 anchored, no-GPU), PIVOT_OPTIONS_FRESH.md (10 unconstrained no-GPU),
PIVOT_OPTIONS_GPU_RELAXED.md (10 trainable). Finalists:

1. Abstaining Verifier (GPU-relaxed #1, NOVEL 7.5) -- 3-way PRM {correct, incorrect, unverifiable} with
   MC-variance self-supervision + asymmetric cost (reuses FEAR code). EMNLP/ACL-main path. Zero-training
   first de-risk. Natural evolution of the abstention work into verification. RECOMMENDED highest ceiling.
2. Social-Choice Theory of self-consistency (fresh #1, NOVEL 7.5) -- impossibility theorem + clone-robust
   Condorcet rule. Pure theory + CPU, no GPU. NeurIPS/ICLR-theory path. Biggest profile swing, highest risk.
3. Pivot A: compute-aware selective prediction (probe-gate + K*(c) law) -- already de-risked end-to-end
   (matched-population K*, cross-domain transfer, TruthfulQA in-domain 0.99). Safest, fastest to a paper.
  **[Retracted 2026-10-18]** The TruthfulQA-MC in-domain AUROC (0.990) is an artifact: in this data the correct option is always listed first, so the probe reads which letter the model chose. The MATH-to-TruthfulQA transfer value is withdrawn with it. See RESULTS.md section 5.

>>> DECISION 2: pick 1, 2, or 3 (or another from the files). On your pick I will: scaffold the repo, run the
    first de-risk experiment, add the killer baselines, and draft the paper outline. <<<

## What I will NOT do unprompted
- Launch multi-hour training or fire the project's heavy P1/calibration evals on the shared, currently-busy
  GPUs without your go (resource + interference risk). Say the word and I will.
