# Changelog

## 2026-10-18
- Added research/POST.md, the write-up of the abstention-dial result.
- Audited every public number against the raw evaluation files. Withdrawn as untraceable: 3B FD1 results, the TruthfulQA-MC probe AUROC of 0.990, the probe compute crossover (K*) and cross-domain probe numbers. Corrected: 3B GSM8K abstention ratio, PRM800K ordering at temperature 1.0, fig01 title.
- Marked the single-shot utility result as retracted in RESULTS.md and research/FINDINGS.md.
- Reworded the novelty claim to credit concurrent work (Che et al. 2026, arXiv 2608.00301).
- Figures reproducible with `python scripts/make_plots.py`; requirements pinned.
- Follow-up E4: 95% bootstrap error bars on fig01.
- Follow-up E2: wise versus wasteful abstention split (RESULTS.md section 9, POST.md section 7).
- Follow-up E3: P(True) and semantic-entropy baselines on the probe population (RESULTS.md section 9).
- Follow-up E1: probe re-check with question-grouped CV, nested layer selection, shuffled-label and text-only controls. Probe vs self-consistency (K*) and MATH-to-GSM8K/MMLU-STEM transfer reproduce and are restored.
- Retracted: TruthfulQA-MC probe AUROC 0.990. The correct option is always listed first in the project's TruthfulQA-MC data, so the probe reads the chosen letter; all TruthfulQA-MC accuracy numbers are flagged.
- Corrected: the cross-condition probe transfer (0.989 / 0.992) was an abstained-vs-attempted probe, not a correctness probe; the "shared correctness direction" claim is withdrawn from README, RESULTS.md and the post.
- Follow-up Q3: offline advantage analysis explaining the falling wise fraction; simple per-question abstain rewards did not help offline. Post mechanism wording made precise for t = 0 and t = 1. Post links the invariance test.
