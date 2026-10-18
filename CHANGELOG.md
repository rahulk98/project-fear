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
