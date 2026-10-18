# todo

## Pending

### Part A: post + repo
- [ ] A7 Rahul: review research/POST.md, publish on LessWrong, send URL (then add to README + project page)

### Part B: website (handed to website-builder session, 2026-10-18)
- [ ] B1 Project page grpo-abstention-dial/ in rahulk98.github.io; waiting on website-builder report

### Part C: follow-up experiments (WAIT for Rahul's go before starting any)
Raw data: deeplearning.cl.uni-trier.de (ldvpc84, account s2sadesh) ~/master-thesis-asymmetric-training/outputs (eval rollouts, probing npz). Shared box, other jobs running there.
- [ ] C1 TruthfulQA-MC probe AUROC 0.990 sanity check (GroupKFold, shuffled labels, text-only baseline, n, bootstrap CI). Data: outputs/probing/qwen2_5_1_5b_S0_float32_truthful_qa_mc_hidden.npz, 1600 committed rows, 598 unique questions. Currently withdrawn from all public pages.
- [ ] C2 Wise vs wasteful abstention split on cached rollouts
- [ ] C3 P(True) + semantic entropy at matched compute, same questions as probe
- [ ] C4 Missing seeds for main 1.5B conditions (>=3), error bars on dial plot. Existing: S0/S3/C2 seeds 42+123; others seed 42 only.
- [ ] C5 3B no-normalization ablation, more seeds (only if compute allows)
- [ ] C6 Re-run lost analyses (matched_population, cross_domain_probe, sc_selectivity) so the probe K* numbers can return

## Done
- [x] A1 Novelty check: Che et al. arXiv 2608.00301 (31 Jul 2026) derives GRPO penalty washout formally; affine invariance is textbook. Framed as independent empirical work.
- [x] A2 Number audit vs raw files on ldvpc84. Untraceable and removed: 3B FD1, TruthfulQA 0.990, probe K*/transfer (JSONs never saved). Fixed: 3B GSM8K ratio ~7x not 2x, PRM800K order breaks at T=1.0, fig01 title.
- [x] A3 research/POST.md drafted (2396 words of prose), commit 17d53a8
- [x] A4 Repo polish: figures reproducible (python scripts/make_plots.py), requirements pinned from asym-thesis env freeze, README/RESULTS/CITATION corrected, commits 1849df1, 56e1e69, fd8d77b; pushed
- [x] A5 Application blurb (in chat 2026-10-18)
