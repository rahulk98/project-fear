# todo

## Pending

### Part A: post + repo
- [ ] A7 Rahul: review research/POST.md, publish on LessWrong, send URL (then add to README + project page)

### Part B: website (handed to website-builder session, 2026-10-18)
- [ ] B1 Project page grpo-abstention-dial/ built and checked locally by website-builder (1280px, 390px); awaiting Rahul's push approval in that session

### Part C: follow-up experiments (WAIT for Rahul's go before starting any)
Raw data: deeplearning.cl.uni-trier.de (ldvpc84, account s2sadesh) ~/master-thesis-asymmetric-training/outputs (eval rollouts, probing npz). Shared box, other jobs running there.
- [ ] C4 Missing seeds for main 1.5B conditions (>=3), error bars on dial plot. Existing: S0/S3/C2 seeds 42+123; others seed 42 only.
- [ ] C5 3B no-normalization ablation, more seeds (only if compute allows)

- [ ] D1 Workshop paper plan (Rahul's go needed): one fix that can carry per-question confidence (proper-scoring confidence reward or decoupled abstention advantage), wise-fraction metric, 3 seeds; ~40-70 GPU-h on A100/rented GPU
- [ ] D2 RCoT CoT-faithfulness project: project page + post in the same format

- [ ] Q1 TruthfulQA-MC re-eval with shuffled options (9 models, K=16) + probe re-check: GPU run on ldvpc84 GPU 1, agent tqa-finish analysing; ETA ~14:30 CEST

## Done
- [x] D3 Regenerated summary JSONs and fig02-fig08 from final rollouts (2026-10-18). Commit 6bbdc8d
- [x] Q2 Inspect eval evals/abstention/ (10 unit tests pass) (2026-10-18). Commit 370404a
- [x] Q3 Offline advantage analysis (2026-10-18). Commit 59f566a
- [x] Q4 t-slider on project page (website-builder, local; push with page)
- [x] Post trimmed to 2604 words excl. tables (2026-10-18). Commit 7298f57
- [x] C1/C6 E1 probe re-check (2026-10-18): TruthfulQA 0.990 retracted (gold always option A); K* and MATH->GSM8K/MMLU transfer reproduce; cross-condition "correctness" transfer was an abstention probe, claim withdrawn. Commit 20385c0
- [x] C2 E2 wise vs wasteful (2026-10-18): wise fraction falls with t on MATH, random on GSM8K. Commit 528ee4b
- [x] C3 E3 P(True) near chance, semantic entropy K=4 0.81 MATH (2026-10-18). Commit 1ce907a
- [x] E4 fig01 bootstrap error bars (2026-10-18). Commit 528ee4b
- [x] A1 Novelty check: Che et al. arXiv 2608.00301 (31 Jul 2026) derives GRPO penalty washout formally; affine invariance is textbook. Framed as independent empirical work.
- [x] A2 Number audit vs raw files on ldvpc84. Untraceable and removed: 3B FD1, TruthfulQA 0.990, probe K*/transfer (JSONs never saved). Fixed: 3B GSM8K ratio ~7x not 2x, PRM800K order breaks at T=1.0, fig01 title.
- [x] A3 research/POST.md drafted (2396 words of prose), commit 17d53a8
- [x] A4 Repo polish: figures reproducible (python scripts/make_plots.py), requirements pinned from asym-thesis env freeze, README/RESULTS/CITATION corrected, commits 1849df1, 56e1e69, fd8d77b; pushed
- [x] A5 Application blurb (in chat 2026-10-18)
