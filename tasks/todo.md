# todo

## Pending

### Part A: post + repo (today, 2026-10-18)
- [ ] A2 Number audit: every number in POST.md traced to RESULTS.md / research/*.md (raw JSONs are not in the repo; provenance is doc-level only)
- [ ] A3 Draft research/POST.md (1500-2500 words); figures from docs/figures/
- [ ] A4 Repo polish: README links to post; fix retracted section 6 in RESULTS.md; pull TruthfulQA 0.990 from README/RESULTS until checked; replace "no published paper stated this" claim; one-command figure repro; pin requirements; CITATION.cff
- [ ] A5 3-sentence application blurb
- [ ] A6 Commit per step; push after check-in

### Part B: website (handed to website-builder session)
- [ ] B1 Send POST.md + figures + brief to website-builder once A3/A4 are done

### Part C: follow-up experiments (WAIT for Rahul's go before starting any)
- [ ] C1 TruthfulQA-MC probe AUROC 0.990 sanity check (GroupKFold, shuffled labels, text-only baseline, n, bootstrap CI)
- [ ] C2 Wise vs wasteful abstention split on cached rollouts
- [ ] C3 P(True) + semantic entropy at matched compute, same questions as probe
- [ ] C4 Missing seeds for main 1.5B conditions (>=3), error bars on dial plot
- [ ] C5 3B no-normalization ablation, more seeds (only if compute allows)
- Note: raw result JSONs and rollouts are not in this checkout; C1-C3 need the cached data location (university server?)

## Done
- [x] A1 Novelty check (2026-10-18): Che et al. arXiv 2608.00301 (31 Jul 2026) already derives GRPO penalty washout (lambda_eff = 1, threshold 1/2) in the sparse-answer regime; affine invariance of z-scoring is textbook. Claim reworded, see POST.md related work.
