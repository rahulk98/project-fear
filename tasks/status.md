# status

## RESUME (2026-10-18, CEST)
- Running: follow-up batch on ldvpc84 (deeplearning.cl.uni-trier.de, account s2sadesh), workdir ~/fear_followups_20261018/{e1,e2,e3,e4}. Three Sonnet agents: e1-probe-recheck (CPU), e2-wise-wasteful (E2 + E4, CPU), e3-ptrue-se (one idle 2080 Ti, est. < 1 GPU-h).
- Constraints: /home on ldvpc84 is 100% full (3.9 GB free, shared); outputs < 100 MB, no caches. One GPU job only, never GPU 0. Never modify ~/master-thesis-asymmetric-training there.
- Results land in results/followups/*.json locally. Verify each before touching RESULTS.md, POST.md, project page (via website-builder) or CHANGELOG.md.
- Next: verify results, decide whether TruthfulQA 0.990 and K* can return; add error bars to fig01; update RESULTS.md + changelog + page.
- Dropped by Rahul (2026-10-18): extra seeds and 3B ablation (no new training).
