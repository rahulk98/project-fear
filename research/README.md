# Supporting research

This directory holds the deeper analysis behind the headline results in
[../RESULTS.md](../RESULTS.md). It is the evidence trail for Finding O, the
advantage-normalization ablations, the single-pass probe and its compute-crossover law, and
the cross-domain transfer test.

## Documents

- `FINAL_VERDICT.md` - the adversarially-refuted synthesis. Four follow-up experiments were
  written, run on the cached rollouts, each refuted, then combined. All numbers recomputed
  from raw files. Contains the matched-population, matched-compute, bootstrap-CI version of
  the probe-versus-self-consistency comparison (the reviewer-proof K* fix).
- `FINDINGS.md` - where punishment-oriented training actually wins: single-shot expected
  utility under asymmetric error cost, with per-benchmark cost crossovers.
- `INSIGHTS.md` - the long-form running insights log behind the 23-finding catalogue.
- `RESULTS_THREE_PIVOTS.md`, `PIVOT1_RESULTS.md`, `PIVOT3_RESULTS.md` - results from the
  exploratory pivots evaluated on the cached rollouts.
- `DECISIONS.md` - the decisions made while turning the negative result into a defensible
  contribution.

## Scripts

These run offline on cached rollouts (no training, mostly no GPU):

- `cost_sensitive_utility.py` - the asymmetric-cost deployment-utility analysis.
- `cross_domain_probe.py`, `matched_population.py`, `fear_domain_split.py` - the
  cross-domain probe transfer and the matched-population K* recomputation.
- `probe_gate.py`, `probe_bestofn.py`, `probe_adaptive_compute.py`,
  `extract_probe_states.py`, `dump_probe_scores.py` - the linear-probe selective-prediction
  study.
- `sc_selectivity.py`, `plot_selectivity.py` - self-consistency selectivity curves and the
  compute-crossover plot.

## Figures

`figures/` holds the single-shot utility, per-sample utility, cost-crossover, and
selectivity-versus-compute plots referenced in `FINDINGS.md` and `FINAL_VERDICT.md`.
