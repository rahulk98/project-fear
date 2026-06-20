#!/usr/bin/env python3
"""Figure: selective-prediction AUROC vs inference compute (samples) for S0.

Contrasts the single forward-pass probe (idea #2) and the free logit confidence
against K-sample self-consistency (the K*(c) crossover of idea #1). Shows one probe
pass ~ K=3-4 self-consistency, and the free logit signal is near-useless.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT, FIG = os.path.join(ROOT, "outputs"), os.path.join(ROOT, "figures")
sc = json.load(open(os.path.join(OUT, "sc_selectivity.json")))
pr = json.load(open(os.path.join(OUT, "probe_auroc.json")))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, bench in zip(axes, ["math_test", "gsm8k_test"]):
    Ks = [2, 4, 8, 16]
    sc_auroc = [sc[bench][f"K{k}"]["sc_mode_conf_auroc"] for k in Ks]
    ax.plot(Ks, sc_auroc, "o-", color="#1f77b4", lw=2, label="self-consistency (K samples)")
    probe = pr["S0"][bench]["probe"]
    logit = pr["S0"][bench]["logit_maxprob"]
    ax.scatter([1], [probe], color="red", zorder=5, s=70,
               label=f"probe, 1 pass (AUROC {probe:.3f})")
    ax.axhline(probe, color="red", ls="--", lw=1, alpha=0.6)
    ax.scatter([1], [logit], color="gray", zorder=5, s=70, marker="x",
               label=f"free logit conf, 1 pass ({logit:.3f})")
    # mark where SC crosses the probe line
    cross = None
    for i in range(1, len(Ks)):
        if sc_auroc[i - 1] < probe <= sc_auroc[i]:
            cross = Ks[i - 1] + (Ks[i] - Ks[i - 1]) * (probe - sc_auroc[i - 1]) / (sc_auroc[i] - sc_auroc[i - 1])
    if cross:
        ax.axvline(cross, color="green", ls=":", lw=1.2)
        ax.text(cross + 0.2, 0.66, f"K*~{cross:.1f}\n(probe = SC here)", color="green", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels([1, 2, 4, 8, 16])
    ax.set_xlabel("inference compute (samples per problem)")
    ax.set_ylabel("selective-prediction AUROC (correct vs incorrect)")
    ax.set_title(f"One probe pass ~ K=3-4 self-consistency  [{bench}, S0]")
    ax.set_ylim(0.55, 0.95); ax.legend(fontsize=8, loc="lower right")
fig.suptitle("Probe-gated single-shot selection (idea #2) vs sampling budget K (idea #1 K* crossover)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_selectivity_vs_compute.png"), dpi=130)
print("wrote", os.path.join(FIG, "fig4_selectivity_vs_compute.png"))
