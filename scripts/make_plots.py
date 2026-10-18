"""Generate 8 project-report figures from existing analysis JSONs.

One-off helper for docs/PROJECT_REPORT.md. Reads:
  results/analysis_summary.json   (per-condition x per-benchmark metrics)
  results/trajectory_summary.json (per-problem stacking + difficulty axis)
  results/stacking_null.json      (empirical null comparison)

Writes 8 PNGs to docs/figures/. Pure matplotlib + numpy; no seaborn.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
EVAL = REPO / "results"
FIG = REPO / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

EV_ORDER = ["S0", "C1", "S1", "C2", "S2", "S3", "S4"]
CONDS = ["S0", "S1", "S2", "S3", "S4", "C1", "C2"]
BENCHES = ["math_test", "gsm8k_test", "aime_2024", "aime_2025", "mmlu_stem", "truthful_qa_mc"]
BENCH_LABEL = {
    "math_test": "MATH",
    "gsm8k_test": "GSM8K",
    "aime_2024": "AIME24",
    "aime_2025": "AIME25",
    "mmlu_stem": "MMLU-STEM",
    "truthful_qa_mc": "TruthfulQA",
}
COND_COLOR = {
    "S0": "#1f77b4",
    "S1": "#ff7f0e",
    "S2": "#2ca02c",
    "S3": "#d62728",
    "S4": "#9467bd",
    "C1": "#17becf",
    "C2": "#bcbd22",
}

A = json.loads((EVAL / "analysis_summary.json").read_text())
T = json.loads((EVAL / "trajectory_summary.json").read_text())
N = json.loads((EVAL / "stacking_null.json").read_text())
CI = json.loads((EVAL / "abstention_ci.json").read_text())["cells"]


def fig01_abstention_sweep():
    width = 0.11
    x = np.arange(len(BENCHES))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, c in enumerate(CONDS):
        cells = [CI[f"{c}/{b}"] for b in BENCHES]
        vals = [k["rate"] for k in cells]
        err = [[k["rate"] - k["ci_lo"] for k in cells], [k["ci_hi"] - k["rate"] for k in cells]]
        ax.bar(x + (i - 3) * width, vals, width, yerr=err, capsize=2, error_kw={"lw": 0.8},
               label=c, color=COND_COLOR[c])
    ax.set_xticks(x)
    ax.set_xticklabels([BENCH_LABEL[b] for b in BENCHES])
    ax.set_ylabel("Abstention rate (K=16 mean)\nbars: 95% bootstrap CI over problems")
    ax.set_title("Abstention rate per condition and benchmark (Qwen 2.5 1.5B, K=16)")
    ax.legend(ncol=7, loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig01_abstention_sweep.png", dpi=130)
    plt.close(fig)


def fig02_ece_corrected():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, bench in zip(axes, ["math_test", "gsm8k_test"]):
        old = [A["per_condition_bench"][c][bench]["ece_passatk_vs_any"] for c in CONDS]
        new = [A["per_condition_bench"][c][bench]["ece_modeconf_vs_modecorrect"] for c in CONDS]
        x = np.arange(len(CONDS))
        ax.bar(x - 0.2, old, 0.4, label="Leaky ECE (Pass@K vs correct_any)", color="#cc6666")
        ax.bar(x + 0.2, new, 0.4, label="Corrected ECE (mode-conf vs mode-correct)", color="#4477aa")
        ax.set_xticks(x)
        ax.set_xticklabels(CONDS)
        ax.set_title(BENCH_LABEL[bench])
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Expected Calibration Error")
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle("ECE: leaky metric vs corrected metric (pattern S0 -> S4 survives)")
    fig.tight_layout()
    fig.savefig(FIG / "fig02_ece_corrected.png", dpi=130)
    plt.close(fig)


def fig03_reliability_diagrams():
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for row, bench in enumerate(["math_test", "gsm8k_test"]):
        for col, cond in enumerate(["S0", "S4"]):
            ax = axes[row, col]
            bins = A["per_condition_bench"][cond][bench]["reliability_bins"]
            confs = [b["bin_conf_mean"] for b in bins]
            accs = [b["bin_accuracy"] for b in bins]
            counts = [b["count"] for b in bins]
            sizes = np.array(counts) / max(counts) * 350 + 20
            ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6, label="perfect calibration")
            ax.scatter(confs, accs, s=sizes, alpha=0.7,
                       color=COND_COLOR[cond], edgecolor="black", linewidth=0.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Mean mode-vote confidence in bin")
            ax.set_ylabel("Empirical accuracy in bin")
            ax.set_title(f"{cond} - {BENCH_LABEL[bench]}  "
                         f"(ECE = {A['per_condition_bench'][cond][bench]['ece_modeconf_vs_modecorrect']:.3f})")
            ax.grid(alpha=0.3)
            ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("Reliability diagrams (bubble area = bin count). "
                 "Points ABOVE the diagonal = under-confident.")
    fig.tight_layout()
    fig.savefig(FIG / "fig03_reliability_diagrams.png", dpi=130)
    plt.close(fig)


def fig04_auacc_curves():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    covs = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    for c in CONDS:
        sel = A["per_condition_bench"][c]["math_test"]["selective_accuracy"]
        ys = [sel[f"selacc_at_cov_{cov}"] for cov in covs]
        ax.plot(covs, ys, marker="o", label=c, color=COND_COLOR[c], linewidth=2)
    ax.set_xlabel("Coverage (fraction of problems ranked by mode-vote confidence)")
    ax.set_ylabel("Accuracy in covered subset")
    ax.set_title("Selective accuracy curves on MATH-test "
                 "(higher = better selective prediction)")
    ax.legend(ncol=4, fontsize=9, frameon=False, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(FIG / "fig04_auacc_curves.png", dpi=130)
    plt.close(fig)


def fig05_discrimination_heatmap():
    levels = ["1", "2", "3", "4", "5"]
    matrix = np.zeros((len(levels), len(CONDS)))
    for j, c in enumerate(CONDS):
        ml = A["math_by_level"][c]
        for i, lvl in enumerate(levels):
            matrix[i, j] = ml.get(lvl, {}).get("abstention_rate", 0.0)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(CONDS)))
    ax.set_xticklabels(CONDS)
    ax.set_yticks(np.arange(len(levels)))
    ax.set_yticklabels([f"L{l}" for l in levels])
    ax.set_xlabel("Condition")
    ax.set_ylabel("MATH difficulty level")
    for i in range(len(levels)):
        for j in range(len(CONDS)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="black" if matrix[i, j] < 0.5 else "white", fontsize=9)
    fig.colorbar(im, ax=ax, label="Abstention rate")
    ax.set_title("Discrimination collapse: L5/L1 abstention ratio drops from "
                 "5.4x (S0) to 1.2x (S4)")
    fig.tight_layout()
    fig.savefig(FIG / "fig05_discrimination_heatmap.png", dpi=130)
    plt.close(fig)


def fig06_stacking_null():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, bench in zip(axes, ["math_test", "gsm8k_test"]):
        d = N[bench]
        obs = d["observed_pct"] * 100
        nullA_m = d["null_independent_bernoulli"]["mean_pct"] * 100
        nullA_lo = d["null_independent_bernoulli"]["ci_2_5_pct"] * 100
        nullA_hi = d["null_independent_bernoulli"]["ci_97_5_pct"] * 100
        nullB_m = d["null_column_shuffle"]["mean_pct"] * 100
        nullB_lo = d["null_column_shuffle"]["ci_2_5_pct"] * 100
        nullB_hi = d["null_column_shuffle"]["ci_97_5_pct"] * 100
        zA = d["null_independent_bernoulli"]["z_ratio_obs_vs_null"]
        zB = d["null_column_shuffle"]["z_ratio_obs_vs_null"]
        labels = ["Naive\n(uniform-random)", "Null A\n(indep. Bernoulli)",
                  "Null B\n(column-shuffle)", "Observed"]
        values = [6.25, nullA_m, nullB_m, obs]
        errs_lo = [0, nullA_m - nullA_lo, nullB_m - nullB_lo, 0]
        errs_hi = [0, nullA_hi - nullA_m, nullB_hi - nullB_m, 0]
        colors = ["#999999", "#88aaee", "#4477aa", "#cc4444"]
        x = np.arange(len(labels))
        ax.bar(x, values, color=colors,
               yerr=[errs_lo, errs_hi], capsize=6, edgecolor="black", linewidth=0.6)
        for xi, v in zip(x, values):
            ax.text(xi, v + 2, f"{v:.1f}%", ha="center", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f"{BENCH_LABEL[bench]} "
                     f"(z = {zA:.0f} sigma A, {zB:.0f} sigma B)")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("% of problems with stacked abstention pattern")
    fig.suptitle("Per-problem stacking observed vs empirical nulls "
                 "(observed is 25-98 sigma above marginal-preserving null)")
    fig.tight_layout()
    fig.savefig(FIG / "fig06_stacking_null.png", dpi=130)
    plt.close(fig)


def fig07_jaccard():
    j = A["jaccard_abstention"]["math_test"]["jaccard"]
    conds = EV_ORDER
    matrix = np.array([[j[a][b] for b in conds] for a in conds])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(conds)))
    ax.set_yticks(np.arange(len(conds)))
    ax.set_xticklabels(conds)
    ax.set_yticklabels(conds)
    for i in range(len(conds)):
        for k in range(len(conds)):
            ax.text(k, i, f"{matrix[i, k]:.2f}", ha="center", va="center",
                    color="black" if matrix[i, k] < 0.6 else "white", fontsize=9)
    fig.colorbar(im, ax=ax, label="Jaccard similarity")
    ax.set_title("Per-problem majority-abstention agreement (MATH-test)\n"
                 "S2-C2 = 0.70 (matched-threshold conditions agree)")
    fig.tight_layout()
    fig.savefig(FIG / "fig07_jaccard.png", dpi=130)
    plt.close(fig)


def fig08_subject_ordering():
    subs = T["math_test"]["difficulty_axis"]["by_subject_sorted_easiest_to_hardest"]
    labels = [s["subject"] for s in subs]
    means = [s["mean_abst_count"] for s in subs]
    counts = [s["n"] for s in subs]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, means, color="#4477aa", edgecolor="black", linewidth=0.5)
    for xi, m, n in zip(x, means, counts):
        ax.text(xi, m + 0.05, f"{m:.2f}\n(n={n})", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean abstain-count across 7 conditions")
    ax.set_title("Recovered difficulty axis on MATH subjects (Spearman rho = +0.234, "
                 "p < 1e-62). Ordering matches human intuition: "
                 "Prealgebra < ... < Precalculus")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(means) * 1.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig08_subject_ordering.png", dpi=130)
    plt.close(fig)


def main() -> int:
    fig01_abstention_sweep()
    fig02_ece_corrected()
    fig03_reliability_diagrams()
    fig04_auacc_curves()
    fig05_discrimination_heatmap()
    fig06_stacking_null()
    fig07_jaccard()
    fig08_subject_ordering()
    print(f"OK 8 figures -> {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
