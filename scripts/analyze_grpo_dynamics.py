"""Pull per-step GRPO training dynamics from WandB and verify the binary-group
equivalence prediction.

Theory: under GRPO z-score normalization, all 7 reward conditions (S0-S4, C1-C2)
produce identical gradients in groups that contain only two outcomes (e.g.,
correct + wrong with zero abstentions). They diverge only when a third outcome
(abstention) enters the group. Predictions:

  1. Early training: nearly identical reward_mean / reward_std / loss / kl
     trajectories across all conditions.
  2. As model starts abstaining: trajectories diverge in the order of
     "abstain advantage" magnitude (S0 < S1 < S2 < S3 < S4 ; C1 < C2).
  3. S4 (alpha=0) hits reward_std collapse soonest (more all-abstain groups).

This script downloads each condition's WandB run history, writes one CSV per
run to outputs/analysis/grpo_dynamics/, and produces a 4-panel comparison plot
per seed plus a JSON summary of divergence statistics.

Usage:
    python scripts/analyze_grpo_dynamics.py [--model qwen2_5_1_5b] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless server
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import wandb  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "outputs" / "analysis" / "grpo_dynamics"

ENTITY = "ndsanjana-student"
PROJECT = "fear-project"

CONDITIONS = ["S0", "S1", "S2", "S3", "S4", "C1", "C2"]
CONDITION_COLORS = {
    "S0": "#1f77b4", "S1": "#ff7f0e", "S2": "#2ca02c", "S3": "#d62728",
    "S4": "#9467bd", "C1": "#8c564b", "C2": "#e377c2",
}

# TRL 0.15.2 GRPOTrainer logs under train/ prefix; keep fallbacks for older versions.
METRIC_KEYS = {
    "reward_mean":      ["train/reward", "reward"],
    "reward_std":       ["train/reward_std", "reward_std"],
    "kl":               ["train/kl", "kl"],
    "loss":             ["train/loss", "loss"],
    "completion_length":["train/completion_length", "completion_length"],
    "grad_norm":        ["train/grad_norm", "grad_norm"],
}


def find_first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns and df[c].notna().any():
            return c
    return None


def normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce raw wandb history to a clean per-step frame with standard column names.
    Drops the final save-event row (which has all train/ metrics NaN)."""
    step_col = "train/global_step" if "train/global_step" in df.columns else "_step"
    out = pd.DataFrame({"step": df[step_col]})
    for canon, candidates in METRIC_KEYS.items():
        col = find_first_present(df, candidates)
        if col is not None:
            out[canon] = df[col].values
    metric_cols = [c for c in out.columns if c != "step"]
    if metric_cols:
        out = out.dropna(subset=metric_cols, how="all")
    out = out.dropna(subset=["step"]).sort_values("step").reset_index(drop=True)
    return out


def fetch_run_histories(model: str, seed: int) -> dict[str, pd.DataFrame]:
    """Return {condition: history_df} for the given model+seed across all 7 conditions."""
    api = wandb.Api(timeout=60)
    pattern = re.compile(rf"^{re.escape(model)}_(S[0-4]|C[12])_s{seed}$")

    print(f"Querying {ENTITY}/{PROJECT} for runs matching {pattern.pattern} ...")
    runs = api.runs(f"{ENTITY}/{PROJECT}", per_page=200)

    by_cond: dict[str, list] = {}
    for r in runs:
        m = pattern.match(r.name or "")
        if not m:
            continue
        by_cond.setdefault(m.group(1), []).append(r)

    histories: dict[str, pd.DataFrame] = {}
    for cond in CONDITIONS:
        matches = by_cond.get(cond, [])
        if not matches:
            print(f"  {cond}: no run found")
            continue
        # Selection priority: finished > most-recent. Among finished runs, prefer
        # the one with the most logged steps (longest training run).
        finished = [r for r in matches if r.state == "finished"]
        if finished:
            chosen = max(finished, key=lambda r: r.summary.get("train/global_step", 0))
        else:
            chosen = sorted(matches, key=lambda r: r.created_at, reverse=True)[0]
        steps = chosen.summary.get("train/global_step", "?")
        print(f"  {cond}: {chosen.name} ({chosen.id}, state={chosen.state}, steps={steps})")
        hist_raw = chosen.history(samples=10000, pandas=True)
        histories[cond] = normalize_history(hist_raw)
    return histories


def save_per_run_csvs(histories: dict[str, pd.DataFrame], model: str, seed: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for cond, df in histories.items():
        path = OUT_DIR / f"{model}_{cond}_s{seed}.csv"
        df.to_csv(path, index=False)
    print(f"Saved {len(histories)} per-run CSVs to {OUT_DIR}/")


def divergence_summary(
    histories: dict[str, pd.DataFrame],
    std_floor: float = 1e-4,
    divergence_threshold: float = 0.05,
) -> dict:
    """Quantitative summary: when did each condition diverge from S0, and how
    often did reward_std collapse below the std_floor."""
    if "S0" not in histories:
        return {"error": "S0 reference run missing"}
    s0 = histories["S0"]
    if "reward_mean" not in s0.columns:
        return {"error": "reward_mean not logged in S0 run"}

    summary: dict = {}
    for cond, df in histories.items():
        entry: dict = {"n_steps_logged": int(len(df))}
        if "reward_std" in df.columns:
            n_collapse = int((df["reward_std"] < std_floor).sum())
            first_collapse = df.loc[df["reward_std"] < std_floor, "step"]
            entry["std_collapse_step_count"] = n_collapse
            entry["std_collapse_fraction"] = round(n_collapse / max(1, len(df)), 4)
            entry["std_first_collapse_step"] = int(first_collapse.iloc[0]) if len(first_collapse) else None
        if "reward_mean" in df.columns and cond != "S0":
            joined = df[["step", "reward_mean"]].merge(
                s0[["step", "reward_mean"]].rename(columns={"reward_mean": "reward_mean_s0"}),
                on="step", how="inner",
            )
            joined["abs_delta"] = (joined["reward_mean"] - joined["reward_mean_s0"]).abs()
            diverge = joined.loc[joined["abs_delta"] > divergence_threshold, "step"]
            entry["divergence_step_vs_S0"] = int(diverge.iloc[0]) if len(diverge) else None
            entry["final_abs_delta_vs_S0"] = round(float(joined["abs_delta"].iloc[-1]), 4) if len(joined) else None
        if "reward_mean" in df.columns:
            nonnan = df["reward_mean"].dropna()
            if len(nonnan):
                entry["final_reward_mean"] = round(float(nonnan.iloc[-1]), 4)
                entry["mean_reward_mean"] = round(float(nonnan.mean()), 4)
        if "completion_length" in df.columns:
            nonnan = df["completion_length"].dropna()
            if len(nonnan):
                entry["final_completion_length"] = round(float(nonnan.iloc[-1]), 2)
                entry["initial_completion_length"] = round(float(nonnan.iloc[0]), 2)
        summary[cond] = entry
    return summary


def plot_dynamics(histories: dict[str, pd.DataFrame], model: str, seed: int) -> None:
    panels = [
        ("reward_mean", "Group reward mean", "reward"),
        ("reward_std", "Group reward std (log)", "reward_std"),
        ("completion_length", "Mean completion length (tokens)", "tokens"),
        ("kl", "KL divergence vs reference policy", "KL"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()
    has_any = False
    for ax, (metric, title, ylabel) in zip(axes, panels):
        for cond in CONDITIONS:
            if cond not in histories:
                continue
            df = histories[cond]
            if metric not in df.columns:
                continue
            ax.plot(df["step"], df[metric], label=cond,
                    color=CONDITION_COLORS[cond], linewidth=1.2, alpha=0.85)
            has_any = True
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if metric == "reward_std":
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.axhline(1e-4, color="black", linestyle=":", linewidth=0.8, label="std_floor")
        ax.legend(fontsize=8, loc="best")
    for ax in axes[-2:]:
        ax.set_xlabel("Training step")
    fig.suptitle(f"GRPO training dynamics: {model} seed={seed}", fontsize=14)
    fig.tight_layout()
    out_path = OUT_DIR / f"dynamics_{model}_s{seed}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")
    if not has_any:
        print("WARNING: no metrics plotted - check METRIC_KEYS against actual wandb columns")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2_5_1_5b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--divergence-threshold", type=float, default=0.05,
                   help="|reward_mean - reward_mean_S0| above this counts as divergent")
    p.add_argument("--std-floor", type=float, default=1e-4)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    histories = fetch_run_histories(args.model, args.seed)
    if not histories:
        raise SystemExit("No runs found - check entity/project/model/seed")

    save_per_run_csvs(histories, args.model, args.seed)
    plot_dynamics(histories, args.model, args.seed)

    summary = divergence_summary(histories, args.std_floor, args.divergence_threshold)
    summary["_meta"] = {
        "model": args.model, "seed": args.seed,
        "divergence_threshold": args.divergence_threshold,
        "std_floor": args.std_floor,
        "wandb_project": f"{ENTITY}/{PROJECT}",
    }
    summary_path = OUT_DIR / f"summary_{args.model}_s{args.seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_path}")

    print("\n=== Divergence summary ===")
    for cond in CONDITIONS:
        if cond in summary:
            print(f"  {cond}: {summary[cond]}")


if __name__ == "__main__":
    main()
