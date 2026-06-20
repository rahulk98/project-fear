#!/usr/bin/env python
"""Experiment 2 probe analysis: linear probes on extracted hidden states.

Usage:
    python scripts/probe_analysis.py --model qwen2_5_1_5b --conditions S0,S2,C2 \\
        --benchmark math_test --probing-dir outputs/probing

Analyses: (a) layer-sweep AUROC correct-vs-incorrect, (b) abstention probe at
best layer, (c) logit-entropy bootstrap summaries, (d) cross-condition transfer
matrix + cosine probe-weight similarity. Outputs JSON summary + matplotlib figures.
No GPU required; runs on the laptop with sklearn and numpy.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asym_rewards.metrics import auroc, bootstrap_ci  # noqa: E402
from asym_rewards.probing import LABEL_MAP  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load H, y from NPZ + meta from companion JSON. Returns (H, y, meta_json)."""
    data = np.load(str(path))
    H = data["H"].astype(np.float32)
    y = data["y"].astype(np.int32)
    meta_path = path.with_suffix("").with_suffix(".meta.json")
    if meta_path.exists():
        meta_json = json.loads(meta_path.read_text())
    else:
        meta_json = {}
    return H, y, meta_json


def _train_test_split(
    H: np.ndarray,
    y: np.ndarray,
    test_frac: float = 0.3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = max(1, int(len(y) * test_frac))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return H[train_idx], y[train_idx], H[test_idx], y[test_idx]


def _fit_probe(H_train: np.ndarray, y_train: np.ndarray, pos_label: int, neg_label: int):
    """Fit StandardScaler + L2-penalised LogisticRegression on a binary subset."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    mask = (y_train == pos_label) | (y_train == neg_label)
    Hs = H_train[mask]
    ys = (y_train[mask] == pos_label).astype(int)
    if Hs.shape[0] < 4 or len(np.unique(ys)) < 2:
        return None, None
    scaler = StandardScaler()
    Hs_scaled = scaler.fit_transform(Hs)
    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=0)
    clf.fit(Hs_scaled, ys)
    return scaler, clf


def _probe_auroc(
    scaler, clf, H_test: np.ndarray, y_test: np.ndarray, pos_label: int, neg_label: int
) -> float:
    """AUROC of a fitted probe on the test fold. Returns nan if degenerate."""
    if scaler is None or clf is None:
        return float("nan")
    mask = (y_test == pos_label) | (y_test == neg_label)
    Hs = H_test[mask]
    ys = (y_test[mask] == pos_label).astype(int)
    if Hs.shape[0] < 2 or len(np.unique(ys)) < 2:
        return float("nan")
    scores = clf.predict_proba(scaler.transform(Hs))[:, 1]
    return auroc(scores, ys)


def _bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Bootstrap 95% CI for mean(a) - mean(b) using numpy only (no scipy)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(a), len(a))
        ib = rng.integers(0, len(b), len(b))
        diffs.append(a[ia].mean() - b[ib].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def _cosine_sim(w1: np.ndarray, w2: np.ndarray) -> float:
    n1 = np.linalg.norm(w1)
    n2 = np.linalg.norm(w2)
    if n1 < 1e-12 or n2 < 1e-12:
        return float("nan")
    return float(np.dot(w1, w2) / (n1 * n2))


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def layer_sweep_auroc(
    H: np.ndarray,
    y: np.ndarray,
    pos_label: int = LABEL_MAP["correct"],
    neg_label: int = LABEL_MAP["incorrect"],
    test_frac: float = 0.3,
) -> tuple[list[float], list[tuple[float, float]], int | None]:
    """Fit correct-vs-incorrect probes at every layer; return (auROCs, CIs, best_layer_idx)."""
    H_tr, y_tr, H_te, y_te = _train_test_split(H, y, test_frac=test_frac)
    n_layers = H.shape[1] if H.ndim == 3 else 1
    aucs: list[float] = []
    cis: list[tuple[float, float]] = []
    for li in range(n_layers):
        Hl_tr = H_tr[:, li, :] if H.ndim == 3 else H_tr
        Hl_te = H_te[:, li, :] if H.ndim == 3 else H_te
        scaler, clf = _fit_probe(Hl_tr, y_tr, pos_label, neg_label)
        auc = _probe_auroc(scaler, clf, Hl_te, y_te, pos_label, neg_label)
        aucs.append(auc)
        if not np.isnan(auc):
            mask = (y_te == pos_label) | (y_te == neg_label)
            Hs_te = Hl_te[mask]
            ys_te = (y_te[mask] == pos_label).astype(int)
            scores = clf.predict_proba(scaler.transform(Hs_te))[:, 1]
            ci = bootstrap_ci(auroc, scores, ys_te, n_boot=500)
        else:
            ci = (float("nan"), float("nan"))
        cis.append(ci)

    valid = [(i, v) for i, v in enumerate(aucs) if not np.isnan(v)]
    best_layer = max(valid, key=lambda x: x[1])[0] if valid else None
    return aucs, cis, best_layer


def abstention_probe_at_layer(
    H: np.ndarray,
    y: np.ndarray,
    layer_idx: int,
    test_frac: float = 0.3,
) -> tuple[float, tuple[float, float], object | None, object | None]:
    """Abstained-vs-attempted probe at a single layer. Returns (auroc, ci, scaler, clf)."""
    pos_label = LABEL_MAP["abstained"]
    neg_labels = [LABEL_MAP["correct"], LABEL_MAP["incorrect"]]
    # Treat correct and incorrect as a combined "attempted" class (label 0).
    y_bin = np.where(np.isin(y, neg_labels), 0, np.where(y == pos_label, 1, -1))
    keep = y_bin != -1
    H_sub = H[:, layer_idx, :] if H.ndim == 3 else H
    H_sub = H_sub[keep]
    y_sub = y_bin[keep]
    if len(y_sub) < 4 or len(np.unique(y_sub)) < 2:
        return float("nan"), (float("nan"), float("nan")), None, None

    H_tr, y_tr, H_te, y_te = _train_test_split(H_sub, y_sub, test_frac=test_frac)
    # A single-class TRAIN fold cannot be fit -> no probe at all.
    if len(np.unique(y_tr)) < 2:
        return float("nan"), (float("nan"), float("nan")), None, None
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    H_tr_s = scaler.fit_transform(H_tr)
    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=0)
    clf.fit(H_tr_s, y_tr)
    # A single-class TEST fold only makes the IN-condition AUROC undefined; the
    # trained probe is still valid and is kept for the cross-condition transfer
    # matrix. Return nan AUROC but a live scaler/clf.
    if len(np.unique(y_te)) < 2:
        return float("nan"), (float("nan"), float("nan")), scaler, clf
    scores = clf.predict_proba(scaler.transform(H_te))[:, 1]
    auc = auroc(scores, y_te)
    ci = bootstrap_ci(auroc, scores, y_te, n_boot=500) if not np.isnan(auc) else (float("nan"), float("nan"))
    return auc, ci, scaler, clf


def entropy_summary(meta_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-label entropy + max-prob means from meta dicts."""
    by_label: dict[str, list[float]] = {}
    for m in meta_items:
        lbl = m.get("label", "unknown")
        ent = m.get("logit_entropy")
        if ent is not None:
            by_label.setdefault(lbl, []).append(float(ent))
    out: dict[str, Any] = {}
    for lbl, vals in by_label.items():
        arr = np.array(vals)
        out[lbl] = {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(vals)}
    # Bootstrap mean-diff: correct vs incorrect entropy
    corr = np.array(by_label.get("correct", [float("nan")]))
    incorr = np.array(by_label.get("incorrect", [float("nan")]))
    if len(corr) > 1 and len(incorr) > 1:
        lo, hi = _bootstrap_mean_diff(corr, incorr)
        out["correct_vs_incorrect_entropy_diff_ci"] = [lo, hi]
    return out


# ---------------------------------------------------------------------------
# Figure rendering (matplotlib Agg)
# ---------------------------------------------------------------------------

def _save_layer_sweep_fig(conditions: list[str], aucs_per_cond: list[list[float]],
                          layer_names: list[str], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for cond, aucs in zip(conditions, aucs_per_cond):
        # aucs may already contain None (JSON-serialized nan) -- guard before
        # np.isnan, which raises TypeError on None.
        ys = [float("nan") if (a is None or np.isnan(a)) else a for a in aucs]
        xs = list(range(len(ys)))
        ax.plot(xs, ys, label=cond, marker="o", markersize=3)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("AUROC (correct vs incorrect)")
    ax.set_title("Correct-vs-incorrect probe AUROC by layer")
    ax.legend(fontsize=8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    tick_step = max(1, len(layer_names) // 10)
    ax.set_xticks(range(0, len(layer_names), tick_step))
    ax.set_xticklabels(layer_names[::tick_step], rotation=45, fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def _save_transfer_matrix_fig(conditions: list[str], matrix: np.ndarray, out_path: Path,
                               title: str = "Cross-condition abstention probe transfer (AUROC)") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(4, len(conditions)), max(4, len(conditions))))
    masked = np.where(np.isnan(matrix), 0.5, matrix)
    im = ax.imshow(masked, vmin=0.4, vmax=1.0, cmap="RdYlGn")
    ax.set_xticks(range(len(conditions)))
    ax.set_yticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    ax.set_yticklabels(conditions)
    ax.set_xlabel("Test condition")
    ax.set_ylabel("Train condition")
    ax.set_title(title)
    for i in range(len(conditions)):
        for j in range(len(conditions)):
            val = matrix[i, j]
            txt = f"{val:.2f}" if not np.isnan(val) else "nan"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def _save_entropy_dist_fig(conditions: list[str], entropy_data: list[dict[str, Any]],
                           out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(conditions)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, cond, edata in zip(axes, conditions, entropy_data):
        for lbl, color in [("correct", "green"), ("incorrect", "red"), ("abstained", "blue")]:
            stats = edata.get(lbl, {})
            if stats:
                ax.bar(lbl, stats.get("mean", 0), yerr=stats.get("std", 0),
                       color=color, alpha=0.7, capsize=3)
        ax.set_title(cond, fontsize=9)
        ax.set_ylabel("Mean logit entropy" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.suptitle("Last-token logit entropy by label and condition")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-condition transfer
# ---------------------------------------------------------------------------

def cross_condition_transfer(
    conditions: list[str],
    load_npz,
    args,
    best_layers: dict[str, "int | None"],
) -> dict:
    """Compute the cross-condition abstention-probe transfer matrix and cosine
    similarity matrix at a single COMMON LAYER.

    Probe weights and hidden-state coordinates are only comparable WITHIN the
    same transformer layer; using each condition's own best layer would make
    off-diagonal transfer AUROCs and cosines meaningless. Common layer = the
    modal best layer across conditions (ties broken by first occurrence).

    Args:
        conditions: ordered list of condition identifiers to evaluate.
        load_npz: callable (path) -> (H, y, meta_json) -- matches _load_npz
            signature; injected for testability.
        args: argparse.Namespace with attributes: model, dtype, benchmark; used
            to construct NPZ paths relative to the probing directory.
        best_layers: mapping condition -> best_layer_index (or None) from the
            layer-sweep step. Drives common-layer selection.

    Returns:
        dict with keys:
            present_conds   : list[str] -- conditions with valid probes
            common_layer    : int
            transfer_matrix : np.ndarray shape (nc, nc), AUROC of train-A on
                              test-B hidden states at common_layer
            cosine_matrix   : np.ndarray shape (nc, nc), symmetric cosine
                              similarity of probe weight vectors
    """
    probing_dir = Path(args.probing_dir)

    valid_bl = [bl for bl in best_layers.values() if bl is not None]
    common_layer = Counter(valid_bl).most_common(1)[0][0] if valid_bl else 0
    print(f"  transfer/cosine at common layer {common_layer}")

    pos_label = LABEL_MAP["abstained"]
    neg_labels_set = {LABEL_MAP["correct"], LABEL_MAP["incorrect"]}
    common_probes: dict[str, tuple] = {}
    common_test: dict[str, tuple] = {}

    for cond in conditions:
        npz_path_c = probing_dir / f"{args.model}_{cond}_{args.dtype}_{args.benchmark}_hidden.npz"
        if not npz_path_c.exists():
            continue
        H_c, y_c, _ = load_npz(npz_path_c)
        if H_c.ndim == 3:
            li = common_layer if common_layer < H_c.shape[1] else H_c.shape[1] - 1
        else:
            li = 0
        _, _, sc_c, clf_c = abstention_probe_at_layer(H_c, y_c, li)
        if sc_c is None:
            continue
        common_probes[cond] = (sc_c, clf_c)
        Hc_layer = H_c[:, li, :] if H_c.ndim == 3 else H_c
        y_bin_c = np.where(
            np.isin(y_c, list(neg_labels_set)), 0,
            np.where(y_c == pos_label, 1, -1),
        )
        keep_c = y_bin_c != -1
        common_test[cond] = (Hc_layer[keep_c], y_bin_c[keep_c])

    present_conds = [c for c in conditions if c in common_probes and c in common_test]
    nc = len(present_conds)
    transfer_matrix = np.full((nc, nc), float("nan"))
    cosine_matrix = np.full((nc, nc), float("nan"))

    for i, cond_a in enumerate(present_conds):
        scaler_a, clf_a = common_probes[cond_a]
        w_a = clf_a.coef_[0]
        for j, cond_b in enumerate(present_conds):
            # Apply A's probe to B's hidden states AT THE SAME (common) layer.
            # Diagonal is in-sample-optimistic (probe A saw some of A's rows);
            # the off-diagonal transfer is the clean signal.
            Hb_layer, y_b_bin = common_test[cond_b]
            if len(y_b_bin) < 2 or len(np.unique(y_b_bin)) < 2:
                transfer_matrix[i, j] = float("nan")
            else:
                scores = clf_a.predict_proba(scaler_a.transform(Hb_layer))[:, 1]
                transfer_matrix[i, j] = auroc(scores, y_b_bin)
            if j >= i:
                w_b = common_probes[cond_b][1].coef_[0]
                sim = _cosine_sim(w_a, w_b)
                cosine_matrix[i, j] = sim
                cosine_matrix[j, i] = sim

    return {
        "present_conds": present_conds,
        "common_layer": int(common_layer),
        "transfer_matrix": transfer_matrix,
        "cosine_matrix": cosine_matrix,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp-2 probe analysis pipeline.")
    p.add_argument("--model", required=True)
    p.add_argument("--conditions", required=True, help="Comma-separated condition labels, e.g. S0,S2,C2")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--probing-dir", default=str(REPO_ROOT / "outputs" / "probing"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "probing" / "analysis"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    probing_dir = Path(args.probing_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = REPO_ROOT / "docs" / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    all_aucs: list[list[float]] = []
    layer_names_ref: list[str] = []
    abstention_aucs: dict[str, float] = {}
    abstention_probes: dict[str, tuple[object, object]] = {}
    best_layers: dict[str, int | None] = {}
    entropy_by_cond: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"model": args.model, "benchmark": args.benchmark, "conditions": {}}

    for cond in conditions:
        stem = f"{args.model}_{cond}_{args.dtype}_{args.benchmark}_hidden"
        npz_path = probing_dir / f"{stem}.npz"
        if not npz_path.exists():
            print(f"  WARNING: {npz_path} not found; skipping condition {cond}")
            all_aucs.append([float("nan")])
            entropy_by_cond.append({})
            continue

        H, y, meta_json = _load_npz(npz_path)
        layer_names = meta_json.get("layer_names", [f"layer_{i}" for i in range(H.shape[1] if H.ndim == 3 else 1)])
        if not layer_names_ref:
            layer_names_ref = layer_names

        meta_items: list[dict[str, Any]] = meta_json.get("items", [])

        # (a) Layer sweep
        aucs, cis, best_layer = layer_sweep_auroc(H, y)
        all_aucs.append(aucs)
        best_layers[cond] = best_layer
        print(f"  {cond}: layer sweep done. Best layer={best_layer}, "
              f"AUROC={aucs[best_layer]:.3f}" if best_layer is not None else f"  {cond}: no valid layer")

        # (b) Abstention probe at best layer
        bl = best_layer if best_layer is not None else (H.shape[1] - 1 if H.ndim == 3 else 0)
        ab_auc, ab_ci, ab_scaler, ab_clf = abstention_probe_at_layer(H, y, bl)
        abstention_aucs[cond] = ab_auc
        if ab_scaler is not None:
            abstention_probes[cond] = (ab_scaler, ab_clf)
        print(f"  {cond}: abstention AUROC={ab_auc:.3f}" if not np.isnan(ab_auc) else f"  {cond}: abstention AUROC=nan")

        # (c) Entropy summaries
        esum = entropy_summary(meta_items)
        entropy_by_cond.append(esum)

        summary["conditions"][cond] = {
            "layer_aucs": [a if not np.isnan(a) else None for a in aucs],
            "layer_cis": [[lo if not np.isnan(lo) else None, hi if not np.isnan(hi) else None] for lo, hi in cis],
            "best_layer": best_layer,
            "best_layer_name": layer_names[best_layer] if best_layer is not None and best_layer < len(layer_names) else None,
            # best_layer is chosen by argmax over the SAME test fold used for
            # evaluation, so the AUROC at best_layer is an upper bound (selection
            # bias). A proper estimate requires selecting the best layer on a
            # held-out validation split and reporting AUROC on a separate test split.
            "best_layer_auroc_is_selection_biased": True,
            "abstention_auroc": ab_auc if not np.isnan(ab_auc) else None,
            "abstention_auroc_ci": [ab_ci[0] if not np.isnan(ab_ci[0]) else None,
                                    ab_ci[1] if not np.isnan(ab_ci[1]) else None],
            "entropy_summary": esum,
        }

    # (d) Cross-condition transfer matrix + cosine similarity
    xfer = cross_condition_transfer(conditions, _load_npz, args, best_layers)
    present_conds = xfer["present_conds"]
    common_layer = xfer["common_layer"]
    transfer_matrix = xfer["transfer_matrix"]
    cosine_matrix = xfer["cosine_matrix"]
    nc = len(present_conds)

    summary["transfer_common_layer"] = int(common_layer)
    summary["transfer_matrix"] = {
        "conditions": present_conds,
        "common_layer": int(common_layer),
        "auroc": [[v if not np.isnan(v) else None for v in row] for row in transfer_matrix.tolist()],
    }
    summary["cosine_similarity"] = {
        "conditions": present_conds,
        "matrix": [[v if not np.isnan(v) else None for v in row] for row in cosine_matrix.tolist()],
    }

    # Write JSON summary
    summary_path = out_dir / f"{args.model}_probe_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  summary -> {summary_path}")

    # Render figures
    valid_conds_sweep = [c for c in conditions if c in summary["conditions"] and summary["conditions"][c].get("layer_aucs")]
    if valid_conds_sweep and layer_names_ref:
        _save_layer_sweep_fig(
            conditions=valid_conds_sweep,
            aucs_per_cond=[summary["conditions"][c]["layer_aucs"] for c in valid_conds_sweep],
            layer_names=layer_names_ref,
            out_path=figs_dir / "exp2_layer_sweep.png",
        )
        print(f"  figure -> {figs_dir / 'exp2_layer_sweep.png'}")

    if nc > 0 and not np.all(np.isnan(transfer_matrix)):
        _save_transfer_matrix_fig(
            conditions=present_conds,
            matrix=transfer_matrix,
            out_path=figs_dir / "exp2_transfer_matrix.png",
        )
        print(f"  figure -> {figs_dir / 'exp2_transfer_matrix.png'}")

    if entropy_by_cond and any(entropy_by_cond):
        present_ent_conds = [c for c, e in zip(conditions, entropy_by_cond) if e]
        present_ent = [e for e in entropy_by_cond if e]
        if present_ent_conds:
            _save_entropy_dist_fig(
                conditions=present_ent_conds,
                entropy_data=present_ent,
                out_path=figs_dir / "exp2_entropy_dist.png",
            )
            print(f"  figure -> {figs_dir / 'exp2_entropy_dist.png'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
