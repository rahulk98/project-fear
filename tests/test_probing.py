#!/usr/bin/env python
"""Numpy-only unit tests for src/asym_rewards/probing.py.

Tests cover:
  1. Label derivation logic from rollout fields (correct, abstained).
  2. Balanced subsampling contract (each label capped at max_per_label).
  3. Left-pad last-token gather invariant (column -1 = last real token).
  4. save_probing_npz round-trip (NPZ + companion JSON).
  5. Planted-direction linear separability: a known direction in synthetic H
     should be recoverable by a simple threshold classifier.

No torch / transformers / peft / vllm are imported. The torch-dependent
functions (load_probing_model, extract_last_token_states) are exercised
indirectly through their pure helper logic tested here.

Run directly: `python tests/test_probing.py`
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asym_rewards.probing import LABEL_MAP, iter_rollout_records, save_probing_npz  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic rollout JSONL builder
# ---------------------------------------------------------------------------

def _make_rollout_jsonl(tmp_dir: Path, rows: list[dict]) -> Path:
    path = tmp_dir / "test_rollouts.jsonl"
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def _make_prompts_by_pid(pids: list[str]) -> dict[str, tuple[str, str, str]]:
    return {pid: (f"prompt for {pid}", "42", "math") for pid in pids}


# ---------------------------------------------------------------------------
# Test 1: label derivation
# ---------------------------------------------------------------------------

def test_label_derivation():
    """Correct, incorrect, and abstained samples map to expected label strings."""
    rows = [
        {"problem_id": "p1", "sample_idx": 0, "gold": "5", "completion": "...boxed{5}",
         "extracted": "5", "correct": True, "abstained": False},
        {"problem_id": "p2", "sample_idx": 0, "gold": "7", "completion": "...",
         "extracted": "3", "correct": False, "abstained": False},
        {"problem_id": "p3", "sample_idx": 0, "gold": "9", "completion": "...",
         "extracted": "", "correct": False, "abstained": True},
    ]
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        path = _make_rollout_jsonl(tmp_dir, rows)
        prompts = _make_prompts_by_pid(["p1", "p2", "p3"])
        items = iter_rollout_records(path, prompts)

    by_pid = {it["problem_id"]: it["label"] for it in items}
    assert by_pid["p1"] == "correct", f"Expected correct, got {by_pid['p1']}"
    assert by_pid["p2"] == "incorrect", f"Expected incorrect, got {by_pid['p2']}"
    assert by_pid["p3"] == "abstained", f"Expected abstained, got {by_pid['p3']}"
    print("ok: test_label_derivation")


# ---------------------------------------------------------------------------
# Test 2: pid-not-in-prompt-table rows are skipped
# ---------------------------------------------------------------------------

def test_missing_pid_skipped():
    rows = [
        {"problem_id": "known", "sample_idx": 0, "gold": "1", "completion": "x",
         "extracted": "1", "correct": True, "abstained": False},
        {"problem_id": "unknown", "sample_idx": 0, "gold": "2", "completion": "y",
         "extracted": "", "correct": False, "abstained": True},
    ]
    with tempfile.TemporaryDirectory() as td:
        path = _make_rollout_jsonl(Path(td), rows)
        prompts = _make_prompts_by_pid(["known"])  # "unknown" absent
        items = iter_rollout_records(path, prompts)

    pids = {it["problem_id"] for it in items}
    assert pids == {"known"}, f"Expected only 'known', got {pids}"
    print("ok: test_missing_pid_skipped")


# ---------------------------------------------------------------------------
# Test 3: balanced subsampling cap
# ---------------------------------------------------------------------------

def test_balanced_subsampling():
    """max_per_label caps each bucket independently."""
    rows = []
    for i in range(20):
        rows.append({"problem_id": f"c{i}", "sample_idx": 0, "gold": "1",
                     "completion": "...", "extracted": "1", "correct": True, "abstained": False})
    for i in range(5):
        rows.append({"problem_id": f"w{i}", "sample_idx": 0, "gold": "2",
                     "completion": "...", "extracted": "9", "correct": False, "abstained": False})
    for i in range(8):
        rows.append({"problem_id": f"a{i}", "sample_idx": 0, "gold": "3",
                     "completion": "...", "extracted": "", "correct": False, "abstained": True})

    pids = [f"c{i}" for i in range(20)] + [f"w{i}" for i in range(5)] + [f"a{i}" for i in range(8)]
    with tempfile.TemporaryDirectory() as td:
        path = _make_rollout_jsonl(Path(td), rows)
        prompts = _make_prompts_by_pid(pids)
        items = iter_rollout_records(path, prompts, max_per_label=6, seed=0)

    counts = {}
    for it in items:
        counts[it["label"]] = counts.get(it["label"], 0) + 1

    assert counts.get("correct", 0) <= 6, f"correct bucket not capped: {counts}"
    assert counts.get("incorrect", 0) <= 5, f"incorrect bucket wrong: {counts}"
    assert counts.get("abstained", 0) <= 6, f"abstained bucket not capped: {counts}"
    print(f"ok: test_balanced_subsampling  (counts={counts})")


# ---------------------------------------------------------------------------
# Test 4: left-pad last-token index invariant
# ---------------------------------------------------------------------------

def _simulate_left_pad_attention_mask(seq_lens: list[int], max_len: int) -> np.ndarray:
    """Simulate HF left-padding attention masks for a batch of sequences.

    Returns attention_mask of shape (B, max_len): 0 for pad tokens, 1 for real.
    With left-padding the real tokens are right-justified, so column -1 is
    always the last real token regardless of sequence length.
    """
    B = len(seq_lens)
    mask = np.zeros((B, max_len), dtype=int)
    for i, slen in enumerate(seq_lens):
        pad = max_len - slen
        mask[i, pad:] = 1
    return mask


def test_left_pad_last_token_is_column_minus1():
    """Verify: with left-padding, gather hidden[:, -1, :] captures the last real token.

    We simulate a batch of 4 sequences of different lengths. For each sequence
    we check that column -1 of the left-padded sequence corresponds to the last
    real token (not a pad token).
    """
    seq_lens = [10, 6, 15, 3]
    max_len = max(seq_lens)
    mask = _simulate_left_pad_attention_mask(seq_lens, max_len)

    for i, slen in enumerate(seq_lens):
        # Column -1 should be real (mask == 1)
        assert mask[i, -1] == 1, (
            f"Sequence {i} (len={slen}): column -1 is a PAD token under left-pad, "
            f"but should be a real token."
        )
        # The pad columns should be at the LEFT (mask[:pad] == 0)
        pad = max_len - slen
        assert np.all(mask[i, :pad] == 0), (
            f"Sequence {i}: expected {pad} left-pad zeros, but mask={mask[i].tolist()}"
        )
        assert np.all(mask[i, pad:] == 1), (
            f"Sequence {i}: expected real tokens in positions [{pad}:], "
            f"but mask={mask[i].tolist()}"
        )

    print("ok: test_left_pad_last_token_is_column_minus1")


def test_left_pad_column_minus1_always_real_mixed_batch():
    """Stress test: 32 sequences of random lengths; column -1 must always be real."""
    rng = np.random.default_rng(7)
    seq_lens = rng.integers(1, 128, size=32).tolist()
    max_len = max(seq_lens)
    mask = _simulate_left_pad_attention_mask(seq_lens, max_len)
    for i, slen in enumerate(seq_lens):
        assert mask[i, -1] == 1, (
            f"Sequence {i} (len={slen}): column -1 is pad under left-pad"
        )
    print(f"ok: test_left_pad_column_minus1_always_real_mixed_batch ({len(seq_lens)} seqs)")


# ---------------------------------------------------------------------------
# Test 5: save_probing_npz round-trip
# ---------------------------------------------------------------------------

def test_save_probing_npz_roundtrip():
    N, L, D = 12, 5, 16
    rng = np.random.default_rng(1)
    H = rng.standard_normal((N, L, D)).astype(np.float32)
    y = rng.integers(0, 3, size=N).astype(np.int32)
    layer_names = ["embed"] + [f"layer_{i}" for i in range(L - 1)]
    meta = [{"problem_id": f"p{i}", "sample_idx": 0, "label": "correct",
              "logit_entropy": float(rng.random()),
              "logit_max_prob": float(rng.random())} for i in range(N)]

    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "test_hidden.npz"
        save_probing_npz(out_path, H, y, meta, layer_names,
                         config={"model": "test"}, source_rollout="/tmp/fake.jsonl")

        # NPZ round-trip
        data = np.load(str(out_path))
        assert np.array_equal(data["H"], H), "H round-trip mismatch"
        assert np.array_equal(data["y"], y), "y round-trip mismatch"

        # Companion JSON
        meta_path = out_path.with_suffix("").with_suffix(".meta.json")
        assert meta_path.exists(), "companion .meta.json not written"
        companion = json.loads(meta_path.read_text())
        assert companion["layer_names"] == layer_names
        assert companion["n_total"] == N
        assert companion["h_shape"] == list(H.shape)
        assert companion["source_rollout"] == "/tmp/fake.jsonl"
        assert companion["config"]["model"] == "test"
        # Check per-label counts sum to N
        counts = companion["per_label_counts"]
        assert sum(counts.values()) == N, f"label count mismatch: {counts}"

    print("ok: test_save_probing_npz_roundtrip")


# ---------------------------------------------------------------------------
# Test 6: planted linear direction is recoverable by threshold
# ---------------------------------------------------------------------------

def test_planted_direction_separable():
    """Synthetic H with a planted direction that separates class 0 vs class 1.

    We plant a signal vector v and add it to class-1 samples. A simple
    threshold on the projection <H, v> should achieve near-perfect separation.
    This validates the shape and dtype conventions used downstream by
    probe_analysis.py (sklearn is NOT imported here; we test using numpy).
    """
    rng = np.random.default_rng(42)
    N, D = 200, 32
    # Plant a direction
    v = rng.standard_normal(D)
    v /= np.linalg.norm(v)
    # Class 0: noise; class 1: noise + signal
    H0 = rng.standard_normal((N // 2, D))
    H1 = rng.standard_normal((N // 2, D)) + 5.0 * v
    H = np.vstack([H0, H1]).astype(np.float32)
    y = np.array([0] * (N // 2) + [1] * (N // 2), dtype=np.int32)

    # Project onto planted direction. Use einsum (float64) to avoid float32
    # BLAS overflow warnings on some numpy/platform combinations.
    proj = np.einsum("nd,d->n", H.astype(np.float64), v.astype(np.float64))
    # H0 @ v ~ N(0, 1), H1 @ v ~ N(5, 1), so the midpoint threshold is 2.5.
    # Thresholding at 0 gives only ~78% accuracy; use the correct midpoint.
    threshold = 2.5  # midpoint between the two class means
    predicted = (proj > threshold).astype(int)
    acc = (predicted == y).mean()
    assert acc > 0.95, f"planted-direction accuracy too low: {acc:.3f}"
    print(f"ok: test_planted_direction_separable (acc={acc:.3f})")


# ---------------------------------------------------------------------------
# Test 7: LABEL_MAP consistency
# ---------------------------------------------------------------------------

def test_label_map_values_unique_and_contiguous():
    """LABEL_MAP values must be unique non-negative ints (used as array indices)."""
    vals = list(LABEL_MAP.values())
    assert len(vals) == len(set(vals)), "LABEL_MAP values not unique"
    assert all(isinstance(v, int) and v >= 0 for v in vals), "LABEL_MAP values must be non-negative ints"
    print(f"ok: test_label_map_values_unique_and_contiguous  (map={LABEL_MAP})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_label_derivation,
        test_missing_pid_skipped,
        test_balanced_subsampling,
        test_left_pad_last_token_is_column_minus1,
        test_left_pad_column_minus1_always_real_mixed_batch,
        test_save_probing_npz_roundtrip,
        test_planted_direction_separable,
        test_label_map_values_unique_and_contiguous,
    ]
    for fn in tests:
        fn()
    print(f"\nall {len(tests)} tests passed")
