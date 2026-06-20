#!/usr/bin/env python
"""Unit tests for eval output-naming + provenance helpers in eval_calibration.py.

Guards the seed/K filename-collision fix: a non-default training seed must land
in a distinct filename slot, the default seed must keep legacy names, an explicit
--output-tag must win, and the provenance guard must detect adapter/k/dtype clashes.

Pure-function tests; the heavy vLLM imports in eval_calibration are lazy so the
module imports cheaply. Run directly: `python tests/test_eval_output_naming.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_calibration import _parse_seed, derive_seed_tag, provenance_mismatch  # noqa: E402


def test_default_seed_is_untagged():
    # seed 42 (default) keeps legacy filenames so existing results are not orphaned
    assert derive_seed_tag("outputs/grpo/qwen2_5_1_5b_S0_s42", "") == ""


def test_nondefault_seed_autotag():
    assert derive_seed_tag("outputs/grpo/qwen2_5_1_5b_S0_s456", "") == "_s456"
    assert derive_seed_tag("outputs/grpo/qwen2_5_1_5b_S3_s123", "") == "_s123"


def test_explicit_output_tag_overrides_seed():
    # raw_rloo variant (seed 42) relies on the explicit tag; it must win
    assert derive_seed_tag("outputs/grpo/qwen2_5_math_1_5b_S2_s42_raw_rloo", "raw_rloo") == "_raw_rloo"
    # explicit tag also wins over a non-default seed
    assert derive_seed_tag("outputs/grpo/qwen2_5_1_5b_S0_s456", "custom") == "_custom"


def test_no_adapter_is_untagged():
    assert derive_seed_tag(None, "") == ""


def test_parse_seed():
    assert _parse_seed("outputs/grpo/qwen2_5_1_5b_S0_s456") == 456
    assert _parse_seed("outputs/grpo/qwen2_5_math_1_5b_S2_s42_raw_rloo") == 42
    assert _parse_seed("outputs/grpo/qwen2_5_1_5b_S0_s456/") == 456  # trailing slash tolerated
    assert _parse_seed(None) is None
    assert _parse_seed("Qwen/Qwen2.5-1.5B") is None  # base model, no seed token


def test_provenance_mismatch_detects_adapter_seed():
    prior = {"adapter": "outputs/grpo/x_s42", "k": 32, "dtype": "float16"}
    cur = {"adapter": "outputs/grpo/x_s456", "k": 32, "dtype": "float16"}
    clashes = provenance_mismatch(prior, cur)
    assert any(k == "adapter" for k, _, _ in clashes)


def test_provenance_mismatch_detects_k():
    prior = {"adapter": "a", "k": 16, "dtype": "float16"}
    cur = {"adapter": "a", "k": 32, "dtype": "float16"}
    assert any(k == "k" for k, _, _ in provenance_mismatch(prior, cur))


def test_provenance_match_is_clean():
    prior = {"adapter": "a", "k": 32, "dtype": "float16"}
    assert provenance_mismatch(prior, dict(prior)) == []


def test_provenance_none_existing_not_flagged():
    # a legacy/partial sidecar with a missing field must not raise a false clash
    prior = {"adapter": None, "k": 32, "dtype": "float16"}
    cur = {"adapter": "a", "k": 32, "dtype": "float16"}
    assert provenance_mismatch(prior, cur) == []


if __name__ == "__main__":
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for f in fns:
        try:
            f()
            print(f"PASS {f.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {f.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
