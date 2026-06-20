"""Tests for _zhu_signed_collapse_guard (grpo.py).

numpy/stdlib only - no torch, no math_verify, no vllm.

The guard is a pure module-level function that receives a fake trainer object
and a list of outcome strings. It appends metric keys, maintains a rolling
10-step abstention history, and raises RuntimeError only when ALL of the last
10 steps had abstention > 0.97.

grpo.py has heavy top-level imports (datasets, peft, pandas) that are absent on
the laptop. We stub them out via sys.modules before importing the module so
the module body runs without errors. The guard function itself is pure stdlib.

Run:
    python3 tests/test_zhu_signed_guard.py
"""
from __future__ import annotations

import sys
import types
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Import _zhu_signed_collapse_guard (a pure-stdlib function) from grpo.py, whose
# module-level imports (datasets, peft, pandas, torch, ...) are absent on the dev
# laptop. We stub those out ONLY for the duration of the import, then restore
# sys.modules exactly. Without the restore, the fake modules + the stub-built
# asym_rewards.* would leak and break any later test in the same pytest process
# that imports the real torch/math_verify (codex xhigh P2).
# ---------------------------------------------------------------------------

def _make_stub(name: str):
    m = types.ModuleType(name)
    m.__getattr__ = lambda attr: types.SimpleNamespace()  # type: ignore[assignment]
    return m


def _import_guard():
    """Import the guard under temporary dependency stubs, then restore sys.modules."""
    before = set(sys.modules)
    stub_names = [
        "datasets", "pandas", "pd", "peft", "trl", "transformers", "torch",
        "asym_rewards.data", "asym_rewards.models", "math_verify",
    ]
    for name in stub_names:
        if name not in sys.modules:
            sys.modules[name] = _make_stub(name)
    try:
        from asym_rewards.grpo import _zhu_signed_collapse_guard as guard
        return guard
    finally:
        # Drop everything we added (the fakes + any asym_rewards.* imported under
        # them) so later tests re-import the real modules cleanly. We only ever
        # added absent keys, so deleting the delta never clobbers a real module.
        # ALSO strip the submodule attributes the import machinery attached to a
        # pre-existing parent package object (e.g. asym_rewards.grpo set as
        # `asym_rewards.grpo`): deleting sys.modules alone leaves those reachable
        # via `from asym_rewards import grpo` in the same process (codex xhigh P2).
        added = set(sys.modules) - before
        for name in sorted(added, key=len, reverse=True):
            del sys.modules[name]
            if "." in name:
                pkg, _, sub = name.rpartition(".")
                if pkg not in added:  # parent pre-existed -> we attached `sub` to it
                    parent_mod = sys.modules.get(pkg)
                    if parent_mod is not None:
                        try:
                            delattr(parent_mod, sub)
                        except AttributeError:
                            pass


_zhu_signed_collapse_guard = _import_guard()


# ---------------------------------------------------------------------------
# Fake trainer: supports _metrics (defaultdict-list) and _zhu_signed_abstain_hist
# ---------------------------------------------------------------------------

class _FakeTrainer:
    def __init__(self):
        self._metrics: dict = defaultdict(list)
        # _zhu_signed_abstain_hist may or may not exist; the guard initializes
        # it lazily, so we do NOT pre-set it here.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mostly_silent(n: int = 100, silent_frac: float = 0.99) -> list:
    """Return an outcomes list whose silent fraction exceeds 0.97.

    Default n=100 so that int(100 * 0.99) = 99 -> 99/100 = 0.99 > 0.97.
    Using n=8 would give int(8*0.99)=7 -> 7/8=0.875 which is below the
    threshold; this function must produce rates that genuinely exceed 0.97.
    """
    n_silent = int(n * silent_frac)
    n_wrong = n - n_silent
    return ["silent"] * n_silent + ["wrong"] * n_wrong


def _mixed_outcomes(n: int = 8) -> list:
    """Return an outcomes list with a clearly mixed distribution."""
    return (["correct"] * (n // 3) + ["wrong"] * (n // 3)
            + ["silent"] * (n - 2 * (n // 3)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_raise_on_single_high_abstention_batch():
    """One >0.97-silent batch must NOT raise (only 1 of 10 required steps)."""
    trainer = _FakeTrainer()
    outcomes = _mostly_silent()
    # Should not raise
    _zhu_signed_collapse_guard(trainer, outcomes)
    assert 'zhu_signed_abstain_rate' in trainer._metrics
    assert trainer._metrics['zhu_signed_abstain_rate'][-1] > 0.97


def test_no_raise_on_10_mixed_batches():
    """10 consecutive mixed batches must NOT raise."""
    trainer = _FakeTrainer()
    for _ in range(10):
        _zhu_signed_collapse_guard(trainer, _mixed_outcomes())
    # No exception; history has exactly 10 entries (window cap).
    assert len(trainer._zhu_signed_abstain_hist) == 10


def test_raises_after_10_consecutive_high_abstention_batches():
    """10 consecutive >0.97-silent batches MUST raise RuntimeError."""
    trainer = _FakeTrainer()
    raised = False
    try:
        for _ in range(10):
            _zhu_signed_collapse_guard(trainer, _mostly_silent())
    except RuntimeError as exc:
        raised = True
        assert "zhu_signed collapse" in str(exc), f"unexpected message: {exc}"
    assert raised, "Expected RuntimeError was NOT raised after 10 high-abstention steps"


def test_no_raise_when_9th_step_is_below_threshold():
    """9 high + 1 mixed: window has 10 entries but last is < 0.97, no raise."""
    trainer = _FakeTrainer()
    # 9 high-abstention steps
    for _ in range(9):
        _zhu_signed_collapse_guard(trainer, _mostly_silent())
    # 10th step is mixed -> abstain_rate < 0.97
    _zhu_signed_collapse_guard(trainer, _mixed_outcomes())
    # Should reach here without raising


def test_rolling_window_capped_at_10():
    """After 15 calls the history should contain exactly 10 entries."""
    trainer = _FakeTrainer()
    for _ in range(15):
        try:
            _zhu_signed_collapse_guard(trainer, _mostly_silent())
        except RuntimeError:
            break  # guard raises at step 10, break then check history
    assert len(trainer._zhu_signed_abstain_hist) == 10


def test_metric_keys_populated():
    """All expected metric keys must be populated on every call."""
    trainer = _FakeTrainer()
    outcomes = _mixed_outcomes(8)
    _zhu_signed_collapse_guard(trainer, outcomes)
    assert 'zhu_signed_abstain_rate' in trainer._metrics, "missing zhu_signed_abstain_rate"
    assert 'zhu_signed_outcome_entropy_proxy' in trainer._metrics, "missing entropy proxy"
    # collapse_abort only set on actual abort - not here
    assert 'zhu_signed_collapse_abort' not in trainer._metrics


def test_collapse_abort_key_set_on_raise():
    """zhu_signed_collapse_abort must be appended before the RuntimeError."""
    trainer = _FakeTrainer()
    for _ in range(9):
        _zhu_signed_collapse_guard(trainer, _mostly_silent())
    try:
        _zhu_signed_collapse_guard(trainer, _mostly_silent())
    except RuntimeError:
        pass
    assert 'zhu_signed_collapse_abort' in trainer._metrics
    assert trainer._metrics['zhu_signed_collapse_abort'][-1] == 1.0


def test_empty_outcomes_list():
    """Empty outcomes list should not crash; abstain_rate=0, entropy_proxy=0."""
    trainer = _FakeTrainer()
    _zhu_signed_collapse_guard(trainer, [])
    assert trainer._metrics['zhu_signed_abstain_rate'][-1] == 0.0
    assert trainer._metrics['zhu_signed_outcome_entropy_proxy'][-1] == 0.0


def test_abstain_rate_correct_fraction():
    """Verify abstain_rate equals n_silent/n."""
    trainer = _FakeTrainer()
    outcomes = ["silent", "silent", "wrong", "correct"]  # 2/4 = 0.5
    _zhu_signed_collapse_guard(trainer, outcomes)
    assert abs(trainer._metrics['zhu_signed_abstain_rate'][-1] - 0.5) < 1e-9


def test_entropy_proxy_zero_for_single_outcome():
    """All-silent outcomes -> unique=1 -> entropy_proxy=0.0."""
    trainer = _FakeTrainer()
    _zhu_signed_collapse_guard(trainer, ["silent"] * 5)
    assert trainer._metrics['zhu_signed_outcome_entropy_proxy'][-1] == 0.0


def test_entropy_proxy_nonzero_for_multiple_outcomes():
    """Multiple distinct outcomes -> entropy_proxy > 0."""
    trainer = _FakeTrainer()
    _zhu_signed_collapse_guard(trainer, ["silent", "correct", "wrong"])
    assert trainer._metrics['zhu_signed_outcome_entropy_proxy'][-1] > 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_all() -> None:
    tests = [
        test_no_raise_on_single_high_abstention_batch,
        test_no_raise_on_10_mixed_batches,
        test_raises_after_10_consecutive_high_abstention_batches,
        test_no_raise_when_9th_step_is_below_threshold,
        test_rolling_window_capped_at_10,
        test_metric_keys_populated,
        test_collapse_abort_key_set_on_raise,
        test_empty_outcomes_list,
        test_abstain_rate_correct_fraction,
        test_entropy_proxy_zero_for_single_outcome,
        test_entropy_proxy_nonzero_for_multiple_outcomes,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {t.__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
