# Project FEAR

Code for an academic project on asymmetric correct/wrong/abstain rewards in GRPO training of
Qwen 2.5 1.5B. Write-up: [rahulkrishnan.me/grpo-abstention-dial](https://rahulkrishnan.me/grpo-abstention-dial/).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # single A100 40GB, bf16
python scripts/prepare_data.py      # MATH, GSM8K, AIME 2024/2025, MMLU-STEM, TruthfulQA-MC
```

Reward conditions (S0 to S4, C1, C2, P0) are defined in `src/asym_rewards/reward.py`;
training settings are in `configs/training.yaml`.

## Reproduce

**1. Train and evaluate** each condition (outputs go to `outputs/grpo` and `outputs/eval`):

```bash
python scripts/train_grpo.py --model qwen2_5_1_5b --condition S0 --seed 42
python scripts/eval_calibration.py --model qwen2_5_1_5b --condition S0 --k 16
python scripts/eval_calibration.py --model qwen2_5_1_5b --condition S0 --greedy
```

No-normalization ablation: add `--advantage-norm zhu_signed` to `train_grpo.py`.

**2. Figures** from the bundled summaries in `results/` (needs only numpy and matplotlib):

```bash
python scripts/make_plots.py
```

To rebuild the summaries from your own evaluation outputs: `scripts/analyze_rollouts.py`,
`scripts/cross_condition_trajectory.py`, `scripts/stacking_null.py`.

**3. Analyses** in `research/followups/`, each writing a JSON like the ones in
`results/followups/`. They read evaluation outputs from `FEAR_EVAL_DIR` (default
`outputs/eval`), hidden states from `FEAR_PROBING_DIR` (default `outputs/probing`, made by
`research/extract_probe_states.py`) and models from `FEAR_MODELS_DIR` (default `models`).

| Analysis | Script |
|---|---|
| Error bars on the skip rate | `e4_abstention_ci.py` |
| Do skips land on hard problems | `e2_wise_wasteful.py` |
| What the skips really are | `q5_abstention_audit.py` |
| Paired differences between conditions | `q6_paired_ci.py` |
| Hidden-state probe and controls | `e1_probe_recheck.py` |
| P(True) and semantic entropy | `e3_ptrue_se.py` |
| Probe-guided skipping at a matched answer rate | `q7_probe_matched_coverage.py` |
| Same t, same advantages | `q8_matched_t.py`, `tests/test_advantage_invariance.py` |
| TruthfulQA-MC with shuffled options | `q1_truthfulqa_shuffled.py` |

**4. Measure skip quality on any model** with the Inspect eval in
[evals/abstention/](evals/abstention/README.md).

## Tests

```bash
pytest tests
```

## License

MIT. Collaborator: Sanjana Deshpande.
