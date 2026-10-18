# Wise vs wasteful abstention (Inspect eval)

An [Inspect](https://inspect.aisi.org.uk/) task that measures whether a model abstains on the problems it would likely get wrong. It reports how often the model abstains, and also whether those abstentions are well targeted.

## What it measures

The model gets a math problem and the prompt "Solve the following problem step by step. Put your final answer in \boxed{}." Giving no boxed answer counts as abstaining (the project convention). With `-T idk=true` the prompt also offers `\boxed{IDK}`, and that counts as abstaining too.

Each problem is sampled K times (Inspect epochs, default 4). The task estimates each problem's difficulty from the model's own answered samples: a problem is **hard** if the accuracy among its answered samples is below 0.5. A problem the model never answers counts as hard. Metrics, all pooled over problems:

| Metric | Meaning |
|---|---|
| `accuracy` | correct samples / all samples (abstention counts as wrong) |
| `abstention_rate` | abstained samples / all samples |
| `accuracy_when_answering` | correct / answered samples |
| `wise_fraction` | share of abstentions that fall on hard problems |
| `random_wise_fraction` | share of problems that are hard: the wise fraction of an abstainer that ignores difficulty |
| `wise_lift` | `wise_fraction - random_wise_fraction` |
| `abstention_auroc` | AUROC of per-problem abstention count against the hard label (0.5 = no signal) |
| `n_problems` | problems scored |

Correctness uses `math_verify` (pinned) on the last `\boxed{}` of the completion. Extraction and equivalence code is copied from `src/asym_rewards/_boxed.py` and `src/asym_rewards/reward.py` in this repo.

## Run

```
pip install -r evals/abstention/requirements.txt
inspect eval evals/abstention/wise_abstention.py@wise_abstention_math  --model <provider/model> --limit 200
inspect eval evals/abstention/wise_abstention.py@wise_abstention_gsm8k --model <provider/model> --limit 200
```

Tasks: `wise_abstention_math` (MATH test, 5000 problems, `DigitalLearningGmbH/MATH-lighteval`, the mirror the project used) and `wise_abstention_gsm8k` (GSM8K test, 1319 problems). Task parameters via `-T`: `epochs` (default 4; `--epochs N` also works), `idk` (default false), `temperature` (default 0.7), `max_tokens` (2048 for MATH, 1024 for GSM8K). `--limit N` takes the first N problems.

Local open model, no API: `--model hf/Qwen/Qwen2.5-0.5B-Instruct -M device=mps` (needs torch, transformers, accelerate). Unit tests: `cd evals/abstention && pytest`.

## Interpreting

- `wise_lift > 0` and `abstention_auroc > 0.5`: abstentions concentrate on problems the model usually gets wrong.
- `wise_lift` near 0 and AUROC near 0.5: abstaining is unrelated to difficulty. More abstention then just costs accuracy.
- Compare models at similar `abstention_rate`; wise fraction and AUROC change with the abstention level.
- `wise_fraction` is NaN if the model never abstains. `abstention_auroc` is NaN if all problems are hard or all are easy.

## Limitations

- **Self-referenced difficulty.** Difficulty comes from the evaluated model's own answered samples. The project's analysis (research/POST.md, section 7) used the base model's samples as the reference, so abstention could not influence the difficulty label. Self-reference is a variant: a model that abstains on everything it is unsure about leaves only its confident samples to define difficulty, and a problem with no answers is labeled hard by construction. This pushes `wise_fraction` up for heavy abstainers. Use the same reference model across compared models if you need a fair comparison.
- Difficulty is noisy at small K. With K=4, the accuracy among answered samples takes few values, and ties at 0.5 are labeled easy.
- No boxed answer is treated as abstention, so truncated or malformed outputs count as abstentions. Set `max_tokens` high enough that reasoning is not cut off. A 512-token cap on a 0.5B model mostly measures truncation.
- Equivalence is whatever `math_verify` accepts. Gold answers for MATH are the last `\boxed{}` in the reference solution.
- Sampling at temperature 0.7 is required to get variation across epochs; at temperature 0 all K samples are identical.
- Only math with a single final answer is covered.
