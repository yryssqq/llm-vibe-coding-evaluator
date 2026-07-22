# llm-vibe-coding-evaluator

[![CI](https://github.com/yryssqq/llm-vibe-coding-evaluator/actions/workflows/ci.yml/badge.svg)](https://github.com/yryssqq/llm-vibe-coding-evaluator/actions/workflows/ci.yml)

Measures what happens when you take LLM-generated code at face value.

## Abstract

"Vibe coding" means accepting model output with little or no review. The risks are usually described in words: hallucinated APIs, silent logic bugs, unsafe operations. This turns that into numbers. It feeds algorithmic tasks to a model, runs each generated solution in a sandbox against real tests, and reports pass rate, failure types, and how often the model reaches for filesystem/network/`eval` access it was never asked for.

Companion to my article, *Vibe Coding and the Future of Software Development* (Curieux Academic Journal, p. 155): https://www.curieuxacademicjournal.com/_files/ugd/99711c_00f1cef862a8433e9766f5b902b06f85.pdf

## Methodology

- **Tasks**: 18 algorithmic problems (`data/tasks.json`), 6 each at easy/medium/hard: strings, search, dynamic programming, graph shortest-path, backtracking. Each ships a prompt, an entry point, and assert-based tests.
- **Models**: `Qwen2.5-Coder-32B-Instruct` (code-specialized) vs. `Llama-3.1-8B-Instruct` (general, 4x smaller), same prompt, one sample per task.
- **Execution**: each reply is stripped to raw code, screened by an AST pass, then run in its own subprocess under a timeout and CPU/memory limits.
- **Aggregation**: results go to CSV, aggregated with pandas (pass@1, error types, unsafe rate), charts with seaborn.

## Empirical Results & Key Findings

| model | pass@1 | unsafe code |
|---|---|---|
| Qwen2.5-Coder-32B-Instruct | 18/18 (100%) | 0% |
| Llama-3.1-8B-Instruct | 17/18 (94%) | 6% |

![pass@1 by model](results/report/pass_rate_by_model.png)

![error type vs complexity](results/report/error_vs_complexity.png)

The code-specialized model got everything right. The general model's one miss is the interesting case. On the Dijkstra task Llama-3.1-8B wrote a correct solution but `import`ed `sys` just to use `sys.maxsize` as a starting "infinity". The safety screen flags any `sys` import, so it scored `UnsafeCode` instead of pass.

That is the finding, not a bug. The screen can't tell `sys.maxsize` from `sys.exit()`, so it blocks both. It trades false positives for safety on purpose. In a real no-review workflow, that `sys` import is exactly the kind of thing that ships unexamined. "It ran and passed the tests" is a weaker guarantee than it sounds.

## Sandbox

Generated code is never trusted. Before running, an AST pass rejects anything importing or calling `os`, `socket`, `subprocess`, `eval`, `__import__`, and the like, and logs it as `UnsafeCode` (how often a model reaches for that unprompted is part of what's measured). Code that passes still runs in its own subprocess, in a temp dir, under a timeout and CPU/memory limits.

Good enough for benchmarking known-shape tasks, not a hardened boundary. For anything adversarial, run it in a container.

## Files

```
data/tasks.json         18 tasks with prompts and assert-based tests
src/llm_vibe_eval/       loading, LLM client, sandbox, analytics
run_evaluation.py        tasks -> model -> sandbox -> results/*.csv
analyze_results.py       results -> summary.json + charts
```

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=...   # or HF_TOKEN for HuggingFace models

python run_evaluation.py --provider openai --model gpt-4o-mini
python analyze_results.py results/gpt-4o-mini.csv
```

Tests: `pytest`.

## License

MIT
