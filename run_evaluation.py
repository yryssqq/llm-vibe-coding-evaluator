#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_vibe_eval import dataset_manager, llm_client, sandbox_evaluator  # noqa: E402
from llm_vibe_eval.models import EvaluationResult  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_evaluation")

RESULTS_FIELDS = [
    "task_id", "model", "complexity", "passed", "error_type",
    "error_message", "duration_seconds", "unsafe_patterns", "generated_code",
]


def _row(result: EvaluationResult) -> dict:
    return {
        "task_id": result.task_id,
        "model": result.model,
        "complexity": result.complexity.value,
        "passed": result.passed,
        "error_type": result.error_type.value,
        "error_message": result.error_message,
        "duration_seconds": round(result.duration_seconds, 4),
        "unsafe_patterns": "; ".join(result.unsafe_patterns),
        "generated_code": result.generated_code,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["huggingface", "openai"], required=True)
    parser.add_argument("--model", required=True, help="Model id, e.g. gpt-4o-mini or bigcode/starcoder2-15b")
    parser.add_argument("--dataset", default="data/tasks.json")
    parser.add_argument("--output", default=None, help="Defaults to results/<model>.csv")
    parser.add_argument("--samples", type=int, default=1, help="Generations per task (for pass@k)")
    parser.add_argument("--timeout", type=float, default=3.0, help="Sandbox execution timeout in seconds")
    parser.add_argument(
        "--api-key-env", default=None,
        help="Env var holding the API key (defaults to HF_TOKEN / OPENAI_API_KEY)",
    )
    args = parser.parse_args()

    default_env = "HF_TOKEN" if args.provider == "huggingface" else "OPENAI_API_KEY"
    api_key = os.environ.get(args.api_key_env or default_env)

    tasks = dataset_manager.load_tasks(args.dataset)
    logger.info("Loaded %d tasks from %s", len(tasks), args.dataset)

    backend = llm_client.build_backend(args.provider, args.model, api_key=api_key)

    safe_model_name = args.model.replace("/", "__")
    output_path = Path(args.output or f"results/{safe_model_name}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        writer.writeheader()

        for task in tasks:
            for sample_idx in range(args.samples):
                generation = backend.generate(task)
                result = sandbox_evaluator.evaluate(
                    task_id=task.task_id,
                    model=backend.name,
                    complexity=task.complexity,
                    code=generation.code,
                    test_code=task.test_code,
                    timeout=args.timeout,
                )
                writer.writerow(_row(result))
                f.flush()
                status = "PASS" if result.passed else result.error_type.value
                logger.info(
                    "[%d/%d] %s (sample %d/%d): %s",
                    tasks.index(task) + 1, len(tasks), task.task_id,
                    sample_idx + 1, args.samples, status,
                )

    logger.info("Wrote results to %s", output_path)


if __name__ == "__main__":
    main()
