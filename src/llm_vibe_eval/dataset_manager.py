from __future__ import annotations

import json
from pathlib import Path

from .models import Complexity, Task


class DatasetError(ValueError):
    pass


def load_tasks(path: str | Path) -> list[Task]:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset file not found: {path}")

    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise DatasetError("Dataset file must contain a JSON list of tasks")

    tasks: list[Task] = []
    seen_ids: set[str] = set()
    for i, item in enumerate(raw):
        missing = {"task_id", "prompt", "complexity", "entry_point", "test_code"} - item.keys()
        if missing:
            raise DatasetError(f"Task at index {i} is missing fields: {sorted(missing)}")
        if item["task_id"] in seen_ids:
            raise DatasetError(f"Duplicate task_id: {item['task_id']}")
        seen_ids.add(item["task_id"])

        try:
            complexity = Complexity(item["complexity"])
        except ValueError as exc:
            raise DatasetError(
                f"Task {item['task_id']} has invalid complexity '{item['complexity']}'"
            ) from exc

        tasks.append(
            Task(
                task_id=item["task_id"],
                prompt=item["prompt"],
                complexity=complexity,
                entry_point=item["entry_point"],
                test_code=item["test_code"],
                tags=item.get("tags", []),
            )
        )
    return tasks


def filter_by_complexity(tasks: list[Task], complexity: Complexity) -> list[Task]:
    return [t for t in tasks if t.complexity == complexity]
