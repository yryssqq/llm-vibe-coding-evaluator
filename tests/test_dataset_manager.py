import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_vibe_eval import dataset_manager
from llm_vibe_eval.models import Complexity

DATASET_PATH = Path(__file__).parent.parent / "data" / "tasks.json"


def test_load_shipped_dataset():
    tasks = dataset_manager.load_tasks(DATASET_PATH)
    assert len(tasks) >= 15
    ids = [t.task_id for t in tasks]
    assert len(ids) == len(set(ids)), "task_ids must be unique"


def test_filter_by_complexity():
    tasks = dataset_manager.load_tasks(DATASET_PATH)
    easy = dataset_manager.filter_by_complexity(tasks, Complexity.EASY)
    assert all(t.complexity == Complexity.EASY for t in easy)
    assert len(easy) > 0


def test_missing_file_raises(tmp_path):
    with pytest.raises(dataset_manager.DatasetError):
        dataset_manager.load_tasks(tmp_path / "nope.json")


def test_missing_field_raises(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text('[{"task_id": "x", "prompt": "p", "complexity": "easy"}]')
    with pytest.raises(dataset_manager.DatasetError):
        dataset_manager.load_tasks(bad_file)
