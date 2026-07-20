import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_vibe_eval import analytics_engine
from llm_vibe_eval.models import Complexity, ErrorType, EvaluationResult


def _make_results():
    return [
        EvaluationResult("t1", "model-a", Complexity.EASY, True, ErrorType.NONE, "", 0.1),
        EvaluationResult("t2", "model-a", Complexity.EASY, False, ErrorType.ASSERTION_ERROR, "boom", 0.1),
        EvaluationResult("t3", "model-a", Complexity.HARD, False, ErrorType.TIMEOUT, "slow", 3.0),
        EvaluationResult("t1", "model-b", Complexity.EASY, True, ErrorType.NONE, "", 0.2),
        EvaluationResult("t2", "model-b", Complexity.EASY, True, ErrorType.NONE, "", 0.2),
        EvaluationResult("t3", "model-b", Complexity.HARD, False, ErrorType.UNSAFE_CODE, "os", 0.0, ["import 'os'"]),
    ]


def test_results_to_dataframe_shape():
    df = analytics_engine.results_to_dataframe(_make_results())
    assert len(df) == 6
    assert set(df.columns) >= {"task_id", "model", "complexity", "passed", "error_type"}


def test_pass_at_1_overall():
    df = analytics_engine.results_to_dataframe(_make_results())
    assert analytics_engine.pass_at_1(df) == 3 / 6


def test_pass_at_1_by_model():
    df = analytics_engine.results_to_dataframe(_make_results())
    rates = analytics_engine.pass_at_1(df, ["model"]).set_index("model")["pass@1"]
    assert rates["model-a"] == 1 / 3
    assert rates["model-b"] == 2 / 3


def test_unsafe_code_rate():
    df = analytics_engine.results_to_dataframe(_make_results())
    assert analytics_engine.unsafe_code_rate(df) == 1 / 6


def test_heatmap_handles_zero_failures(tmp_path):
    results = [
        EvaluationResult("t1", "model-a", Complexity.EASY, True, ErrorType.NONE, "", 0.1),
        EvaluationResult("t2", "model-a", Complexity.HARD, True, ErrorType.NONE, "", 0.1),
    ]
    df = analytics_engine.results_to_dataframe(results)
    output = analytics_engine.plot_error_complexity_heatmap(df, tmp_path / "heatmap.png")
    assert output.exists()
