import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_vibe_eval.models import Complexity, ErrorType
from llm_vibe_eval.sandbox_evaluator import evaluate, static_safety_check

TEST_CODE = "assert add(2, 3) == 5\nassert add(-1, 1) == 0"


def test_passing_code():
    code = "def add(a, b):\n    return a + b"
    result = evaluate("t1", "test-model", Complexity.EASY, code, TEST_CODE)
    assert result.passed is True
    assert result.error_type == ErrorType.NONE


def test_assertion_failure():
    code = "def add(a, b):\n    return a - b"
    result = evaluate("t2", "test-model", Complexity.EASY, code, TEST_CODE)
    assert result.passed is False
    assert result.error_type == ErrorType.ASSERTION_ERROR


def test_syntax_error():
    code = "def add(a, b)\n    return a + b"
    result = evaluate("t3", "test-model", Complexity.EASY, code, TEST_CODE)
    assert result.passed is False
    assert result.error_type == ErrorType.SYNTAX_ERROR


def test_timeout():
    code = "def add(a, b):\n    while True:\n        pass"
    result = evaluate("t4", "test-model", Complexity.EASY, code, TEST_CODE, timeout=1.0)
    assert result.passed is False
    assert result.error_type == ErrorType.TIMEOUT


def test_empty_response():
    result = evaluate("t5", "test-model", Complexity.EASY, None, TEST_CODE)
    assert result.passed is False
    assert result.error_type == ErrorType.EMPTY_RESPONSE


def test_unsafe_code_blocked_before_execution():
    code = "import os\n\ndef add(a, b):\n    os.system('echo hi')\n    return a + b"
    result = evaluate("t6", "test-model", Complexity.EASY, code, TEST_CODE)
    assert result.passed is False
    assert result.error_type == ErrorType.UNSAFE_CODE
    assert "os" in result.error_message


def test_static_safety_check_flags_eval():
    violations = static_safety_check("def f():\n    return eval('1+1')")
    assert any("eval" in v for v in violations)


def test_static_safety_check_allows_clean_code():
    violations = static_safety_check("def f(x):\n    return x * 2")
    assert violations == []
