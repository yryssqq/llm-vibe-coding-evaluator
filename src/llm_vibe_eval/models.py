from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Complexity(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ErrorType(str, Enum):
    NONE = "none"
    SYNTAX_ERROR = "SyntaxError"
    INDENTATION_ERROR = "IndentationError"
    ASSERTION_ERROR = "AssertionError"
    TIMEOUT = "Timeout"
    RUNTIME_ERROR = "RuntimeError"
    UNSAFE_CODE = "UnsafeCode"
    EMPTY_RESPONSE = "EmptyResponse"


@dataclass
class Task:
    task_id: str
    prompt: str
    complexity: Complexity
    entry_point: str
    test_code: str
    tags: list[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    task_id: str
    model: str
    raw_response: str
    code: str | None
    attempt: int = 1
    error: str | None = None


@dataclass
class EvaluationResult:
    task_id: str
    model: str
    complexity: Complexity
    passed: bool
    error_type: ErrorType
    error_message: str
    duration_seconds: float
    unsafe_patterns: list[str] = field(default_factory=list)
    generated_code: str = ""
