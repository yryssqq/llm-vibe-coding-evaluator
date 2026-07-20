from __future__ import annotations

import ast
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .models import Complexity, ErrorType, EvaluationResult

UNSAFE_IMPORT_MODULES = {
    "os", "sys", "subprocess", "socket", "shutil", "requests", "urllib",
    "ctypes", "multiprocessing", "threading", "importlib", "pickle",
    "marshal", "pty", "signal", "pathlib", "http", "ftplib", "telnetlib",
    "smtplib", "asyncio",
}
UNSAFE_CALL_NAMES = {"eval", "exec", "__import__", "open", "compile", "input"}

DEFAULT_TIMEOUT_SECONDS = 3.0
DEFAULT_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024


def static_safety_check(code: str) -> list[str]:
    violations: list[str] = []
    try:
        tree = ast.parse(code)
    except (SyntaxError, IndentationError):
        return []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in UNSAFE_IMPORT_MODULES:
                    violations.append(f"import '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in UNSAFE_IMPORT_MODULES:
                violations.append(f"from '{node.module}' import ...")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in UNSAFE_CALL_NAMES:
                violations.append(f"call to '{node.func.id}(...)'")
    return violations


def _preexec_limits(memory_limit_bytes: int, cpu_seconds: int):
    import resource

    is_macos = platform.system() == "Darwin"

    def _limit():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        if not is_macos:
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))

    return _limit


def _classify_stderr(stderr: str) -> tuple[ErrorType, str]:
    if not stderr.strip():
        return ErrorType.RUNTIME_ERROR, "Unknown failure (empty stderr)"
    last_line = stderr.strip().splitlines()[-1]
    for err_type in (ErrorType.SYNTAX_ERROR, ErrorType.INDENTATION_ERROR, ErrorType.ASSERTION_ERROR):
        if err_type.value in last_line:
            return err_type, last_line
    return ErrorType.RUNTIME_ERROR, last_line


def evaluate(
    task_id: str,
    model: str,
    complexity: Complexity,
    code: str | None,
    test_code: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES,
) -> EvaluationResult:
    if not code:
        return EvaluationResult(
            task_id=task_id,
            model=model,
            complexity=complexity,
            passed=False,
            error_type=ErrorType.EMPTY_RESPONSE,
            error_message="Model returned no extractable code",
            duration_seconds=0.0,
            generated_code="",
        )

    violations = static_safety_check(code)
    if violations:
        return EvaluationResult(
            task_id=task_id,
            model=model,
            complexity=complexity,
            passed=False,
            error_type=ErrorType.UNSAFE_CODE,
            error_message="; ".join(violations),
            duration_seconds=0.0,
            unsafe_patterns=violations,
            generated_code=code,
        )

    full_script = f"{code}\n\n{test_code}\n"

    with tempfile.TemporaryDirectory(prefix="vibe_sandbox_") as tmpdir:
        script_path = Path(tmpdir) / "candidate.py"
        script_path.write_text(full_script)

        kwargs = {}
        if platform.system() != "Windows":
            cpu_seconds = max(1, int(timeout) + 1)
            kwargs["preexec_fn"] = _preexec_limits(memory_limit_bytes, cpu_seconds)

        start = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
                **kwargs,
            )
            duration = time.monotonic() - start
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
            return EvaluationResult(
                task_id=task_id,
                model=model,
                complexity=complexity,
                passed=False,
                error_type=ErrorType.TIMEOUT,
                error_message=f"Execution exceeded {timeout}s timeout",
                duration_seconds=duration,
                generated_code=code,
            )

        if proc.returncode == 0:
            return EvaluationResult(
                task_id=task_id,
                model=model,
                complexity=complexity,
                passed=True,
                error_type=ErrorType.NONE,
                error_message="",
                duration_seconds=duration,
                generated_code=code,
            )

        error_type, message = _classify_stderr(proc.stderr)
        return EvaluationResult(
            task_id=task_id,
            model=model,
            complexity=complexity,
            passed=False,
            error_type=error_type,
            error_message=message,
            duration_seconds=duration,
            generated_code=code,
        )
