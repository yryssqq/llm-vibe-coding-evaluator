from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod

from .models import GenerationResult, Task

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "You are a code generation engine. You will be given a programming task. "
    "Respond with ONLY the raw Python code that solves it. "
    "Do not include explanations, markdown fences, or comments about your reasoning. "
    "Implement exactly the function signature described in the task."
)

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _extract_code(raw_response: str) -> str | None:
    if not raw_response or not raw_response.strip():
        return None
    match = CODE_FENCE_RE.search(raw_response)
    code = match.group(1) if match else raw_response
    code = code.strip()
    return code or None


class BaseLLMBackend(ABC):
    name: str

    @abstractmethod
    def _call(self, prompt: str) -> str: ...

    def generate(self, task: Task, max_retries: int = 3, backoff_seconds: float = 1.5) -> GenerationResult:
        full_prompt = f"{SYSTEM_INSTRUCTION}\n\nTask:\n{task.prompt}"
        last_error: str | None = None
        for attempt in range(1, max_retries + 1):
            try:
                raw_response = self._call(full_prompt)
                code = _extract_code(raw_response)
                return GenerationResult(
                    task_id=task.task_id,
                    model=self.name,
                    raw_response=raw_response,
                    code=code,
                    attempt=attempt,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "LLM call failed (task=%s, model=%s, attempt=%d/%d): %s",
                    task.task_id, self.name, attempt, max_retries, last_error,
                )
                if attempt < max_retries:
                    time.sleep(backoff_seconds * attempt)

        return GenerationResult(
            task_id=task.task_id,
            model=self.name,
            raw_response="",
            code=None,
            attempt=max_retries,
            error=last_error,
        )


class HuggingFaceBackend(BaseLLMBackend):
    def __init__(self, model_id: str, token: str | None = None, max_new_tokens: int = 512):
        from huggingface_hub import InferenceClient

        self.name = model_id
        self.max_new_tokens = max_new_tokens
        self._client = InferenceClient(model=model_id, token=token)

    def _call(self, prompt: str) -> str:
        response = self._client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
        )
        return response.choices[0].message.content or ""


class OpenAIBackend(BaseLLMBackend):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None, max_tokens: int = 512):
        from openai import OpenAI

        self.name = model
        self.max_tokens = max_tokens
        self._client = OpenAI(api_key=api_key)

    def _call(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""


def build_backend(provider: str, model: str, api_key: str | None = None) -> BaseLLMBackend:
    if provider == "huggingface":
        return HuggingFaceBackend(model_id=model, token=api_key)
    if provider == "openai":
        return OpenAIBackend(model=model, api_key=api_key)
    raise ValueError(f"Unknown provider '{provider}'. Use 'huggingface' or 'openai'.")
