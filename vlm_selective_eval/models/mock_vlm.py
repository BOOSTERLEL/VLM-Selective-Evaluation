"""Deterministic mock VLM adapter for offline demos and tests."""

from __future__ import annotations

from pathlib import Path
import json
import random
from typing import Any

from vlm_selective_eval.constants import ANSWERABLE, UNANSWERABLE
from vlm_selective_eval.config import ModelConfig
from .base import BaseVLMAdapter


def _parse_image_semantics(image_path: str) -> tuple[bool, str]:
    stem = Path(image_path).stem
    parts = stem.split("__")
    if len(parts) >= 3:
        is_answerable = parts[1] == "answerable"
        answer = parts[2].replace("_", " ")
        return is_answerable, answer
    return True, "unknown"


class MockVLMAdapter(BaseVLMAdapter):
    """Mock adapter with configurable formatting and error behavior."""

    def __init__(self, config: ModelConfig, seed: int) -> None:
        self.config = config
        self.rng = random.Random(seed)

    def _predict(self, image_path: str) -> tuple[str, str, float]:
        is_answerable, expected_answer = _parse_image_semantics(image_path)
        if is_answerable:
            if self.rng.random() < self.config.answer_error_rate:
                status = UNANSWERABLE
                answer = ""
                conf = round(self.rng.uniform(0.2, 0.6), 3)
            else:
                status = ANSWERABLE
                if self.rng.random() < self.config.answer_error_rate:
                    answer = "incorrect"
                    conf = round(self.rng.uniform(0.4, 0.7), 3)
                else:
                    answer = expected_answer
                    conf = round(self.rng.uniform(0.75, 0.98), 3)
        else:
            if self.rng.random() < self.config.hallucination_rate:
                status = ANSWERABLE
                answer = expected_answer
                conf = round(self.rng.uniform(0.55, 0.92), 3)
            else:
                status = UNANSWERABLE
                answer = ""
                conf = round(self.rng.uniform(0.7, 0.97), 3)
        return status, answer, conf

    def _format_structured(self, status: str, answer: str, conf: float) -> str:
        payload = {"answer": answer, "status": status, "confidence": conf}
        if self.rng.random() < self.config.malformed_rate:
            return f"answer={answer}; status={status}; confidence={conf}"
        return json.dumps(payload, ensure_ascii=False)

    def _format_free_text(self, status: str, answer: str, conf: float) -> str:
        base = f"Answer: {answer}. Status: {status}. Confidence: {conf:.2f}."
        if self.rng.random() < self.config.free_text_missing_field_rate:
            if self.rng.random() < 0.5:
                return f"Answer: {answer}. Confidence: {conf:.2f}."
            return f"Answer: {answer}. Status: {status}."
        return base

    def _format_free_text_tagged(self, status: str, answer: str, conf: float) -> str:
        # Tagged mode is intentionally easier to parse than loose free-text.
        base = f"Answer: {answer}\nStatus: {status}\nConfidence: {conf:.2f}"
        tagged_missing_rate = self.config.free_text_missing_field_rate * 0.1
        if self.rng.random() < tagged_missing_rate:
            if self.rng.random() < 0.5:
                return f"Answer: {answer}\nStatus: {status}"
            return f"Answer: {answer}\nConfidence: {conf:.2f}"
        return base

    def generate(self, image_path: str, prompt: str, **kwargs: Any) -> str:
        del kwargs
        status, answer, conf = self._predict(image_path)
        if "Return ONLY valid JSON" in prompt:
            return self._format_structured(status=status, answer=answer, conf=conf)
        if (
            "exactly three lines" in prompt
            and "Status: ANSWERABLE or UNANSWERABLE" in prompt
            and "Confidence: <float in [0,1]>" in prompt
        ):
            return self._format_free_text_tagged(status=status, answer=answer, conf=conf)
        return self._format_free_text(status=status, answer=answer, conf=conf)
