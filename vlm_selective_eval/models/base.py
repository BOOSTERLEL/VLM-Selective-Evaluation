"""Base VLM adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseVLMAdapter(ABC):
    """Minimal VLM interface returning raw text only."""

    @abstractmethod
    def generate(self, image_path: str, prompt: str, **kwargs: Any) -> str:
        """Run inference and return a raw text completion."""
