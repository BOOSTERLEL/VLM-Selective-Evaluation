"""Factory for model adapters."""

from __future__ import annotations

from vlm_selective_eval.config import ModelConfig
from .base import BaseVLMAdapter
from .hf_adapter import HuggingFaceVLMAdapter
from .mock_vlm import MockVLMAdapter


def create_model_adapter(config: ModelConfig, seed: int) -> BaseVLMAdapter:
    adapter = config.adapter.lower().strip()
    if adapter == "mock":
        return MockVLMAdapter(config=config, seed=seed)
    if adapter in {"hf", "huggingface"}:
        return HuggingFaceVLMAdapter(config=config)
    raise ValueError(f"Unsupported model adapter: {config.adapter}")
