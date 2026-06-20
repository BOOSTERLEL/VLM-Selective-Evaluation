"""Configuration loading for CLI workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

import yaml


@dataclass
class ExperimentConfig:
    name: str = "pilot_v1"
    output_dir: str = "outputs/pilot_v1"
    seed: int = 7


@dataclass
class DatasetConfig:
    mode: str = "synthetic"
    data_root: str = "data"
    num_pairs: int = 8
    image_size: int = 256
    evidence_removal: str = "mask"
    source_path: str | None = None


@dataclass
class ModelConfig:
    adapter: str = "mock"
    model_name: str = "mock-vlm"
    device: str = "cpu"
    vram_gb: int | None = None
    max_new_tokens: int = 128
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    malformed_rate: float = 0.1
    answer_error_rate: float = 0.1
    hallucination_rate: float = 0.15
    free_text_missing_field_rate: float = 0.1


@dataclass
class PromptConfig:
    mode: str = "structured"
    include_system: bool = True
    system_text: str = (
        "You are a careful visual question answering assistant. "
        "Follow output instructions exactly."
    )


@dataclass
class EvaluationConfig:
    n_calibration_bins: int = 10


@dataclass
class PipelineConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _apply_dataclass_overrides(instance: Any, overrides: dict[str, Any]) -> Any:
    for key, value in overrides.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
    return instance


def _build_config(raw: dict[str, Any]) -> PipelineConfig:
    cfg = PipelineConfig()
    _apply_dataclass_overrides(cfg.experiment, raw.get("experiment", {}))
    _apply_dataclass_overrides(cfg.dataset, raw.get("dataset", {}))
    _apply_dataclass_overrides(cfg.model, raw.get("model", {}))
    _apply_dataclass_overrides(cfg.prompt, raw.get("prompt", {}))
    _apply_dataclass_overrides(cfg.evaluation, raw.get("evaluation", {}))
    return cfg


def load_config(path: str | Path) -> PipelineConfig:
    """Load YAML or JSON config from disk."""
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        raw = yaml.safe_load(text) or {}
    elif suffix == ".json":
        raw = json.loads(text)
    else:
        raise ValueError(f"Unsupported config extension: {suffix}")
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping object.")
    return _build_config(raw)
