"""Data schemas used across dataset, inference, and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class PairSample:
    """Single model input for paired answerable/unanswerable evaluation."""

    sample_id: str
    pair_id: str
    task: str
    image_path: str
    question: str
    ground_truth_answerability: str
    ground_truth_answer: str
    evidence_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PairSample":
        return cls(**data)


@dataclass
class ParsedOutput:
    """Parsed model output for both structured and free-text modes."""

    answer: Optional[str]
    status: Optional[str]
    confidence: Optional[float]
    parse_ok: bool
    format_ok: bool
    parser_name: str
    status_reliable: bool = False
    confidence_reliable: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionRecord:
    """Prediction row serialized to JSONL."""

    sample_id: str
    pair_id: str
    task: str
    model_name: str
    prompt_mode: str
    image_path: str
    question: str
    raw_output: str
    parsed_answer: Optional[str]
    parsed_status: Optional[str]
    parsed_confidence: Optional[float]
    parse_ok: bool
    format_ok: bool
    parser_name: str
    ground_truth_answerability: str
    ground_truth_answer: str
    status_reliable: bool = False
    confidence_reliable: bool = False
    parse_notes: list[str] = field(default_factory=list)
    evidence_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
