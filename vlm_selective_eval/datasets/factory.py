"""Factory for dataset adapters."""

from __future__ import annotations

from .base import DatasetAdapter
from .docvqa import DocVQADatasetAdapter
from .gqa import GQADatasetAdapter
from .infographicvqa import InfographicVQADatasetAdapter
from .stvqa import STVQADatasetAdapter
from .synthetic import SyntheticDatasetAdapter
from .textvqa import TextVQADatasetAdapter


def create_dataset_adapter(mode: str) -> DatasetAdapter:
    mode_key = mode.lower().strip()
    if mode_key == "synthetic":
        return SyntheticDatasetAdapter()
    if mode_key in {"textvqa", "textvqa_mock", "textvqa_style"}:
        return TextVQADatasetAdapter()
    if mode_key in {"stvqa", "st_vqa", "scene_text_vqa"}:
        return STVQADatasetAdapter()
    if mode_key in {"docvqa", "doc_vqa", "document_vqa"}:
        return DocVQADatasetAdapter()
    if mode_key in {"infographicvqa", "infographicsvqa", "infographic_vqa"}:
        return InfographicVQADatasetAdapter()
    if mode_key in {"gqa", "gqa_mock", "gqa_simple"}:
        return GQADatasetAdapter()
    raise ValueError(f"Unsupported dataset mode: {mode}")
