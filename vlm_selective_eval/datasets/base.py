"""Base dataset adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from vlm_selective_eval.config import DatasetConfig
from vlm_selective_eval.schemas import PairSample


class DatasetAdapter(ABC):
    """Build paired answerable/unanswerable samples."""

    @abstractmethod
    def build_pairs(
        self,
        config: DatasetConfig,
        output_root: Path,
        seed: int,
    ) -> list[PairSample]:
        """Construct paired samples and return them in-memory."""
