"""Answer normalization for exact-match style comparison."""

from __future__ import annotations

import re


def normalize_answer(text: str | None) -> str:
    """Normalize short answers for robust exact match."""
    if text is None:
        return ""
    out = text.lower().strip()
    out = re.sub(r"[^\w\s]", " ", out)
    out = re.sub(r"\b(a|an|the)\b", " ", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()
