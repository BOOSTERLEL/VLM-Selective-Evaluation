"""Quadrant mapping logic."""

from __future__ import annotations

from vlm_selective_eval.constants import (
    ANSWERABLE,
    QUADRANT_ANSWERABLE_CORRECT,
    QUADRANT_ANSWERABLE_WRONG,
    QUADRANT_UNANSWERABLE_ABSTAIN,
    QUADRANT_UNANSWERABLE_ASSERT,
    UNANSWERABLE,
)
from .normalize import normalize_answer


def is_answer_correct(predicted_answer: str | None, ground_truth_answer: str | None) -> bool:
    return normalize_answer(predicted_answer) == normalize_answer(ground_truth_answer)


def map_quadrant(
    gt_answerability: str,
    gt_answer: str | None,
    pred_status: str | None,
    pred_answer: str | None,
) -> str:
    """Map sample into one of four required quadrants."""
    if gt_answerability == ANSWERABLE:
        if pred_status == ANSWERABLE and is_answer_correct(pred_answer, gt_answer):
            return QUADRANT_ANSWERABLE_CORRECT
        return QUADRANT_ANSWERABLE_WRONG
    if gt_answerability == UNANSWERABLE:
        if pred_status == UNANSWERABLE:
            return QUADRANT_UNANSWERABLE_ABSTAIN
        return QUADRANT_UNANSWERABLE_ASSERT
    raise ValueError(f"Invalid ground-truth answerability label: {gt_answerability}")
