"""Shared ANLS-style official scoring helpers for OCR-VQA datasets."""

from __future__ import annotations

from typing import Any


def normalize_anls_answer(text: str | None) -> str:
    """Normalize answer for ANLS.

    ANLS is case-insensitive but space-sensitive, so we only strip outer
    whitespace and lowercase the string.
    """
    if text is None:
        return ""
    return str(text).strip().lower()


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if left_char == right_char else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def anls_score(
    prediction: str | None,
    answers: list[str],
    threshold: float = 0.5,
) -> float:
    """Compute ANLS score against a list of reference answers."""
    if not answers:
        return 0.0

    pred = normalize_anls_answer(prediction)
    best = 0.0
    for answer in answers:
        gold = normalize_anls_answer(answer)
        max_len = max(len(pred), len(gold), 1)
        normalized_distance = levenshtein_distance(pred, gold) / max_len
        score = 1.0 - normalized_distance if normalized_distance < threshold else 0.0
        if score > best:
            best = score
    return float(best)


def evaluate_anls_from_predictions(
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Re-evaluate existing prediction rows using ANLS."""
    if not rows:
        raise ValueError("No prediction rows provided.")

    answerable_rows = [
        row for row in rows if str(row.get("ground_truth_answerability", "")).upper() == "ANSWERABLE"
    ]
    if not answerable_rows:
        raise ValueError("No answerable rows found in predictions.")

    scores: list[float] = []
    missing_multi_answers = 0
    total_reference_answers = 0
    for row in answerable_rows:
        metadata = row.get("metadata") or {}
        answers = metadata.get("official_answers") or metadata.get("answers") or []
        if not isinstance(answers, list):
            answers = []
        answers = [str(x).strip() for x in answers if str(x).strip()]
        if not answers:
            missing_multi_answers += 1
            answers = [str(row.get("ground_truth_answer", ""))]
        total_reference_answers += len(answers)
        scores.append(anls_score(row.get("parsed_answer"), answers, threshold=threshold))

    mean_score = float(sum(scores) / len(scores))
    return {
        "official_metric_name": metric_name,
        "official_score_mean": mean_score,
        "official_anls_mean": mean_score,
        "n_answerable": len(answerable_rows),
        "multi_answer_available_rate": float((len(answerable_rows) - missing_multi_answers) / len(answerable_rows)),
        "used_fallback_single_answer_count": int(missing_multi_answers),
        "avg_reference_answer_count": float(total_reference_answers / len(answerable_rows)),
        "anls_threshold": float(threshold),
    }


def evaluate_anls_metrics_with_standard_schema(
    rows: list[dict[str, Any]],
    n_calibration_bins: int = 10,
) -> dict[str, Any]:
    """Compute ANLS-scored metrics using the standard metrics schema."""
    from .metrics import evaluate_predictions

    return evaluate_predictions(rows, n_calibration_bins=n_calibration_bins)
