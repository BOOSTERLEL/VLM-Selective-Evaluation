"""InfographicVQA official-style scoring utilities (ANLS)."""

from __future__ import annotations

from .anls_official_common import (
    anls_score as infographicvqa_anls_score,
    evaluate_anls_from_predictions,
    evaluate_anls_metrics_with_standard_schema,
)


def evaluate_infographicvqa_official_from_predictions(rows, threshold: float = 0.5):
    return evaluate_anls_from_predictions(
        rows,
        metric_name="anls",
        threshold=threshold,
    )


def evaluate_infographicvqa_official_metrics_with_standard_schema(rows, n_calibration_bins: int = 10):
    return evaluate_anls_metrics_with_standard_schema(rows, n_calibration_bins=n_calibration_bins)
