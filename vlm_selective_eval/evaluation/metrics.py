"""Aggregate metrics, protocol-aligned selective curves, and calibration stats."""

from __future__ import annotations

from typing import Any

import numpy as np

from vlm_selective_eval.constants import (
    ANSWERABLE,
    QUADRANTS,
    QUADRANT_ANSWERABLE_CORRECT,
    QUADRANT_ANSWERABLE_WRONG,
    QUADRANT_UNANSWERABLE_ABSTAIN,
    QUADRANT_UNANSWERABLE_ASSERT,
    UNANSWERABLE,
    VALID_STATUSES,
)
from .normalize import normalize_answer
from .anls_official_common import anls_score
from .textvqa_official import textvqa_soft_accuracy

MISSING_STATUS_POLICY = ANSWERABLE


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _flatten_metrics_row(metrics: dict[str, Any]) -> dict[str, Any]:
    quadrant_counts = metrics["quadrant_counts"]
    quadrant_props = metrics["quadrant_proportions"]
    return {
        "model_name": metrics["model_name"],
        "prompt_mode": metrics["prompt_mode"],
        "task": metrics["task"],
        "n_samples": metrics["n_samples"],
        "answerable_correct_count": quadrant_counts["answerable_correct"],
        "answerable_wrong_count": quadrant_counts["answerable_wrong"],
        "unanswerable_abstain_count": quadrant_counts["unanswerable_abstain"],
        "unanswerable_assert_count": quadrant_counts["unanswerable_assert"],
        "answerable_correct_prop": quadrant_props["answerable_correct"],
        "answerable_wrong_prop": quadrant_props["answerable_wrong"],
        "unanswerable_abstain_prop": quadrant_props["unanswerable_abstain"],
        "unanswerable_assert_prop": quadrant_props["unanswerable_assert"],
        "unnecessary_abstention_rate": metrics["unnecessary_abstention_rate"],
        "hallucinatory_assertion_rate": metrics["hallucinatory_assertion_rate"],
        "coverage": metrics["coverage"],
        "risk": metrics["risk"],
        "global_coverage": metrics["global_coverage"],
        "global_risk": metrics["global_risk"],
        "parse_success_rate": metrics["parse_success_rate"],
        "format_validity_rate": metrics["format_validity_rate"],
        "status_missing_or_unreliable_rate": metrics["status_missing_or_unreliable_rate"],
        "confidence_missing_or_invalid_rate": metrics["confidence_missing_or_invalid_rate"],
        "answer_missing_rate": metrics["answer_missing_rate"],
        "answerable_soft_score_mean": metrics["answerable_soft_score_mean"],
        "calibration_ece": metrics["calibration"]["ece"],
        "calibration_brier": metrics["calibration"]["brier_score"],
        "parse_fail_status_policy": metrics["parse_fail_status_policy"],
    }


def _extract_reference_answers(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata") or {}
    answers = metadata.get("answers") or metadata.get("official_answers") or []
    if not isinstance(answers, list):
        return []
    return [str(x) for x in answers if str(x).strip()]


def _task_uses_textvqa_scoring(row: dict[str, Any]) -> bool:
    task = str(row.get("task", "")).strip().lower()
    source_kind = str((row.get("evidence_metadata") or {}).get("source_kind", "")).strip().lower()
    return "textvqa" in task or source_kind == "textvqa"


def _task_uses_stvqa_scoring(row: dict[str, Any]) -> bool:
    task = str(row.get("task", "")).strip().lower()
    source_kind = str((row.get("evidence_metadata") or {}).get("source_kind", "")).strip().lower()
    return (
        "stvqa" in task
        or "st_vqa" in task
        or "docvqa" in task
        or "infographicvqa" in task
        or "infographicsvqa" in task
        or source_kind in {"stvqa", "docvqa", "infographicvqa"}
    )


def _status_reliable(row: dict[str, Any]) -> bool:
    status = str(row.get("parsed_status", "")).strip().upper()
    default = status in VALID_STATUSES
    return bool(row.get("status_reliable", default)) and default


def _confidence_reliable(row: dict[str, Any]) -> bool:
    raw = row.get("parsed_confidence")
    try:
        conf = float(raw)
    except (TypeError, ValueError):
        conf = None
    default = conf is not None and 0.0 <= conf <= 1.0
    return bool(row.get("confidence_reliable", default)) and default


def _resolved_status_for_scoring(row: dict[str, Any]) -> str:
    status = str(row.get("parsed_status", "")).strip().upper()
    if status in VALID_STATUSES and _status_reliable(row):
        return status
    return MISSING_STATUS_POLICY


def _resolved_confidence_for_curve(row: dict[str, Any]) -> float:
    if not _confidence_reliable(row):
        return 0.0
    conf = float(row.get("parsed_confidence"))
    if conf < 0.0:
        return 0.0
    if conf > 1.0:
        return 1.0
    return conf


def _answer_soft_credit(row: dict[str, Any], pred_status: str | None = None) -> float:
    gt_answerability = row.get("ground_truth_answerability")
    resolved_status = pred_status or _resolved_status_for_scoring(row)
    pred_answer = row.get("parsed_answer")
    gt_answer = row.get("ground_truth_answer", "")
    if gt_answerability != ANSWERABLE or resolved_status != ANSWERABLE:
        return 0.0

    answers = _extract_reference_answers(row)
    if answers:
        if _task_uses_textvqa_scoring(row):
            return float(textvqa_soft_accuracy(pred_answer, answers))
        if _task_uses_stvqa_scoring(row):
            return float(anls_score(pred_answer, answers))
        normalized_pred = normalize_answer(pred_answer)
        return float(any(normalized_pred == normalize_answer(answer) for answer in answers))
    return float(normalize_answer(pred_answer) == normalize_answer(gt_answer))


def _map_protocol_quadrant(row: dict[str, Any], pred_status: str) -> str:
    gt_answerability = row.get("ground_truth_answerability")
    if gt_answerability == ANSWERABLE:
        score = _answer_soft_credit(row, pred_status=pred_status)
        if pred_status == ANSWERABLE and score > 0.0:
            return QUADRANT_ANSWERABLE_CORRECT
        return QUADRANT_ANSWERABLE_WRONG
    if gt_answerability == UNANSWERABLE:
        if pred_status == UNANSWERABLE:
            return QUADRANT_UNANSWERABLE_ABSTAIN
        return QUADRANT_UNANSWERABLE_ASSERT
    raise ValueError(f"Invalid ground-truth answerability label: {gt_answerability}")


def _covered_error(row: dict[str, Any], pred_status: str | None = None) -> float:
    gt_answerability = row.get("ground_truth_answerability")
    resolved_status = pred_status or _resolved_status_for_scoring(row)
    if resolved_status != ANSWERABLE:
        return 0.0
    if gt_answerability != ANSWERABLE:
        return 1.0
    return float(1.0 - _answer_soft_credit(row, pred_status=resolved_status))


def _confidence_points(rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        pred_status = _resolved_status_for_scoring(row)
        if pred_status != ANSWERABLE:
            continue
        conf_f = _resolved_confidence_for_curve(row)
        gt_answerability = row.get("ground_truth_answerability")
        if gt_answerability == ANSWERABLE:
            correct = float(_answer_soft_credit(row, pred_status=pred_status))
        else:
            correct = 0.0
        points.append((conf_f, correct))
    return points


def _compute_calibration(points: list[tuple[float, float]], n_bins: int) -> dict[str, Any]:
    if not points:
        return {"ece": 0.0, "brier_score": 0.0, "n_points": 0, "bins": []}

    bins = []
    total = len(points)
    ece = 0.0
    brier = 0.0
    for conf, correct in points:
        brier += (conf - correct) ** 2
    brier /= total

    for bin_idx in range(n_bins):
        lo = bin_idx / n_bins
        hi = (bin_idx + 1) / n_bins
        in_bin = [p for p in points if (lo <= p[0] < hi) or (bin_idx == n_bins - 1 and p[0] == 1.0)]
        if not in_bin:
            bins.append(
                {
                    "bin_start": lo,
                    "bin_end": hi,
                    "count": 0,
                    "avg_confidence": 0.0,
                    "empirical_accuracy": 0.0,
                }
            )
            continue
        conf_arr = np.array([p[0] for p in in_bin], dtype=float)
        acc_arr = np.array([p[1] for p in in_bin], dtype=float)
        avg_conf = float(conf_arr.mean())
        avg_acc = float(acc_arr.mean())
        count = int(len(in_bin))
        ece += abs(avg_conf - avg_acc) * count / total
        bins.append(
            {
                "bin_start": lo,
                "bin_end": hi,
                "count": count,
                "avg_confidence": avg_conf,
                "empirical_accuracy": avg_acc,
            }
        )
    return {"ece": float(ece), "brier_score": float(brier), "n_points": total, "bins": bins}


def _compute_answerable_risk_coverage_curve(
    rows: list[dict[str, Any]],
    n_points: int = 21,
) -> list[dict[str, float]]:
    thresholds = np.linspace(0.0, 1.0, n_points)
    answerable_rows = [row for row in rows if row.get("ground_truth_answerability") == ANSWERABLE]
    n_total = len(answerable_rows)
    curve: list[dict[str, float]] = []
    for threshold in thresholds:
        selected = []
        for row in answerable_rows:
            pred_status = _resolved_status_for_scoring(row)
            if pred_status != ANSWERABLE:
                continue
            conf = _resolved_confidence_for_curve(row)
            if conf >= float(threshold):
                selected.append(row)
        n_selected = len(selected)
        errors = sum(_covered_error(row, pred_status=ANSWERABLE) for row in selected)
        coverage = _safe_div(n_selected, n_total)
        risk = _safe_div(errors, n_selected)
        curve.append(
            {
                "threshold": round(float(threshold), 4),
                "coverage": float(coverage),
                "risk": float(risk),
                "n_selected": float(n_selected),
            }
        )
    return curve


def _compute_unanswerable_assert_abstain_curve(
    rows: list[dict[str, Any]],
    n_points: int = 21,
) -> list[dict[str, float]]:
    thresholds = np.linspace(0.0, 1.0, n_points)
    unanswerable_rows = [row for row in rows if row.get("ground_truth_answerability") == UNANSWERABLE]
    n_total = len(unanswerable_rows)
    curve: list[dict[str, float]] = []
    for threshold in thresholds:
        n_assert = 0
        for row in unanswerable_rows:
            pred_status = _resolved_status_for_scoring(row)
            if pred_status != ANSWERABLE:
                continue
            conf = _resolved_confidence_for_curve(row)
            if conf >= float(threshold):
                n_assert += 1
        n_abstain = n_total - n_assert
        assert_rate = _safe_div(n_assert, n_total)
        abstain_rate = _safe_div(n_abstain, n_total)
        curve.append(
            {
                "threshold": round(float(threshold), 4),
                "assert_rate": float(assert_rate),
                "abstain_rate": float(abstain_rate),
                "n_assert": float(n_assert),
                "n_abstain": float(n_abstain),
            }
        )
    return curve


def score_prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach protocol-scoring fields per prediction row."""
    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        gt_answerability = row.get("ground_truth_answerability")
        pred_status = _resolved_status_for_scoring(row)
        answer_soft_score: float | None = None
        if gt_answerability == ANSWERABLE:
            answer_soft_score = _answer_soft_credit(row, pred_status=pred_status)
        covered_error = _covered_error(row, pred_status=pred_status)
        quadrant = _map_protocol_quadrant(row, pred_status=pred_status)

        item = dict(row)
        item["status_for_scoring"] = pred_status
        item["quadrant"] = quadrant
        item["answer_soft_score"] = answer_soft_score
        item["covered_error"] = float(covered_error)
        item["status_missing_or_unreliable"] = not _status_reliable(row)
        item["confidence_missing_or_invalid"] = not _confidence_reliable(row)
        scored_rows.append(item)
    return scored_rows


def evaluate_predictions(
    rows: list[dict[str, Any]],
    n_calibration_bins: int = 10,
) -> dict[str, Any]:
    """Compute protocol-aligned quadrant, selective, parsing, and calibration metrics."""
    if not rows:
        raise ValueError("No prediction rows provided for evaluation.")

    quadrant_counts = {k: 0 for k in QUADRANTS}
    n_answerable = 0
    n_unanswerable = 0
    unnecessary_abstentions = 0
    hallucinatory_assertions = 0
    parse_ok_count = 0
    format_ok_count = 0
    status_unreliable_or_missing = 0
    confidence_missing_or_invalid = 0
    answer_missing = 0
    n_answerable_covered = 0
    answerable_covered_errors = 0.0
    n_global_covered = 0
    global_covered_errors = 0.0
    answerable_soft_score_sum = 0.0
    task_values: set[str] = set()
    model_values: set[str] = set()
    prompt_values: set[str] = set()

    for row in rows:
        task_values.add(str(row.get("task", "unknown")))
        model_values.add(str(row.get("model_name", "unknown")))
        prompt_values.add(str(row.get("prompt_mode", "unknown")))
        gt_answerability = row.get("ground_truth_answerability")
        pred_status = _resolved_status_for_scoring(row)

        if bool(row.get("parse_ok", False)):
            parse_ok_count += 1
        if bool(row.get("format_ok", False)):
            format_ok_count += 1
        if not _status_reliable(row):
            status_unreliable_or_missing += 1
        if not _confidence_reliable(row):
            confidence_missing_or_invalid += 1
        if row.get("parsed_answer") is None:
            answer_missing += 1

        if pred_status == ANSWERABLE:
            n_global_covered += 1
            global_covered_errors += _covered_error(row, pred_status=pred_status)

        if gt_answerability == ANSWERABLE:
            n_answerable += 1
            score = _answer_soft_credit(row, pred_status=pred_status)
            answerable_soft_score_sum += score
            quadrant = _map_protocol_quadrant(row, pred_status=pred_status)
            if pred_status == ANSWERABLE:
                n_answerable_covered += 1
                answerable_covered_errors += float(1.0 - score)
            if quadrant == QUADRANT_ANSWERABLE_WRONG and pred_status == UNANSWERABLE:
                unnecessary_abstentions += 1
        elif gt_answerability == UNANSWERABLE:
            n_unanswerable += 1
            quadrant = _map_protocol_quadrant(row, pred_status=pred_status)
            if quadrant == QUADRANT_UNANSWERABLE_ASSERT:
                hallucinatory_assertions += 1
        else:
            raise ValueError(f"Invalid ground-truth answerability label: {gt_answerability}")
        quadrant_counts[quadrant] += 1

    n_samples = len(rows)
    quadrant_proportions = {k: _safe_div(v, n_samples) for k, v in quadrant_counts.items()}
    answerable_coverage = _safe_div(n_answerable_covered, n_answerable)
    answerable_risk = _safe_div(answerable_covered_errors, n_answerable_covered)
    global_coverage = _safe_div(n_global_covered, n_samples)
    global_risk = _safe_div(global_covered_errors, n_global_covered)
    calibration_points = _confidence_points(rows)
    calibration = _compute_calibration(calibration_points, n_bins=n_calibration_bins)
    answerable_curve = _compute_answerable_risk_coverage_curve(rows)
    unanswerable_curve = _compute_unanswerable_assert_abstain_curve(rows)

    summary = {
        "model_name": ",".join(sorted(model_values)),
        "prompt_mode": ",".join(sorted(prompt_values)),
        "task": ",".join(sorted(task_values)),
        "n_samples": n_samples,
        "quadrant_counts": quadrant_counts,
        "quadrant_proportions": quadrant_proportions,
        "unnecessary_abstention_rate": _safe_div(unnecessary_abstentions, n_answerable),
        "hallucinatory_assertion_rate": _safe_div(hallucinatory_assertions, n_unanswerable),
        "coverage": answerable_coverage,
        "risk": answerable_risk,
        "global_coverage": global_coverage,
        "global_risk": global_risk,
        "parse_success_rate": _safe_div(parse_ok_count, n_samples),
        "format_validity_rate": _safe_div(format_ok_count, n_samples),
        "status_missing_or_unreliable_rate": _safe_div(status_unreliable_or_missing, n_samples),
        "confidence_missing_or_invalid_rate": _safe_div(confidence_missing_or_invalid, n_samples),
        "answer_missing_rate": _safe_div(answer_missing, n_samples),
        "parse_fail_status_policy": f"default_to_{MISSING_STATUS_POLICY}",
        "answerable_soft_score_mean": _safe_div(answerable_soft_score_sum, n_answerable),
        "calibration": calibration,
        "answerable_risk_coverage_curve": answerable_curve,
        "coverage_risk_curve": answerable_curve,
        "unanswerable_assert_abstain_curve": unanswerable_curve,
    }
    summary["metrics_row"] = _flatten_metrics_row(summary)
    return summary
