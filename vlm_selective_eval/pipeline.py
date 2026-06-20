"""End-to-end pipeline helpers used by CLI and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vlm_selective_eval.analysis import save_all_plots
from vlm_selective_eval.config import PipelineConfig, load_config
from vlm_selective_eval.datasets import create_dataset_adapter
from vlm_selective_eval.evaluation import (
    evaluate_docvqa_official_from_predictions,
    evaluate_docvqa_official_metrics_with_standard_schema,
    evaluate_predictions,
    evaluate_infographicvqa_official_from_predictions,
    evaluate_infographicvqa_official_metrics_with_standard_schema,
    evaluate_stvqa_official_from_predictions,
    evaluate_stvqa_official_metrics_with_standard_schema,
    score_prediction_rows,
    evaluate_textvqa_official_from_predictions,
    evaluate_textvqa_official_metrics_with_standard_schema,
)
from vlm_selective_eval.io_utils import read_jsonl, write_csv, write_json, write_jsonl
from vlm_selective_eval.models import create_model_adapter
from vlm_selective_eval.parsing import parse_free_text_output, parse_structured_output
from vlm_selective_eval.prompting import build_prompt, is_structured_prompt_mode
from vlm_selective_eval.schemas import PairSample, PredictionRecord


def _default_pairs_path(cfg: PipelineConfig) -> Path:
    return Path(cfg.experiment.output_dir) / "pairs" / f"{cfg.dataset.mode}_pairs.jsonl"


def _default_predictions_path(cfg: PipelineConfig) -> Path:
    model_slug = cfg.model.model_name.replace("/", "_")
    return (
        Path(cfg.experiment.output_dir)
        / "predictions"
        / f"{cfg.dataset.mode}_{cfg.prompt.mode}_{model_slug}_predictions.jsonl"
    )


def _quadrant_rates_row(metrics: dict[str, Any]) -> dict[str, Any]:
    props = metrics["quadrant_proportions"]
    return {
        "n_samples": metrics["n_samples"],
        "answerable_correct_prop": props["answerable_correct"],
        "answerable_wrong_prop": props["answerable_wrong"],
        "unanswerable_abstain_prop": props["unanswerable_abstain"],
        "unanswerable_assert_prop": props["unanswerable_assert"],
        "uar": metrics["unnecessary_abstention_rate"],
        "har": metrics["hallucinatory_assertion_rate"],
    }


def _write_standard_eval_artifacts(
    *,
    out_dir: Path,
    metrics: dict[str, Any],
    scored_rows: list[dict[str, Any]],
) -> None:
    write_json(out_dir / "metrics.json", metrics)
    write_csv(out_dir / "metrics.csv", [metrics["metrics_row"]])
    write_csv(out_dir / "quadrant_rates.csv", [_quadrant_rates_row(metrics)])
    write_csv(out_dir / "risk_coverage.csv", metrics["answerable_risk_coverage_curve"])
    # Backward-compatible alias.
    write_csv(out_dir / "coverage_risk_curve.csv", metrics["coverage_risk_curve"])
    write_csv(out_dir / "assert_abstain.csv", metrics["unanswerable_assert_abstain_curve"])
    write_csv(out_dir / "calibration_bins.csv", metrics["calibration"]["bins"])
    write_jsonl(out_dir / "scored.jsonl", scored_rows)


def build_pairs_from_config(config_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Build paired samples and save JSONL."""
    cfg = load_config(config_path)
    adapter = create_dataset_adapter(cfg.dataset.mode)
    output_root = Path(cfg.experiment.output_dir)
    pairs = adapter.build_pairs(config=cfg.dataset, output_root=output_root, seed=cfg.experiment.seed)
    destination = Path(output_path) if output_path else _default_pairs_path(cfg)
    write_jsonl(destination, [pair.to_dict() for pair in pairs])
    return destination


def run_inference_from_config(
    config_path: str | Path,
    pairs_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Run model inference over a pair file and save prediction JSONL."""
    cfg = load_config(config_path)
    model = create_model_adapter(cfg.model, seed=cfg.experiment.seed)
    rows = read_jsonl(pairs_path)
    samples = [PairSample.from_dict(row) for row in rows]
    predictions: list[dict[str, Any]] = []
    for sample in samples:
        prompt = build_prompt(sample.question, cfg.prompt)
        raw_output = model.generate(
            image_path=sample.image_path,
            prompt=prompt,
            **cfg.model.generation_kwargs,
        )
        if is_structured_prompt_mode(cfg.prompt.mode):
            parsed = parse_structured_output(raw_output)
        else:
            parsed = parse_free_text_output(raw_output)
        record = PredictionRecord(
            sample_id=sample.sample_id,
            pair_id=sample.pair_id,
            task=sample.task,
            model_name=cfg.model.model_name,
            prompt_mode=cfg.prompt.mode,
            image_path=sample.image_path,
            question=sample.question,
            raw_output=raw_output,
            parsed_answer=parsed.answer,
            parsed_status=parsed.status,
            parsed_confidence=parsed.confidence,
            parse_ok=parsed.parse_ok,
            format_ok=parsed.format_ok,
            parser_name=parsed.parser_name,
            ground_truth_answerability=sample.ground_truth_answerability,
            ground_truth_answer=sample.ground_truth_answer,
            status_reliable=parsed.status_reliable,
            confidence_reliable=parsed.confidence_reliable,
            parse_notes=parsed.notes,
            evidence_metadata=sample.evidence_metadata,
            metadata=sample.metadata,
        )
        predictions.append(record.to_dict())
    destination = Path(output_path) if output_path else _default_predictions_path(cfg)
    write_jsonl(destination, predictions)
    return destination


def evaluate_predictions_file(
    predictions_path: str | Path,
    output_dir: str | Path,
    n_calibration_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate predictions and emit JSON + CSV artifacts."""
    rows = read_jsonl(predictions_path)
    metrics = evaluate_predictions(rows, n_calibration_bins=n_calibration_bins)
    scored_rows = score_prediction_rows(rows)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_standard_eval_artifacts(out_dir=out_dir, metrics=metrics, scored_rows=scored_rows)
    return metrics


def evaluate_textvqa_official_file(
    predictions_path: str | Path,
    output_dir: str | Path,
    n_calibration_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate TextVQA predictions with official and matrix-compatible metrics.

    Besides official soft-score outputs, this also writes protocol artifacts:
    `metrics.*`, quadrant rates, risk-coverage, assert-abstain, calibration bins,
    and per-sample `scored.jsonl`.
    """
    return _evaluate_official_file(
        predictions_path=predictions_path,
        output_dir=output_dir,
        n_calibration_bins=n_calibration_bins,
        artifact_prefix="textvqa",
        official_summary_fn=evaluate_textvqa_official_from_predictions,
        standard_metrics_fn=evaluate_textvqa_official_metrics_with_standard_schema,
    )


def evaluate_stvqa_official_file(
    predictions_path: str | Path,
    output_dir: str | Path,
    n_calibration_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate ST-VQA predictions with official ANLS and matrix-compatible metrics."""
    return _evaluate_official_file(
        predictions_path=predictions_path,
        output_dir=output_dir,
        n_calibration_bins=n_calibration_bins,
        artifact_prefix="stvqa",
        official_summary_fn=evaluate_stvqa_official_from_predictions,
        standard_metrics_fn=evaluate_stvqa_official_metrics_with_standard_schema,
    )


def evaluate_docvqa_official_file(
    predictions_path: str | Path,
    output_dir: str | Path,
    n_calibration_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate DocVQA predictions with official ANLS and matrix-compatible metrics."""
    return _evaluate_official_file(
        predictions_path=predictions_path,
        output_dir=output_dir,
        n_calibration_bins=n_calibration_bins,
        artifact_prefix="docvqa",
        official_summary_fn=evaluate_docvqa_official_from_predictions,
        standard_metrics_fn=evaluate_docvqa_official_metrics_with_standard_schema,
    )


def evaluate_infographicvqa_official_file(
    predictions_path: str | Path,
    output_dir: str | Path,
    n_calibration_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate InfographicVQA predictions with official ANLS and matrix-compatible metrics."""
    return _evaluate_official_file(
        predictions_path=predictions_path,
        output_dir=output_dir,
        n_calibration_bins=n_calibration_bins,
        artifact_prefix="infographicvqa",
        official_summary_fn=evaluate_infographicvqa_official_from_predictions,
        standard_metrics_fn=evaluate_infographicvqa_official_metrics_with_standard_schema,
    )


def _evaluate_official_file(
    *,
    predictions_path: str | Path,
    output_dir: str | Path,
    n_calibration_bins: int,
    artifact_prefix: str,
    official_summary_fn: Any,
    standard_metrics_fn: Any,
) -> dict[str, Any]:
    rows = read_jsonl(predictions_path)
    official_metrics = official_summary_fn(rows)
    standard_metrics = standard_metrics_fn(rows, n_calibration_bins=n_calibration_bins)
    scored_rows = score_prediction_rows(rows)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_json(out_dir / f"{artifact_prefix}_official_metrics.json", official_metrics)
    write_csv(out_dir / f"{artifact_prefix}_official_metrics.csv", [official_metrics])
    _write_standard_eval_artifacts(out_dir=out_dir, metrics=standard_metrics, scored_rows=scored_rows)

    merged = dict(official_metrics)
    merged["metrics_row"] = standard_metrics["metrics_row"]
    return merged


def plot_from_metrics_file(metrics_json_path: str | Path, output_dir: str | Path) -> dict[str, str]:
    """Load metrics JSON and write required plots."""
    import json

    metrics = json.loads(Path(metrics_json_path).read_text(encoding="utf-8"))
    return save_all_plots(metrics=metrics, output_dir=output_dir)


def evaluate_and_plot(
    predictions_path: str | Path,
    output_dir: str | Path,
    n_calibration_bins: int = 10,
) -> dict[str, Any]:
    """Convenience helper: evaluate predictions and generate plots."""
    metrics = evaluate_predictions_file(
        predictions_path=predictions_path,
        output_dir=output_dir,
        n_calibration_bins=n_calibration_bins,
    )
    plot_paths = save_all_plots(metrics=metrics, output_dir=Path(output_dir) / "plots")
    metrics["plot_paths"] = plot_paths
    return metrics
