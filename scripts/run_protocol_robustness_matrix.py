"""Run protocol-robustness mini matrix with prompt variants and non-zero temperatures."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from vlm_selective_eval.pipeline import (
    build_pairs_from_config,
    evaluate_docvqa_official_file,
    evaluate_and_plot,
    evaluate_infographicvqa_official_file,
    evaluate_stvqa_official_file,
    evaluate_textvqa_official_file,
    plot_from_metrics_file,
    run_inference_from_config,
)


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())


def _temperature_slug(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Matrix config root must be a mapping.")
    return raw


def _ensure_list(name: str, value: Any) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Matrix config `{name}` must be a non-empty list.")
    return value


def _ensure_positive_temperatures(raw_values: list[Any]) -> list[float]:
    values: list[float] = []
    for idx, raw in enumerate(raw_values):
        try:
            temperature = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid temperature at index {idx}: {raw!r}") from exc
        if temperature <= 0.0:
            raise ValueError(
                "Temperature must be > 0 for protocol robustness matrix. "
                f"Got {temperature} at index {idx}."
            )
        values.append(temperature)
    return values


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _runtime_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ["numpy", "matplotlib", "PyYAML", "Pillow", "torch", "transformers"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _build_run_config(
    *,
    run_id: str,
    run_output_dir: Path,
    seed: int,
    n_bins: int,
    dataset_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    prompt_cfg: dict[str, Any],
    default_system_text: str,
    temperature: float,
) -> dict[str, Any]:
    dataset = {
        "mode": dataset_cfg["mode"],
        "data_root": dataset_cfg.get("data_root", "data"),
        "num_pairs": dataset_cfg.get("num_pairs", 1000),
        "image_size": dataset_cfg.get("image_size", 224),
        "evidence_removal": dataset_cfg.get("evidence_removal", "mask"),
        "source_path": dataset_cfg.get("source_path"),
    }

    model_generation_kwargs = dict(model_cfg.get("generation_kwargs", {}))
    model_generation_kwargs["temperature"] = float(temperature)
    if "top_p" not in model_generation_kwargs:
        model_generation_kwargs["top_p"] = 1.0

    model = {
        "adapter": model_cfg.get("adapter", "hf"),
        "model_name": model_cfg["model_name"],
        "device": model_cfg.get("device", "cuda"),
        "vram_gb": model_cfg.get("vram_gb"),
        "max_new_tokens": model_cfg.get("max_new_tokens", 64),
        "generation_kwargs": model_generation_kwargs,
    }

    prompt = {
        "mode": str(prompt_cfg["mode"]),
        "include_system": bool(prompt_cfg.get("include_system", True)),
        "system_text": str(prompt_cfg.get("system_text", default_system_text)),
    }

    return {
        "experiment": {
            "name": run_id,
            "output_dir": str(run_output_dir).replace("\\", "/"),
            "seed": seed,
        },
        "dataset": dataset,
        "model": model,
        "prompt": prompt,
        "evaluation": {"n_calibration_bins": n_bins},
    }


def _summary_row(
    *,
    run_id: str,
    dataset_name: str,
    model_name: str,
    prompt_variant_name: str,
    prompt_mode: str,
    temperature: float,
    evaluation_mode_requested: str,
    evaluation_mode_effective: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    mr = metrics["metrics_row"]
    return {
        "run_id": run_id,
        "dataset": dataset_name,
        "model": model_name,
        "prompt_variant": prompt_variant_name,
        "prompt_mode": prompt_mode,
        "temperature": float(temperature),
        "evaluation_mode_requested": evaluation_mode_requested,
        "evaluation_mode_effective": evaluation_mode_effective,
        "n_samples": mr["n_samples"],
        "answerable_correct_prop": mr["answerable_correct_prop"],
        "answerable_wrong_prop": mr["answerable_wrong_prop"],
        "unanswerable_abstain_prop": mr["unanswerable_abstain_prop"],
        "unanswerable_assert_prop": mr["unanswerable_assert_prop"],
        "uar": mr["unnecessary_abstention_rate"],
        "har": mr["hallucinatory_assertion_rate"],
        "coverage": mr["coverage"],
        "risk": mr["risk"],
        "global_coverage": mr.get("global_coverage"),
        "global_risk": mr.get("global_risk"),
        "answerable_soft_score_mean": mr.get("answerable_soft_score_mean", 0.0),
        "parse_success_rate": mr["parse_success_rate"],
        "format_validity_rate": mr["format_validity_rate"],
        "status_missing_or_unreliable_rate": mr.get("status_missing_or_unreliable_rate"),
        "confidence_missing_or_invalid_rate": mr.get("confidence_missing_or_invalid_rate"),
        "answer_missing_rate": mr.get("answer_missing_rate"),
        "parse_fail_status_policy": mr.get("parse_fail_status_policy"),
        "ece": mr["calibration_ece"],
        "brier": mr["calibration_brier"],
        "official_metric_name": metrics.get("official_metric_name"),
        "official_score_mean": metrics.get(
            "official_score_mean",
            metrics.get("official_soft_score_mean", metrics.get("official_anls_mean")),
        ),
        "official_soft_score_mean": metrics.get("official_soft_score_mean"),
        "official_anls_mean": metrics.get("official_anls_mean"),
        "multi_answer_available_rate": metrics.get("multi_answer_available_rate"),
        "avg_reference_answer_count": metrics.get("avg_reference_answer_count"),
        "exactly_ten_reference_rate": metrics.get("exactly_ten_reference_rate"),
        "anls_threshold": metrics.get("anls_threshold"),
    }


def _official_eval_family(mode: str) -> str | None:
    mode_key = mode.lower().strip()
    if mode_key in {"textvqa", "textvqa_mock", "textvqa_style"}:
        return "textvqa"
    if mode_key in {"stvqa", "st_vqa", "scene_text_vqa"}:
        return "stvqa"
    if mode_key in {"docvqa", "doc_vqa", "document_vqa"}:
        return "docvqa"
    if mode_key in {"infographicvqa", "infographicsvqa", "infographic_vqa"}:
        return "infographicvqa"
    return None


def run_matrix(matrix_config_path: Path, dry_run: bool, evaluation_mode: str) -> dict[str, Any]:
    evaluation_mode_key = evaluation_mode.lower().strip()
    if evaluation_mode_key not in {"strict", "official"}:
        raise ValueError(f"Unsupported evaluation mode: {evaluation_mode}")

    raw = _read_yaml(matrix_config_path)
    datasets = _ensure_list("datasets", raw.get("datasets"))
    models = _ensure_list("models", raw.get("models"))
    prompt_variants = _ensure_list("prompt_variants", raw.get("prompt_variants"))
    temperatures = _ensure_positive_temperatures(_ensure_list("temperatures", raw.get("temperatures")))

    output_root = Path(raw.get("output_root", "outputs/protocol_robustness_matrix"))
    seed = int(raw.get("seed", 7))
    n_bins = int(raw.get("evaluation_n_bins", 10))
    prompt_system_text = str(
        raw.get(
            "prompt_system_text",
            "You are a careful visual question answering assistant. Follow output instructions exactly.",
        )
    )

    generated_cfg_dir = output_root / "generated_configs"
    summary_dir = output_root / "summary"
    summary_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []

    total_runs = len(datasets) * len(models) * len(prompt_variants) * len(temperatures)
    run_index = 0

    for dataset_cfg in datasets:
        dataset_name = str(dataset_cfg.get("name", dataset_cfg.get("mode", "dataset")))
        if "mode" not in dataset_cfg:
            raise ValueError(f"Dataset `{dataset_name}` missing required key: mode")
        for model_cfg in models:
            model_name = str(model_cfg.get("name", model_cfg.get("model_name", "model")))
            if "model_name" not in model_cfg:
                raise ValueError(f"Model `{model_name}` missing required key: model_name")
            for prompt_variant in prompt_variants:
                if not isinstance(prompt_variant, dict):
                    raise ValueError("Each prompt variant must be a mapping with at least `name` and `mode`.")
                prompt_name = str(prompt_variant.get("name", prompt_variant.get("mode", "prompt")))
                prompt_mode = str(prompt_variant.get("mode", "free_text"))
                if not prompt_name.strip():
                    raise ValueError("Prompt variant `name` cannot be empty.")
                for temperature in temperatures:
                    run_index += 1
                    temp_slug = _temperature_slug(float(temperature))
                    run_id = (
                        f"{_slug(dataset_name)}__{_slug(model_name)}__"
                        f"{_slug(prompt_name)}__temp{temp_slug}"
                    )
                    run_output_dir = output_root / "runs" / run_id
                    run_cfg = _build_run_config(
                        run_id=run_id,
                        run_output_dir=run_output_dir,
                        seed=seed,
                        n_bins=n_bins,
                        dataset_cfg=dataset_cfg,
                        model_cfg=model_cfg,
                        prompt_cfg=prompt_variant,
                        default_system_text=prompt_system_text,
                        temperature=float(temperature),
                    )
                    run_cfg_path = generated_cfg_dir / f"{run_id}.yaml"
                    _write_yaml(run_cfg_path, run_cfg)
                    plan_rows.append(
                        {
                            "run_id": run_id,
                            "dataset": dataset_name,
                            "model": model_name,
                            "prompt_variant": prompt_name,
                            "prompt_mode": prompt_mode,
                            "temperature": float(temperature),
                            "evaluation_mode_requested": evaluation_mode_key,
                            "config_path": str(run_cfg_path).replace("\\", "/"),
                        }
                    )

                    if dry_run:
                        print(f"[dry-run {run_index}/{total_runs}] {run_id}")
                        continue

                    print(f"[run {run_index}/{total_runs}] {run_id}")
                    pairs_path = build_pairs_from_config(config_path=run_cfg_path, output_path=None)
                    preds_path = run_inference_from_config(
                        config_path=run_cfg_path,
                        pairs_path=pairs_path,
                        output_path=None,
                    )
                    dataset_mode = str(dataset_cfg["mode"])
                    effective_eval_mode = evaluation_mode_key
                    official_family = _official_eval_family(dataset_mode)
                    if evaluation_mode_key == "official" and official_family is None:
                        effective_eval_mode = "strict"
                        print(
                            f"[info] {run_id}: official eval is only available for supported OCR-VQA datasets; "
                            f"fallback to strict for dataset mode `{dataset_mode}`."
                        )

                    if effective_eval_mode == "official":
                        eval_dir = run_output_dir / "evaluation"
                        if official_family == "stvqa":
                            metrics = evaluate_stvqa_official_file(
                                predictions_path=preds_path,
                                output_dir=eval_dir,
                                n_calibration_bins=n_bins,
                            )
                        elif official_family == "docvqa":
                            metrics = evaluate_docvqa_official_file(
                                predictions_path=preds_path,
                                output_dir=eval_dir,
                                n_calibration_bins=n_bins,
                            )
                        elif official_family == "infographicvqa":
                            metrics = evaluate_infographicvqa_official_file(
                                predictions_path=preds_path,
                                output_dir=eval_dir,
                                n_calibration_bins=n_bins,
                            )
                        else:
                            metrics = evaluate_textvqa_official_file(
                                predictions_path=preds_path,
                                output_dir=eval_dir,
                                n_calibration_bins=n_bins,
                            )
                        plot_paths = plot_from_metrics_file(
                            metrics_json_path=eval_dir / "metrics.json",
                            output_dir=eval_dir / "plots",
                        )
                        metrics["plot_paths"] = plot_paths
                    else:
                        metrics = evaluate_and_plot(
                            predictions_path=preds_path,
                            output_dir=run_output_dir / "evaluation",
                            n_calibration_bins=n_bins,
                        )
                    summary_rows.append(
                        _summary_row(
                            run_id=run_id,
                            dataset_name=dataset_name,
                            model_name=model_name,
                            prompt_variant_name=prompt_name,
                            prompt_mode=prompt_mode,
                            temperature=float(temperature),
                            evaluation_mode_requested=evaluation_mode_key,
                            evaluation_mode_effective=effective_eval_mode,
                            metrics=metrics,
                        )
                    )

    _write_csv(summary_dir / "run_plan.csv", plan_rows)

    summary_payload = {
        "matrix_config": str(matrix_config_path).replace("\\", "/"),
        "output_root": str(output_root).replace("\\", "/"),
        "total_runs": total_runs,
        "dry_run": dry_run,
        "evaluation_mode": evaluation_mode_key,
        "temperature_constraint": "temperature > 0",
        "seed": seed,
        "evaluation_n_bins": n_bins,
        "git_commit": _git_commit_hash(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": _runtime_package_versions(),
    }
    if dry_run:
        (summary_dir / "summary.json").write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary_payload

    _write_csv(summary_dir / "run_metrics_summary.csv", summary_rows)
    summary_payload["completed_runs"] = len(summary_rows)
    (summary_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run protocol robustness matrix from YAML config.")
    parser.add_argument("--matrix", required=True, help="Path to matrix YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Generate run plan only.")
    parser.add_argument(
        "--evaluation-mode",
        choices=["strict", "official"],
        default="official",
        help="Choose evaluation style: strict or dataset-specific official scoring.",
    )
    args = parser.parse_args()

    result = run_matrix(
        matrix_config_path=Path(args.matrix),
        dry_run=bool(args.dry_run),
        evaluation_mode=str(args.evaluation_mode),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
