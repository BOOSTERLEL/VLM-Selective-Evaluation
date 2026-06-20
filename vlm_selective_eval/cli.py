"""Command-line interface for pilot evaluation workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from vlm_selective_eval.config import load_config
from vlm_selective_eval.pipeline import (
    build_pairs_from_config,
    evaluate_docvqa_official_file,
    evaluate_and_plot,
    evaluate_infographicvqa_official_file,
    evaluate_predictions_file,
    evaluate_stvqa_official_file,
    evaluate_textvqa_official_file,
    plot_from_metrics_file,
    run_inference_from_config,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vlm-selective-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_cmd = subparsers.add_parser("build-pairs", help="Build paired samples JSONL.")
    build_cmd.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    build_cmd.add_argument("--output", default=None, help="Optional output JSONL path.")

    infer_cmd = subparsers.add_parser("run-inference", help="Run model inference on pairs.")
    infer_cmd.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    infer_cmd.add_argument("--pairs", required=True, help="Input pair JSONL path.")
    infer_cmd.add_argument("--output", default=None, help="Optional output prediction JSONL path.")

    eval_cmd = subparsers.add_parser("evaluate", help="Evaluate prediction JSONL.")
    eval_cmd.add_argument("--predictions", required=True, help="Prediction JSONL path.")
    eval_cmd.add_argument("--output-dir", required=True, help="Directory for metrics files.")
    eval_cmd.add_argument("--n-bins", type=int, default=10, help="Calibration bin count.")

    official_eval_cmd = subparsers.add_parser(
        "evaluate-textvqa-official",
        help="Evaluate TextVQA predictions with official multi-answer soft scoring.",
    )
    official_eval_cmd.add_argument("--predictions", required=True, help="Prediction JSONL path.")
    official_eval_cmd.add_argument("--output-dir", required=True, help="Directory for metrics files.")
    official_eval_cmd.add_argument("--n-bins", type=int, default=10, help="Calibration bin count.")

    stvqa_official_eval_cmd = subparsers.add_parser(
        "evaluate-stvqa-official",
        help="Evaluate ST-VQA predictions with official ANLS scoring.",
    )
    stvqa_official_eval_cmd.add_argument("--predictions", required=True, help="Prediction JSONL path.")
    stvqa_official_eval_cmd.add_argument("--output-dir", required=True, help="Directory for metrics files.")
    stvqa_official_eval_cmd.add_argument("--n-bins", type=int, default=10, help="Calibration bin count.")

    docvqa_official_eval_cmd = subparsers.add_parser(
        "evaluate-docvqa-official",
        help="Evaluate DocVQA predictions with official ANLS scoring.",
    )
    docvqa_official_eval_cmd.add_argument("--predictions", required=True, help="Prediction JSONL path.")
    docvqa_official_eval_cmd.add_argument("--output-dir", required=True, help="Directory for metrics files.")
    docvqa_official_eval_cmd.add_argument("--n-bins", type=int, default=10, help="Calibration bin count.")

    infographic_official_eval_cmd = subparsers.add_parser(
        "evaluate-infographicvqa-official",
        help="Evaluate InfographicVQA predictions with official ANLS scoring.",
    )
    infographic_official_eval_cmd.add_argument("--predictions", required=True, help="Prediction JSONL path.")
    infographic_official_eval_cmd.add_argument("--output-dir", required=True, help="Directory for metrics files.")
    infographic_official_eval_cmd.add_argument("--n-bins", type=int, default=10, help="Calibration bin count.")

    plot_cmd = subparsers.add_parser("plot", help="Generate required plots from metrics.json.")
    plot_cmd.add_argument("--metrics-json", required=True, help="Path to metrics.json.")
    plot_cmd.add_argument("--output-dir", required=True, help="Directory for plot files.")

    full_cmd = subparsers.add_parser(
        "evaluate-and-plot",
        help="Evaluate predictions then generate plots in one command.",
    )
    full_cmd.add_argument("--predictions", required=True, help="Prediction JSONL path.")
    full_cmd.add_argument("--output-dir", required=True, help="Directory for metrics and plots.")
    full_cmd.add_argument("--n-bins", type=int, default=10, help="Calibration bin count.")

    demo_cmd = subparsers.add_parser("demo", help="Run synthetic build->infer->evaluate pipeline.")
    demo_cmd.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    return parser


def entrypoint(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "build-pairs":
        path = build_pairs_from_config(config_path=args.config, output_path=args.output)
        print(f"Built pairs: {path}")
        return 0

    if args.command == "run-inference":
        path = run_inference_from_config(
            config_path=args.config,
            pairs_path=args.pairs,
            output_path=args.output,
        )
        print(f"Saved predictions: {path}")
        return 0

    if args.command == "evaluate":
        metrics = evaluate_predictions_file(
            predictions_path=args.predictions,
            output_dir=args.output_dir,
            n_calibration_bins=args.n_bins,
        )
        print(f"Evaluated {metrics['n_samples']} samples -> {Path(args.output_dir) / 'metrics.json'}")
        return 0

    if args.command == "evaluate-textvqa-official":
        metrics = evaluate_textvqa_official_file(
            predictions_path=args.predictions,
            output_dir=args.output_dir,
            n_calibration_bins=args.n_bins,
        )
        print(
            "Evaluated "
            f"{metrics['n_answerable']} answerable samples -> "
            f"{Path(args.output_dir) / 'textvqa_official_metrics.json'}"
        )
        return 0

    if args.command == "evaluate-stvqa-official":
        metrics = evaluate_stvqa_official_file(
            predictions_path=args.predictions,
            output_dir=args.output_dir,
            n_calibration_bins=args.n_bins,
        )
        print(
            "Evaluated "
            f"{metrics['n_answerable']} answerable samples -> "
            f"{Path(args.output_dir) / 'stvqa_official_metrics.json'}"
        )
        return 0

    if args.command == "evaluate-docvqa-official":
        metrics = evaluate_docvqa_official_file(
            predictions_path=args.predictions,
            output_dir=args.output_dir,
            n_calibration_bins=args.n_bins,
        )
        print(
            "Evaluated "
            f"{metrics['n_answerable']} answerable samples -> "
            f"{Path(args.output_dir) / 'docvqa_official_metrics.json'}"
        )
        return 0

    if args.command == "evaluate-infographicvqa-official":
        metrics = evaluate_infographicvqa_official_file(
            predictions_path=args.predictions,
            output_dir=args.output_dir,
            n_calibration_bins=args.n_bins,
        )
        print(
            "Evaluated "
            f"{metrics['n_answerable']} answerable samples -> "
            f"{Path(args.output_dir) / 'infographicvqa_official_metrics.json'}"
        )
        return 0

    if args.command == "plot":
        plot_paths = plot_from_metrics_file(
            metrics_json_path=args.metrics_json,
            output_dir=args.output_dir,
        )
        print(f"Saved plots: {plot_paths}")
        return 0

    if args.command == "evaluate-and-plot":
        metrics = evaluate_and_plot(
            predictions_path=args.predictions,
            output_dir=args.output_dir,
            n_calibration_bins=args.n_bins,
        )
        print(f"Evaluated and plotted {metrics['n_samples']} samples at {args.output_dir}")
        return 0

    if args.command == "demo":
        cfg = load_config(args.config)
        pair_path = build_pairs_from_config(config_path=args.config, output_path=None)
        pred_path = run_inference_from_config(
            config_path=args.config,
            pairs_path=pair_path,
            output_path=None,
        )
        out = Path(cfg.experiment.output_dir) / "evaluation"
        metrics = evaluate_and_plot(
            predictions_path=pred_path,
            output_dir=out,
            n_calibration_bins=cfg.evaluation.n_calibration_bins,
        )
        print(f"Demo completed. n_samples={metrics['n_samples']} output={out}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(entrypoint())
