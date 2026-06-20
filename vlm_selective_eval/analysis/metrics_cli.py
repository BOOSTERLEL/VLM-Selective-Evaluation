"""Generate protocol metrics artifacts from prediction JSONL."""

from __future__ import annotations

import argparse

from vlm_selective_eval.pipeline import evaluate_predictions_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute quadrant rates and selective curves.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL path.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--n-bins", type=int, default=10, help="Calibration bin count.")
    args = parser.parse_args()

    evaluate_predictions_file(
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        n_calibration_bins=args.n_bins,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
