"""Generate plots from metrics.json artifacts."""

from __future__ import annotations

import argparse

from vlm_selective_eval.pipeline import plot_from_metrics_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Render protocol figures from metrics.json.")
    parser.add_argument("--metrics-json", required=True, help="Path to metrics.json.")
    parser.add_argument("--output-dir", required=True, help="Directory for plot files.")
    args = parser.parse_args()

    plot_from_metrics_file(metrics_json_path=args.metrics_json, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
