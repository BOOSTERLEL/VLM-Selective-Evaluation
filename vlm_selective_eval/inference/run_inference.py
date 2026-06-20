"""Run batched inference for paired protocol samples."""

from __future__ import annotations

import argparse

from vlm_selective_eval.pipeline import run_inference_from_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VLM inference on pair JSONL.")
    parser.add_argument("--config", required=True, help="Path to pipeline config.")
    parser.add_argument("--pairs", required=True, help="Path to pairs JSONL.")
    parser.add_argument("--output", default=None, help="Optional output prediction JSONL path.")
    args = parser.parse_args()

    path = run_inference_from_config(
        config_path=args.config,
        pairs_path=args.pairs,
        output_path=args.output,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
