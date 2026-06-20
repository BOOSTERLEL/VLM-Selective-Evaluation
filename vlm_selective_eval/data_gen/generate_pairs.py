"""Generate paired samples for protocol evaluation."""

from __future__ import annotations

import argparse

from vlm_selective_eval.pipeline import build_pairs_from_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paired answerable/unanswerable samples.")
    parser.add_argument("--config", required=True, help="Path to pipeline config.")
    parser.add_argument("--output", default=None, help="Optional output JSONL path.")
    args = parser.parse_args()

    path = build_pairs_from_config(config_path=args.config, output_path=args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
