"""Parse raw model output into {answer, status, confidence}."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm_selective_eval.parsing import parse_free_text_output, parse_structured_output


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse a model output snippet.")
    parser.add_argument("--mode", choices=["structured", "free_text"], default="structured")
    parser.add_argument("--text", default=None, help="Raw text input. Use --file for multi-line input.")
    parser.add_argument("--file", default=None, help="Optional file path for raw output text.")
    args = parser.parse_args()

    if args.file:
        raw_text = Path(args.file).read_text(encoding="utf-8")
    elif args.text is not None:
        raw_text = args.text
    else:
        raise ValueError("Provide --text or --file.")

    parsed = (
        parse_structured_output(raw_text)
        if args.mode == "structured"
        else parse_free_text_output(raw_text)
    )
    print(json.dumps(parsed.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
