import argparse
import json
from pathlib import Path

try:
    from scripts.build_ocr_vqa_source_common import build_ocr_source_records, load_json
except ModuleNotFoundError:
    from build_ocr_vqa_source_common import build_ocr_source_records, load_json


def build_source_records(
    *,
    annotations_payload,
    image_dir: Path,
    ocr_payload=None,
    ocr_dir: Path | None = None,
    max_samples: int = 0,
):
    return build_ocr_source_records(
        dataset_slug="infographicvqa",
        annotations_payload=annotations_payload,
        image_dir=image_dir,
        ocr_payload=ocr_payload,
        ocr_dir=ocr_dir,
        max_samples=max_samples,
        question_keys=("question", "query"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, help="InfographicVQA annotation JSON.")
    parser.add_argument("--image-dir", required=True, help="Directory containing infographic images.")
    parser.add_argument("--output", required=True, help="Output source JSON path.")
    parser.add_argument("--ocr", default="", help="Optional OCR JSON file.")
    parser.add_argument("--ocr-dir", default="", help="Optional directory containing one OCR JSON per image.")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    annotations_payload = load_json(Path(args.annotations))
    ocr_payload = load_json(Path(args.ocr)) if args.ocr else None
    ocr_dir = Path(args.ocr_dir) if args.ocr_dir else None

    records, stats = build_source_records(
        annotations_payload=annotations_payload,
        image_dir=Path(args.image_dir),
        ocr_payload=ocr_payload,
        ocr_dir=ocr_dir,
        max_samples=args.max_samples,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    print(f"saved: {output_path}")
    print(f"num_records: {len(records)}")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
