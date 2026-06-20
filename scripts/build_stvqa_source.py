import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, desc: str | None = None):
        del desc
        return iterable


def norm_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _extract_answer_text(raw: Any) -> str:
    if isinstance(raw, dict):
        for key in ["answer", "text", "label"]:
            value = raw.get(key)
            if value:
                return str(value).strip()
        return ""
    return str(raw).strip()


def majority_answer(answers: list[Any]) -> str:
    vals = [norm_text(_extract_answer_text(a)) for a in answers if _extract_answer_text(a)]
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ["data", "annotations", "questions", "samples", "dataset"]:
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
            if isinstance(value, dict):
                return [x for x in value.values() if isinstance(x, dict)]
        return [x for x in payload.values() if isinstance(x, dict)]
    raise ValueError("ST-VQA input must be a JSON list or mapping.")


def _candidate_ids(record: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in [
        "question_id",
        "questionId",
        "qa_id",
        "qaId",
        "id",
        "image_id",
        "imageId",
        "image",
        "file_name",
        "file_path",
    ]:
        value = record.get(key)
        if value is None:
            continue
        ids.append(str(value))
        ids.append(Path(str(value)).stem)
    out: list[str] = []
    seen: set[str] = set()
    for value in ids:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _resolve_image_path(record: dict[str, Any], image_dir: Path) -> Path | None:
    raw_values = []
    for key in ["image_path", "file_path", "image", "image_name", "file_name", "image_id", "imageId"]:
        value = record.get(key)
        if value:
            raw_values.append(str(value))

    candidates: list[Path] = []
    for raw in raw_values:
        raw_path = Path(raw)
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(image_dir / raw)
            candidates.append(image_dir / raw_path.name)
        if raw_path.suffix:
            continue
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            candidates.append(image_dir / f"{raw}{ext}")
            candidates.append(image_dir / f"{raw_path.stem}{ext}")

    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path
    return None


def _polygon_to_xyxy(points: list[float]) -> list[int] | None:
    if len(points) < 4 or len(points) % 2 != 0:
        return None
    xs = points[0::2]
    ys = points[1::2]
    return [int(round(min(xs))), int(round(min(ys))), int(round(max(xs))), int(round(max(ys)))]


def _box_to_xyxy(box: Any, image_width: int, image_height: int) -> list[int] | None:
    if box is None:
        return None

    if isinstance(box, dict):
        if all(key in box for key in ["x1", "y1", "x2", "y2"]):
            return [
                int(round(float(box["x1"]))),
                int(round(float(box["y1"]))),
                int(round(float(box["x2"]))),
                int(round(float(box["y2"]))),
            ]
        if all(key in box for key in ["x", "y", "w", "h"]):
            x1 = float(box["x"])
            y1 = float(box["y"])
            x2 = x1 + float(box["w"])
            y2 = y1 + float(box["h"])
            return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
        if all(key in box for key in ["left", "top", "width", "height"]):
            x1 = float(box["left"])
            y1 = float(box["top"])
            x2 = x1 + float(box["width"])
            y2 = y1 + float(box["height"])
            return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
        if all(key in box for key in ["top_left_x", "top_left_y", "width", "height"]):
            x = float(box["top_left_x"])
            y = float(box["top_left_y"])
            w = float(box["width"])
            h = float(box["height"])
            if x <= 1 and y <= 1 and w <= 1 and h <= 1:
                x1 = x * image_width
                y1 = y * image_height
                x2 = (x + w) * image_width
                y2 = (y + h) * image_height
            else:
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
            return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
        for key in ["points", "vertices", "polygon", "quad"]:
            value = box.get(key)
            if isinstance(value, list):
                if value and isinstance(value[0], dict):
                    flat: list[float] = []
                    for item in value:
                        if "x" in item and "y" in item:
                            flat.extend([float(item["x"]), float(item["y"])])
                    return _polygon_to_xyxy(flat)
                if value and isinstance(value[0], (list, tuple)) and len(value[0]) >= 2:
                    flat = []
                    for item in value:
                        flat.extend([float(item[0]), float(item[1])])
                    return _polygon_to_xyxy(flat)
                return _polygon_to_xyxy([float(x) for x in value])

    if isinstance(box, list):
        if len(box) == 4 and all(isinstance(x, (int, float)) for x in box):
            x1, y1, x2, y2 = [float(x) for x in box]
            if x2 <= 1 and y2 <= 1 and image_width > 1 and image_height > 1:
                x1 *= image_width
                y1 *= image_height
                x2 *= image_width
                y2 *= image_height
            return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
        if box and isinstance(box[0], (list, tuple)) and len(box[0]) >= 2:
            flat = []
            for item in box:
                flat.extend([float(item[0]), float(item[1])])
            return _polygon_to_xyxy(flat)
        if all(isinstance(x, (int, float)) for x in box):
            return _polygon_to_xyxy([float(x) for x in box])

    return None


def _normalize_xyxy(box: list[int], image_width: int, image_height: int) -> list[int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), image_width - 1))
    y1 = max(0, min(int(y1), image_height - 1))
    x2 = max(x1 + 1, min(int(x2), image_width))
    y2 = max(y1 + 1, min(int(y2), image_height))
    return [x1, y1, x2, y2]


def _extract_ocr_entries(record: dict[str, Any], image_width: int, image_height: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    info_like = []
    for key in ["ocr_info", "ocr", "ocr_entries", "ocr_results"]:
        value = record.get(key)
        if isinstance(value, list):
            info_like.extend(value)

    for item in info_like:
        if not isinstance(item, dict):
            continue
        text = ""
        for key in ["word", "text", "ocr_text", "transcription", "token", "label"]:
            value = item.get(key)
            if value:
                text = str(value).strip()
                break
        if not text:
            continue
        raw_box = None
        for key in ["bounding_box", "bbox", "box", "points", "vertices", "polygon", "quad"]:
            if key in item:
                raw_box = item[key]
                break
        box = _box_to_xyxy(raw_box, image_width=image_width, image_height=image_height)
        if box is None:
            box = _box_to_xyxy(item, image_width=image_width, image_height=image_height)
        if box is None:
            continue
        entries.append({"text": text, "box": _normalize_xyxy(box, image_width, image_height)})

    tokens = record.get("ocr_tokens")
    boxes = None
    for key in ["ocr_boxes", "ocr_bboxes", "ocr_token_boxes"]:
        value = record.get(key)
        if isinstance(value, list):
            boxes = value
            break
    if isinstance(tokens, list) and isinstance(boxes, list):
        for token, raw_box in zip(tokens, boxes):
            text = str(token).strip()
            if not text:
                continue
            box = _box_to_xyxy(raw_box, image_width=image_width, image_height=image_height)
            if box is None:
                continue
            entries.append({"text": text, "box": _normalize_xyxy(box, image_width, image_height)})

    return entries


def _build_ocr_lookup(payload: Any) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = {}
    for record in _records_from_payload(payload):
        image_width = int(record.get("image_width", record.get("width", 1)) or 1)
        image_height = int(record.get("image_height", record.get("height", 1)) or 1)
        entries = _extract_ocr_entries(record, image_width=image_width, image_height=image_height)
        if not entries:
            continue
        for key in _candidate_ids(record):
            lookup[key] = entries
    return lookup


def find_answer_span(answer: str, ocr_tokens: list[str]) -> tuple[int, int] | None:
    ans_tokens = norm_text(answer).split()
    norm_ocr = [norm_text(token) for token in ocr_tokens]
    n = len(ans_tokens)
    if n == 0:
        return None
    for start in range(len(norm_ocr) - n + 1):
        if norm_ocr[start:start + n] == ans_tokens:
            return start, start + n
    if n == 1:
        for idx, token in enumerate(norm_ocr):
            if ans_tokens[0] and ans_tokens[0] in token:
                return idx, idx + 1
    return None


def merge_boxes(boxes: list[list[int]]) -> list[list[int]]:
    xs1 = [b[0] for b in boxes]
    ys1 = [b[1] for b in boxes]
    xs2 = [b[2] for b in boxes]
    ys2 = [b[3] for b in boxes]
    return [[min(xs1), min(ys1), max(xs2), max(ys2)]]


def build_source_records(
    *,
    annotations_payload: Any,
    image_dir: Path,
    ocr_payload: Any | None = None,
    max_samples: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records = _records_from_payload(annotations_payload)
    if max_samples > 0:
        records = records[:max_samples]

    ocr_lookup = _build_ocr_lookup(ocr_payload) if ocr_payload is not None else {}
    out: list[dict[str, Any]] = []
    stats = {
        "skipped_no_image": 0,
        "skipped_no_question": 0,
        "skipped_no_answer": 0,
        "skipped_no_ocr": 0,
        "skipped_no_match": 0,
    }

    for idx, record in enumerate(tqdm(records, desc="building stvqa source")):
        question = str(record.get("question", "")).strip()
        if not question:
            stats["skipped_no_question"] += 1
            continue

        answers = record.get("answers")
        if not isinstance(answers, list):
            answers = []
        answers = [_extract_answer_text(x) for x in answers if _extract_answer_text(x)]
        answer = majority_answer(answers)
        if not answer:
            fallback = _extract_answer_text(record.get("answer", ""))
            answer = norm_text(fallback) if fallback else ""
            if fallback and not answers:
                answers = [fallback]
        if not answer:
            stats["skipped_no_answer"] += 1
            continue

        image_path = _resolve_image_path(record, image_dir=image_dir)
        if image_path is None:
            stats["skipped_no_image"] += 1
            continue

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        ocr_entries = _extract_ocr_entries(record, image_width=image_width, image_height=image_height)
        if not ocr_entries:
            for key in _candidate_ids(record):
                ocr_entries = ocr_lookup.get(key, [])
                if ocr_entries:
                    break
        if not ocr_entries:
            stats["skipped_no_ocr"] += 1
            continue

        tokens = [entry["text"] for entry in ocr_entries]
        span = find_answer_span(answer, tokens)
        if span is None:
            stats["skipped_no_match"] += 1
            continue

        start, end = span
        selected = [ocr_entries[i]["box"] for i in range(start, end)]
        if not selected:
            stats["skipped_no_match"] += 1
            continue

        qid = record.get("question_id", record.get("questionId", record.get("id", idx)))
        out.append(
            {
                "id": f"stvqa_{qid}",
                "image_path": str(image_path),
                "question": question,
                "answers": answers or [answer],
                "evidence_boxes": merge_boxes(selected),
            }
        )

    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, help="ST-VQA annotation JSON.")
    parser.add_argument("--image-dir", required=True, help="Directory containing ST-VQA images.")
    parser.add_argument("--output", required=True, help="Output source JSON path.")
    parser.add_argument("--ocr", default="", help="Optional OCR JSON if OCR is stored separately.")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    annotations_payload = _load_json(Path(args.annotations))
    ocr_payload = _load_json(Path(args.ocr)) if args.ocr else None

    out, stats = build_source_records(
        annotations_payload=annotations_payload,
        image_dir=Path(args.image_dir),
        ocr_payload=ocr_payload,
        max_samples=args.max_samples,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    print(f"saved: {output_path}")
    print(f"num_records: {len(out)}")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
