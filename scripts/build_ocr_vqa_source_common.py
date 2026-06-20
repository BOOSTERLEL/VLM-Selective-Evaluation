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
        for key in ["answer", "text", "label", "value", "content"]:
            value = raw.get(key)
            if value:
                return str(value).strip()
        return ""
    if isinstance(raw, (list, tuple)):
        values = [_extract_answer_text(item) for item in raw]
        values = [value for value in values if value]
        return values[0] if values else ""
    return str(raw).strip()


def majority_answer(answers: list[Any]) -> str:
    values = [norm_text(_extract_answer_text(item)) for item in answers if _extract_answer_text(item)]
    if not values:
        return ""
    return Counter(values).most_common(1)[0][0]


def _rank_answer_candidates(record: dict[str, Any]) -> list[str]:
    answers_raw = record.get("answers")
    if not isinstance(answers_raw, list):
        answers_raw = []
    answers = [_extract_answer_text(item) for item in answers_raw if _extract_answer_text(item)]

    single_answer = _extract_answer_text(record.get("answer"))
    if single_answer:
        answers.append(single_answer)

    ranked = []
    seen: set[str] = set()
    majority = majority_answer(answers)
    if majority:
        ranked.append(majority)
        seen.add(majority)

    counts = Counter(norm_text(answer) for answer in answers if norm_text(answer))
    for normalized, _ in counts.most_common():
        if normalized not in seen:
            ranked.append(normalized)
            seen.add(normalized)
    return ranked


def extract_answer_list(record: dict[str, Any]) -> list[str]:
    answers_raw = record.get("answers")
    if isinstance(answers_raw, list):
        answers = [_extract_answer_text(item) for item in answers_raw if _extract_answer_text(item)]
    else:
        answers = []
    single_answer = _extract_answer_text(record.get("answer"))
    if single_answer and single_answer not in answers:
        answers.append(single_answer)
    return answers


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ["data", "annotations", "questions", "samples", "dataset"]:
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                return [item for item in value.values() if isinstance(item, dict)]
        return [item for item in payload.values() if isinstance(item, dict)]
    raise ValueError("Dataset input must be a JSON list or mapping.")


def candidate_ids(record: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in [
        "question_id",
        "questionId",
        "qa_id",
        "qaId",
        "ucsf_document_id",
        "document_id",
        "image_id",
        "imageId",
        "image_local_name",
        "image_name",
        "image",
        "file_name",
        "file_path",
        "id",
    ]:
        value = record.get(key)
        if value is None:
            continue
        text = str(value)
        ids.append(text)
        ids.append(Path(text).stem)
    out: list[str] = []
    seen: set[str] = set()
    for value in ids:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def resolve_image_path(record: dict[str, Any], image_dir: Path) -> Path | None:
    raw_values = []
    for key in [
        "image_path",
        "image",
        "image_id",
        "imageId",
        "image_name",
        "image_local_name",
        "file_name",
        "file_path",
        "document_id",
    ]:
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
            candidates.append(image_dir / raw_path.stem)
        if raw_path.suffix:
            continue
        for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF"]:
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
            return [int(round(float(box["x1"]))), int(round(float(box["y1"]))), int(round(float(box["x2"]))), int(round(float(box["y2"])))]
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
        for key in ["polygon", "boundingBox", "bounding_box", "bbox", "box", "points", "vertices", "quad"]:
            value = box.get(key)
            if value is None:
                continue
            return _box_to_xyxy(value, image_width=image_width, image_height=image_height)

    if isinstance(box, list):
        if len(box) == 4 and all(isinstance(item, (int, float)) for item in box):
            x1, y1, x2, y2 = [float(item) for item in box]
            if x2 <= 1 and y2 <= 1 and image_width > 1 and image_height > 1:
                x1 *= image_width
                y1 *= image_height
                x2 *= image_width
                y2 *= image_height
            return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
        if all(isinstance(item, (int, float)) for item in box):
            return _polygon_to_xyxy([float(item) for item in box])
        if box and isinstance(box[0], (list, tuple)) and len(box[0]) >= 2:
            flat: list[float] = []
            for item in box:
                flat.extend([float(item[0]), float(item[1])])
            return _polygon_to_xyxy(flat)
        if box and isinstance(box[0], dict):
            flat = []
            for item in box:
                if "x" in item and "y" in item:
                    flat.extend([float(item["x"]), float(item["y"])])
            return _polygon_to_xyxy(flat)
    return None


def _normalize_xyxy(box: list[int], image_width: int, image_height: int) -> list[int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), image_width - 1))
    y1 = max(0, min(int(y1), image_height - 1))
    x2 = max(x1 + 1, min(int(x2), image_width))
    y2 = max(y1 + 1, min(int(y2), image_height))
    return [x1, y1, x2, y2]


def _append_entry(entries: list[dict[str, Any]], text: str, raw_box: Any, image_width: int, image_height: int) -> None:
    text = str(text).strip()
    if not text:
        return
    box = _box_to_xyxy(raw_box, image_width=image_width, image_height=image_height)
    if box is None:
        return
    entries.append({"text": text, "box": _normalize_xyxy(box, image_width, image_height)})


def extract_ocr_entries(payload: Any, image_width: int, image_height: int) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    entries: list[dict[str, Any]] = []

    for key in ["ocr_info", "ocr", "ocr_entries", "ocr_results", "words", "tokens"]:
        value = payload.get(key)
        if not isinstance(value, list):
            continue
        if key == "tokens":
            boxes = payload.get("ocr_boxes") or payload.get("ocr_bboxes") or payload.get("boxes")
            if isinstance(boxes, list):
                for token, raw_box in zip(value, boxes):
                    _append_entry(entries, str(token), raw_box, image_width, image_height)
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            text = ""
            for text_key in ["word", "text", "content", "ocr_text", "transcription", "token", "label"]:
                item_value = item.get(text_key)
                if item_value:
                    text = str(item_value)
                    break
            raw_box = None
            for box_key in ["boundingBox", "bounding_box", "bbox", "box", "polygon", "points", "vertices", "quad"]:
                if box_key in item:
                    raw_box = item[box_key]
                    break
            if raw_box is None:
                raw_box = item
            _append_entry(entries, text, raw_box, image_width, image_height)

    recognition_results = payload.get("recognitionResults")
    if isinstance(recognition_results, list):
        for page in recognition_results:
            if not isinstance(page, dict):
                continue
            for line in page.get("lines", []) or []:
                if not isinstance(line, dict):
                    continue
                line_text = line.get("text") or line.get("content")
                _append_entry(entries, str(line_text or ""), line.get("boundingBox"), image_width, image_height)
                for word in line.get("words", []) or []:
                    if not isinstance(word, dict):
                        continue
                    _append_entry(
                        entries,
                        str(word.get("text") or word.get("content") or ""),
                        word.get("boundingBox"),
                        image_width,
                        image_height,
                    )

    pages = payload.get("pages")
    if isinstance(pages, list):
        for page in pages:
            if not isinstance(page, dict):
                continue
            page_width = int(round(float(page.get("width", image_width) or image_width)))
            page_height = int(round(float(page.get("height", image_height) or image_height)))
            for word in page.get("words", []) or []:
                if not isinstance(word, dict):
                    continue
                raw_box = word.get("polygon") or word.get("boundingBox") or word.get("bounding_box")
                _append_entry(
                    entries,
                    str(word.get("content") or word.get("text") or ""),
                    raw_box,
                    page_width,
                    page_height,
                )
            for line in page.get("lines", []) or []:
                if not isinstance(line, dict):
                    continue
                raw_box = line.get("polygon") or line.get("boundingBox") or line.get("bounding_box")
                _append_entry(
                    entries,
                    str(line.get("content") or line.get("text") or ""),
                    raw_box,
                    page_width,
                    page_height,
                )

    analyze_result = payload.get("analyzeResult")
    if isinstance(analyze_result, dict):
        entries.extend(extract_ocr_entries(analyze_result, image_width=image_width, image_height=image_height))

    return entries


def _build_ocr_lookup(payload: Any) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = {}
    for record in records_from_payload(payload):
        image_width = int(record.get("image_width", record.get("width", 1)) or 1)
        image_height = int(record.get("image_height", record.get("height", 1)) or 1)
        entries = extract_ocr_entries(record, image_width=image_width, image_height=image_height)
        if not entries:
            continue
        for key in candidate_ids(record):
            lookup[key] = entries
    return lookup


def _build_ocr_dir_index(ocr_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in ocr_dir.rglob("*.json"):
        stem = path.stem
        if stem.endswith(".ocr"):
            stem = Path(stem).stem
        index.setdefault(stem, path)
    return index


def _load_ocr_entries_from_dir(
    *,
    record: dict[str, Any],
    image_path: Path,
    image_width: int,
    image_height: int,
    ocr_dir_index: dict[str, Path],
) -> list[dict[str, Any]]:
    candidates = candidate_ids(record)
    candidates.extend([image_path.stem, image_path.name])
    seen: set[str] = set()
    for candidate in candidates:
        stem = Path(candidate).stem
        if stem in seen:
            continue
        seen.add(stem)
        path = ocr_dir_index.get(stem)
        if not path:
            continue
        payload = load_json(path)
        entries = extract_ocr_entries(payload, image_width=image_width, image_height=image_height)
        if entries:
            return entries
    return []


def find_answer_span(answer: str, ocr_tokens: list[str]) -> tuple[int, int] | None:
    answer_tokens = norm_text(answer).split()
    normalized_ocr = [norm_text(token) for token in ocr_tokens]
    n_tokens = len(answer_tokens)
    if n_tokens == 0:
        return None
    for start in range(len(normalized_ocr) - n_tokens + 1):
        if normalized_ocr[start:start + n_tokens] == answer_tokens:
            return start, start + n_tokens
    if n_tokens == 1:
        for index, token in enumerate(normalized_ocr):
            if answer_tokens[0] and answer_tokens[0] in token:
                return index, index + 1
    return None


def merge_boxes(boxes: list[list[int]]) -> list[list[int]]:
    xs1 = [box[0] for box in boxes]
    ys1 = [box[1] for box in boxes]
    xs2 = [box[2] for box in boxes]
    ys2 = [box[3] for box in boxes]
    return [[min(xs1), min(ys1), max(xs2), max(ys2)]]


def build_ocr_source_records(
    *,
    dataset_slug: str,
    annotations_payload: Any,
    image_dir: Path,
    ocr_payload: Any | None = None,
    ocr_dir: Path | None = None,
    max_samples: int = 0,
    question_keys: tuple[str, ...] = ("question", "query"),
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records = records_from_payload(annotations_payload)
    if max_samples > 0:
        records = records[:max_samples]

    ocr_lookup = _build_ocr_lookup(ocr_payload) if ocr_payload is not None else {}
    ocr_dir_index = _build_ocr_dir_index(ocr_dir) if ocr_dir is not None and ocr_dir.exists() else {}

    out: list[dict[str, Any]] = []
    stats = {
        "skipped_no_image": 0,
        "skipped_no_question": 0,
        "skipped_no_answer": 0,
        "skipped_no_ocr": 0,
        "skipped_no_match": 0,
    }

    for idx, record in enumerate(tqdm(records, desc=f"building {dataset_slug} source")):
        question = ""
        for key in question_keys:
            value = record.get(key)
            if value:
                question = str(value).strip()
                break
        if not question:
            stats["skipped_no_question"] += 1
            continue

        answers = extract_answer_list(record)
        ranked_answers = _rank_answer_candidates(record)
        if not answers and not ranked_answers:
            stats["skipped_no_answer"] += 1
            continue

        image_path = resolve_image_path(record, image_dir=image_dir)
        if image_path is None:
            stats["skipped_no_image"] += 1
            continue

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        ocr_entries = extract_ocr_entries(record, image_width=image_width, image_height=image_height)
        if not ocr_entries and ocr_dir_index:
            ocr_entries = _load_ocr_entries_from_dir(
                record=record,
                image_path=image_path,
                image_width=image_width,
                image_height=image_height,
                ocr_dir_index=ocr_dir_index,
            )
        if not ocr_entries and ocr_lookup:
            for key in candidate_ids(record) + [image_path.stem]:
                ocr_entries = ocr_lookup.get(key, [])
                if ocr_entries:
                    break
        if not ocr_entries:
            stats["skipped_no_ocr"] += 1
            continue

        tokens = [entry["text"] for entry in ocr_entries]
        chosen_answer = ""
        chosen_span: tuple[int, int] | None = None
        for candidate in ranked_answers:
            span = find_answer_span(candidate, tokens)
            if span is not None:
                chosen_answer = candidate
                chosen_span = span
                break
        if chosen_span is None:
            stats["skipped_no_match"] += 1
            continue

        start, end = chosen_span
        selected = [ocr_entries[i]["box"] for i in range(start, end)]
        if not selected:
            stats["skipped_no_match"] += 1
            continue

        question_id = record.get("questionId", record.get("question_id", record.get("id", idx)))
        out.append(
            {
                "id": f"{dataset_slug}_{question_id}",
                "image_path": str(image_path),
                "question": question,
                "answers": answers or [chosen_answer],
                "evidence_boxes": merge_boxes(selected),
            }
        )

    return out, stats
