import argparse
import json
import re
from collections import Counter
from pathlib import Path

from tqdm import tqdm


def norm_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def _extract_answer_text(raw):
    if isinstance(raw, dict):
        return str(raw.get("answer", "")).strip()
    return str(raw).strip()


def majority_answer(answers):
    vals = [norm_text(_extract_answer_text(a)) for a in answers if _extract_answer_text(a)]
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]


def box_to_xyxy(box, img_w, img_h):
    x = float(box["top_left_x"])
    y = float(box["top_left_y"])
    w = float(box["width"])
    h = float(box["height"])

    # 官方 OCR json 里 width/height 是相对比例；top_left_x/top_left_y 通常也是比例
    # 为稳妥起见：若值<=1，则按比例转像素；否则视为已是像素
    if x <= 1 and y <= 1 and w <= 1 and h <= 1:
        x1 = int(round(x * img_w))
        y1 = int(round(y * img_h))
        x2 = int(round((x + w) * img_w))
        y2 = int(round((y + h) * img_h))
    else:
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(x1 + 1, min(x2, img_w))
    y2 = max(y1 + 1, min(y2, img_h))
    return [x1, y1, x2, y2]


def merge_boxes(boxes):
    xs1 = [b[0] for b in boxes]
    ys1 = [b[1] for b in boxes]
    xs2 = [b[2] for b in boxes]
    ys2 = [b[3] for b in boxes]
    return [[min(xs1), min(ys1), max(xs2), max(ys2)]]


def find_answer_span(answer, ocr_tokens):
    """
    在 OCR token 序列中找与答案完全匹配的连续 span。
    例如 answer='gate 7'，ocr_tokens=['gate','7',...]
    """
    ans_tokens = norm_text(answer).split()
    ocr_norm = [norm_text(t) for t in ocr_tokens]
    n = len(ans_tokens)
    if n == 0:
        return None
    for i in range(len(ocr_norm) - n + 1):
        if ocr_norm[i:i+n] == ans_tokens:
            return i, i + n
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--ocr", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    with open(args.questions, "r", encoding="utf-8") as f:
        qdata = json.load(f)["data"]

    with open(args.ocr, "r", encoding="utf-8") as f:
        odata = json.load(f)["data"]

    ocr_map = {str(x["image_id"]): x for x in odata}
    image_dir = Path(args.image_dir)

    out = []
    skipped_no_ocr = 0
    skipped_no_match = 0
    skipped_no_image = 0

    iterable = qdata if args.max_samples <= 0 else qdata[:args.max_samples]

    for item in tqdm(iterable, desc="building textvqa source"):
        image_id = str(item["image_id"])
        qid = item["question_id"]
        question = item["question"]
        img_w = int(item["image_width"])
        img_h = int(item["image_height"])
        answers = item.get("answers", [])
        all_answers = [_extract_answer_text(a) for a in answers if _extract_answer_text(a)]

        answer = majority_answer(answers)
        if not answer:
            continue

        img_path = image_dir / f"{image_id}.jpg"
        if not img_path.exists():
            skipped_no_image += 1
            continue

        ocr_item = ocr_map.get(image_id)
        if ocr_item is None:
            skipped_no_ocr += 1
            continue

        ocr_tokens = ocr_item.get("ocr_tokens", [])
        ocr_info = ocr_item.get("ocr_info", [])

        span = find_answer_span(answer, ocr_tokens)
        if span is None:
            skipped_no_match += 1
            continue

        s, e = span
        selected = []
        for idx in range(s, e):
            if idx >= len(ocr_info):
                continue
            bbox = ocr_info[idx]["bounding_box"]
            selected.append(box_to_xyxy(bbox, img_w, img_h))

        if not selected:
            skipped_no_match += 1
            continue

        evidence_boxes = merge_boxes(selected)

        out.append({
            "id": f"textvqa_{qid}",
            "image_path": str(img_path),
            "question": question,
            "answers": all_answers,
            "evidence_boxes": evidence_boxes,
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"saved: {args.output}")
    print(f"num_records: {len(out)}")
    print(f"skipped_no_image: {skipped_no_image}")
    print(f"skipped_no_ocr: {skipped_no_ocr}")
    print(f"skipped_no_match: {skipped_no_match}")


if __name__ == "__main__":
    main()
