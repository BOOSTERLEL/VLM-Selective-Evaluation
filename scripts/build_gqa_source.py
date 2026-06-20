import argparse
import json
from pathlib import Path

from tqdm import tqdm


def obj_to_box(obj):
    x1 = int(obj["x"])
    y1 = int(obj["y"])
    x2 = x1 + int(obj["w"])
    y2 = y1 + int(obj["h"])
    return [x1, y1, x2, y2]


def dedup_boxes(boxes):
    out = []
    seen = set()
    for b in boxes:
        t = tuple(b)
        if t not in seen:
            seen.add(t)
            out.append(b)
    return out


def collect_object_ids(q):
    ids = []

    ann = q.get("annotations", {})
    for part in ["answer", "question", "fullAnswer"]:
        d = ann.get(part, {})
        if isinstance(d, dict):
            for _, obj_id in d.items():
                if obj_id:
                    ids.append(str(obj_id))

    # 如果 annotations 没有，再尝试 semantic argument 里的 "(12345)"
    for step in q.get("semantic", []):
        arg = str(step.get("argument", ""))
        # 取括号里的 object id
        import re
        ids.extend(re.findall(r"\((\d+)\)", arg))

    return list(dict.fromkeys(ids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--scene-graphs", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--only-balanced", action="store_true")
    args = parser.parse_args()

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = json.load(f)

    with open(args.scene_graphs, "r", encoding="utf-8") as f:
        scene_graphs = json.load(f)

    image_dir = Path(args.image_dir)
    items = list(questions.items())
    if args.max_samples > 0:
        items = items[:args.max_samples]

    out = []
    skipped_no_image = 0
    skipped_no_sg = 0
    skipped_no_box = 0

    for qid, q in tqdm(items, desc="building gqa source"):
        if args.only_balanced and not q.get("isBalanced", False):
            continue

        image_id = str(q["imageId"])
        img_path = image_dir / f"{image_id}.jpg"
        if not img_path.exists():
            skipped_no_image += 1
            continue

        sg = scene_graphs.get(image_id)
        if sg is None:
            skipped_no_sg += 1
            continue

        objects = sg.get("objects", {})
        object_ids = collect_object_ids(q)

        boxes = []
        for oid in object_ids:
            if oid in objects:
                boxes.append(obj_to_box(objects[oid]))

        boxes = dedup_boxes(boxes)
        if not boxes:
            skipped_no_box += 1
            continue

        out.append({
            "id": f"gqa_{qid}",
            "image_path": str(img_path),
            "question": q["question"],
            "answer": str(q["answer"]).strip().lower(),
            "evidence_boxes": boxes,
            "types": q.get("types", {}),
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"saved: {args.output}")
    print(f"num_records: {len(out)}")
    print(f"skipped_no_image: {skipped_no_image}")
    print(f"skipped_no_sg: {skipped_no_sg}")
    print(f"skipped_no_box: {skipped_no_box}")


if __name__ == "__main__":
    main()