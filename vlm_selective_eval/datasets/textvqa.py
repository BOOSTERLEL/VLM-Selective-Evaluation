"""TextVQA-style OCR subset adapter with evidence-removal pair building."""

from __future__ import annotations

from pathlib import Path
import json
from collections import Counter
import re

from PIL import Image, ImageDraw, ImageFont

from vlm_selective_eval.constants import ANSWERABLE, UNANSWERABLE
from vlm_selective_eval.config import DatasetConfig
from vlm_selective_eval.schemas import PairSample
from .base import DatasetAdapter
from .transforms import blur_regions, mask_regions


def _render_text_image(
    out_path: Path,
    text: str,
    image_size: int,
    text_pos: tuple[int, int] = (24, 24),
) -> tuple[int, int, int, int]:
    image = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    x, y = text_pos
    draw.text((x, y), text, fill="black", font=font)
    text_bbox = draw.textbbox((x, y), text, font=font)
    image.save(out_path)
    return text_bbox


def _load_source_records(source_path: Path) -> list[dict]:
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("TextVQA source file must be a JSON list.")
    return raw


def _normalize_short_answer(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def _majority_answer(answers: list[str]) -> str:
    normalized = [_normalize_short_answer(x) for x in answers if _normalize_short_answer(x)]
    if not normalized:
        return ""
    return Counter(normalized).most_common(1)[0][0]


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    slug = slug.strip("._-")
    return slug or "none"


class TextVQADatasetAdapter(DatasetAdapter):
    """Adapter for OCR-style short-answer tasks with removable evidence."""

    def _build_mock_subset(self, config: DatasetConfig, output_root: Path) -> list[dict]:
        image_dir = output_root / "images" / "textvqa_mock"
        image_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            ("STOP", "What word appears on the sign?"),
            ("MILK", "What product name is printed on the label?"),
            ("GATE 7", "What gate number is visible?"),
        ]
        subset: list[dict] = []
        for idx, (text, question) in enumerate(rows):
            sample_id = f"textvqa_mock_{idx:03d}"
            answer = text.lower()
            answerable = image_dir / f"{sample_id}__answerable__{_safe_slug(answer)}.png"
            box = _render_text_image(answerable, text=text, image_size=config.image_size, text_pos=(26, 36))
            subset.append(
                {
                    "id": sample_id,
                    "image_path": str(answerable),
                    "question": question,
                    "answer": answer,
                    "evidence_boxes": [list(box)],
                }
            )
        return subset

    def _apply_removal(
        self,
        image: Image.Image,
        boxes: list[tuple[int, int, int, int]],
        removal_mode: str,
    ) -> Image.Image:
        if removal_mode == "blur":
            return blur_regions(image, boxes)
        return mask_regions(image, boxes)

    def build_pairs(
        self,
        config: DatasetConfig,
        output_root: Path,
        seed: int,
    ) -> list[PairSample]:
        del seed
        source_path = Path(config.source_path) if config.source_path else None
        if source_path and source_path.exists():
            records = _load_source_records(source_path)
        else:
            records = self._build_mock_subset(config=config, output_root=output_root)

        if config.num_pairs > 0:
            records = records[: config.num_pairs]

        image_dir = output_root / "images" / "textvqa_pairs"
        image_dir.mkdir(parents=True, exist_ok=True)
        samples: list[PairSample] = []

        for idx, row in enumerate(records):
            pair_id = f"textvqa_pair_{idx:04d}"
            answers = row.get("answers")
            if not isinstance(answers, list):
                answers = []
            answers = [str(x).strip() for x in answers if str(x).strip()]
            if not answers:
                # Backward compatibility with older source schema.
                legacy = row.get("official_answers")
                if isinstance(legacy, list):
                    answers = [str(x).strip() for x in legacy if str(x).strip()]
            answer = str(row.get("answer", "")).strip() or _majority_answer(answers)
            if not answers and answer:
                answers = [answer]
            answer_slug = _safe_slug(answer)
            question = str(row["question"])
            image_path = Path(str(row["image_path"]))
            evidence_boxes_raw = row.get("evidence_boxes", [])
            evidence_boxes = [tuple(map(int, b)) for b in evidence_boxes_raw]
            if not image_path.exists():
                continue

            answerable_target = image_dir / f"{pair_id}__answerable__{answer_slug}.png"
            image = Image.open(image_path).convert("RGB")
            image = image.resize((config.image_size, config.image_size))
            image.save(answerable_target)
            hidden = self._apply_removal(image, evidence_boxes, config.evidence_removal)
            unanswerable_target = image_dir / f"{pair_id}__unanswerable__{answer_slug}.png"
            hidden.save(unanswerable_target)

            evidence_metadata = {
                "regions": [list(b) for b in evidence_boxes],
                "removal": config.evidence_removal,
                "source_kind": "textvqa",
            }
            metadata = {
                "original_id": row.get("id"),
                "source_path": str(source_path) if source_path else None,
                "answers": answers,
                "official_answers": answers,
            }

            samples.append(
                PairSample(
                    sample_id=f"{pair_id}_a",
                    pair_id=pair_id,
                    task="textvqa_ocr",
                    image_path=str(answerable_target),
                    question=question,
                    ground_truth_answerability=ANSWERABLE,
                    ground_truth_answer=answer,
                    evidence_metadata=evidence_metadata,
                    metadata=metadata,
                )
            )
            samples.append(
                PairSample(
                    sample_id=f"{pair_id}_u",
                    pair_id=pair_id,
                    task="textvqa_ocr",
                    image_path=str(unanswerable_target),
                    question=question,
                    ground_truth_answerability=UNANSWERABLE,
                    ground_truth_answer="",
                    evidence_metadata=evidence_metadata,
                    metadata=metadata,
                )
            )
        return samples
